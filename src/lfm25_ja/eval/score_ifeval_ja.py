"""ifeval_ja scoring step (CPU-only, idempotent, Issue #104)."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from lfm25_ja.eval.generate_ifeval_ja import load_ifeval_config
from lfm25_ja.eval.instruction_verifiers import VERIFIERS, strip_preamble

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_generation(row: dict[str, Any]) -> dict[str, Any]:
    """Score one generated response strict (raw) and loose (preamble-stripped)
    against every instruction_id listed for its prompt."""
    response = row["response"]
    loose_response = strip_preamble(response)
    instruction_results: dict[str, dict[str, Any]] = {}
    for instruction_id in row["instruction_id_list"]:
        verifier = VERIFIERS[instruction_id]
        params = row["kwargs"].get(instruction_id, {})
        strict_pass, strict_reason = verifier(response, params)
        loose_pass, loose_reason = verifier(loose_response, params)
        instruction_results[instruction_id] = {
            "strict": strict_pass,
            "strict_reason": strict_reason,
            "loose": loose_pass,
            "loose_reason": loose_reason,
        }

    strict_pass_all = all(r["strict"] for r in instruction_results.values())
    loose_pass_all = all(r["loose"] for r in instruction_results.values())
    return {
        "id": row["id"],
        "category": row["category"],
        "strict_pass": strict_pass_all,
        "loose_pass": loose_pass_all,
        "instruction_results": instruction_results,
    }


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    num_prompts = len(scores)
    prompt_strict_acc = (
        sum(1 for s in scores if s["strict_pass"]) / num_prompts if num_prompts else 0.0
    )
    prompt_loose_acc = (
        sum(1 for s in scores if s["loose_pass"]) / num_prompts if num_prompts else 0.0
    )

    total_instructions = 0
    strict_instruction_passes = 0
    loose_instruction_passes = 0
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"pass": 0, "count": 0})
    by_verifier: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"strict_pass": 0, "loose_pass": 0, "count": 0}
    )

    for s in scores:
        cat = by_category[s["category"]]
        cat["count"] += 1
        if s["strict_pass"]:
            cat["pass"] += 1
        for instruction_id, result in s["instruction_results"].items():
            total_instructions += 1
            if result["strict"]:
                strict_instruction_passes += 1
            if result["loose"]:
                loose_instruction_passes += 1
            v = by_verifier[instruction_id]
            v["count"] += 1
            if result["strict"]:
                v["strict_pass"] += 1
            if result["loose"]:
                v["loose_pass"] += 1

    instruction_strict_acc = (
        strict_instruction_passes / total_instructions if total_instructions else 0.0
    )
    instruction_loose_acc = (
        loose_instruction_passes / total_instructions if total_instructions else 0.0
    )

    return {
        "num_prompts": num_prompts,
        "num_instructions": total_instructions,
        "prompt_strict_acc": prompt_strict_acc,
        "prompt_loose_acc": prompt_loose_acc,
        "instruction_strict_acc": instruction_strict_acc,
        "instruction_loose_acc": instruction_loose_acc,
        "by_category": {
            cat: {
                "prompt_strict_acc": v["pass"] / v["count"] if v["count"] else 0.0,
                "count": v["count"],
            }
            for cat, v in sorted(by_category.items())
        },
        "by_verifier": {
            vid: {
                "instruction_strict_acc": v["strict_pass"] / v["count"] if v["count"] else 0.0,
                "instruction_loose_acc": v["loose_pass"] / v["count"] if v["count"] else 0.0,
                "count": v["count"],
            }
            for vid, v in sorted(by_verifier.items())
        },
    }


def score_model(model_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = cfg["eval"]
    output_dir = Path(eval_cfg.get("output_dir", "outputs/eval/ifeval_ja"))
    model_dir = output_dir / model_name
    generations_path = model_dir / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(
            f"No generations found for {model_name!r} at {generations_path}; "
            "run generate_ifeval_ja first"
        )

    generations = _load_jsonl(generations_path)
    scores = [score_generation(row) for row in generations]
    aggregate = aggregate_scores(scores)
    aggregate["model"] = model_name

    scores_path = model_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    aggregate_path = model_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return aggregate


def run_scoring(
    config_path: str | Path | None = None,
    models: list[str] | None = None,
) -> list[dict[str, Any]]:
    cfg = load_ifeval_config(config_path)
    all_models = [m["name"] for m in cfg["eval"].get("models", [])]
    target_models = models if models else all_models

    results = []
    for name in target_models:
        results.append(score_model(name, cfg))
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'model':<24}{'prompt_strict':>14}{'prompt_loose':>14}"
        f"{'instr_strict':>14}{'instr_loose':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<24}"
            f"{r['prompt_strict_acc']:>14.3f}"
            f"{r['prompt_loose_acc']:>14.3f}"
            f"{r['instruction_strict_acc']:>14.3f}"
            f"{r['instruction_loose_acc']:>14.3f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Score ifeval_ja generations (Issue #104)")
    parser.add_argument("--config", default=None, help="ifeval_ja eval config YAML path")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names to score")
    args = parser.parse_args()

    results = run_scoring(config_path=args.config, models=args.models)
    print_summary(results)


if __name__ == "__main__":
    main()
