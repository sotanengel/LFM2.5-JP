"""JKB v1 CLI: generate model completions against datasets/eval/jkb, then score (Issue #121)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from lfm25_ja.eval import jkb
from lfm25_ja.eval.japan_probe import BASE_MODEL, FEWSHOT, _parse_model_specs

logger = logging.getLogger(__name__)


def format_prompt(row: dict[str, Any], fewshot: str) -> str:
    """Build the 1-shot prompt for a row, appending an A/B/... choice list for mcq rows."""
    if row["format"] == "mcq":
        choice_lines = "\n".join(f"{c['label']}: {c['text']}" for c in row["choices"])
        return f"{fewshot}質問: {row['prompt']}\n{choice_lines}\n答え:"
    return f"{fewshot}質問: {row['prompt']}\n答え:"


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def generate_for_model(
    label: str,
    path: str,
    rows: list[dict[str, Any]],
    out_dir: Path,
    tokenizer: Any,
    max_new_tokens: int,
    fewshot: str,
    regenerate: bool,
) -> Path:
    """Greedy-decode every row for one model and write generations.jsonl (idempotent)."""
    model_out_dir = out_dir / label
    model_out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = model_out_dir / "generations.jsonl"

    wanted_ids = {row["id"] for row in rows}
    if not regenerate and wanted_ids <= _existing_ids(gen_path):
        logger.info("skip generation for %s: %s already complete", label, gen_path)
        return gen_path

    import torch
    from transformers import AutoModelForCausalLM

    logger.info("loading %s from %s", label, path)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"Failed to load model {label!r} from {path!r}: {exc}") from exc
    model.eval()

    with gen_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            prompt = format_prompt(row, fewshot)
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                )
            response = tokenizer.decode(
                out[0][ids.input_ids.shape[1] :], skip_special_tokens=True
            )
            f.write(
                json.dumps(
                    {"id": row["id"], "prompt": prompt, "response": response}, ensure_ascii=False
                )
                + "\n"
            )
            logger.info("[%s] %d/%d %s: %s", label, i + 1, len(rows), row["id"], response[:60])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gen_path


def score_model(label: str, rows: list[dict[str, Any]], gen_path: Path, out_dir: Path) -> dict:
    """Score one model's generations.jsonl against rows and write scored.jsonl + report.md."""
    raw_texts: dict[str, str] = {}
    with gen_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            raw_texts[entry["id"]] = entry["response"]

    agg = jkb.aggregate(rows, raw_texts)

    model_out_dir = out_dir / label
    scored_path = model_out_dir / "scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for pr in agg["per_row"]:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    report = jkb.render_report_markdown(agg, label)
    (model_out_dir / "report.md").write_text(report, encoding="utf-8")
    gate_agg = {"overall": agg["overall"], "by_domain": agg["by_domain"]}
    (model_out_dir / "aggregate.json").write_text(
        json.dumps(gate_agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return agg


def write_summary(results: dict[str, dict], out_dir: Path) -> None:
    """Write outputs/eval/jkb/summary.md comparing all models on overall + per-domain accuracy."""
    lines = ["# JKB v1 summary", ""]
    header = ["model", "overall", *jkb.JKB_DOMAINS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for label, agg in results.items():
        overall = agg["overall"]
        cells = [label, f"{overall['accuracy'] * 100:.1f}% (n={overall['n']})"]
        for domain in jkb.JKB_DOMAINS:
            stat = agg["by_domain"].get(domain)
            cells.append(f"{stat['accuracy'] * 100:.1f}%" if stat else "-")
        lines.append("| " + " | ".join(cells) + " |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="JKB v1 runner (Issue #121)")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="label=path pairs, e.g. base=LiquidAI/LFM2.5-1.2B-Base "
        "ckpt9000=outputs/.../checkpoint-9000",
    )
    parser.add_argument("--dataset", default="datasets/eval/jkb/eval.jsonl")
    parser.add_argument("--out-dir", default="outputs/eval/jkb")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--fewshot", default=FEWSHOT)
    parser.add_argument(
        "--regenerate", action="store_true", help="Regenerate even if generations.jsonl is complete"
    )
    args = parser.parse_args()

    model_specs = _parse_model_specs(args.models)
    rows = jkb.load_jkb_jsonl(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"Failed to load tokenizer from {BASE_MODEL!r}: {exc}") from exc

    results: dict[str, dict] = {}
    for label, path in model_specs:
        gen_path = generate_for_model(
            label,
            path,
            rows,
            out_dir,
            tokenizer,
            args.max_new_tokens,
            args.fewshot,
            args.regenerate,
        )
        results[label] = score_model(label, rows, gen_path, out_dir)
        print(f"[{label}] overall accuracy = {results[label]['overall']['accuracy'] * 100:.1f}%")

    write_summary(results, out_dir)
    print(f"JKB v1 report written to {out_dir}")


if __name__ == "__main__":
    main()
