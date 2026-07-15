"""ifeval_ja generation step (GPU, Issue #104): greedy-decode 100 prompts per model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from lfm25_ja.utils.config import load_config, merge_configs, project_root

logger = logging.getLogger(__name__)


def load_ifeval_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ifeval_ja config merged with base.yaml, matching load_eval_config
    in run_llm_jp_eval.py."""
    root = project_root()
    base = load_config(root / "configs" / "base.yaml")
    eval_path = Path(path) if path else root / "configs" / "eval" / "ifeval_ja.yaml"
    eval_cfg = load_config(eval_path)
    return merge_configs(base, eval_cfg)


def load_prompts(dataset_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    root = project_root()
    path = Path(dataset_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_generation_plan(
    cfg: dict[str, Any],
    models: list[str] | None = None,
    resume_from: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build the per-model generation plan: model name/path, output path, and
    the (possibly --limit-truncated) prompt count for that model.

    Tolerates a missing dataset file (num_prompts=None in that case) so
    ``--dry-run`` keeps working before ``datasets/eval/ifeval_ja/prompts.jsonl``
    has been added to the repo; a real (non-dry-run) generation run loads the
    dataset again and fails loudly if it is still missing."""
    eval_cfg = cfg["eval"]
    try:
        num_prompts: int | None = len(load_prompts(eval_cfg["dataset_path"], limit=limit))
    except FileNotFoundError:
        logger.warning(
            "dataset not found at %s; plan will omit prompt counts", eval_cfg["dataset_path"]
        )
        num_prompts = None
    output_dir = Path(eval_cfg.get("output_dir", "outputs/eval/ifeval_ja"))

    all_models = list(eval_cfg.get("models", []))
    if resume_from is not None:
        names = [m["name"] for m in all_models]
        if resume_from not in names:
            raise ValueError(f"--resume-from {resume_from!r} not found in configured models")
        all_models = all_models[names.index(resume_from) :]
    if models:
        wanted = set(models)
        all_models = [m for m in all_models if m["name"] in wanted]

    plan = []
    for model in all_models:
        plan.append(
            {
                "name": model["name"],
                "hf_path": model["hf_path"],
                "output_path": str(output_dir / model["name"] / "generations.jsonl"),
                "num_prompts": num_prompts,
            }
        )
    return plan


def _existing_generation_count(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    count = 0
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def generate_for_model(
    plan_item: dict[str, Any],
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Greedy-decode every prompt for one model and write generations.jsonl.
    Skips (idempotent) when the output already has as many lines as prompts,
    unless force=True."""
    output_path = Path(plan_item["output_path"])
    existing = _existing_generation_count(output_path)
    if not force and existing == len(prompts) and existing > 0:
        logger.info(
            "skip %s: %s already has %d/%d generations",
            plan_item["name"],
            output_path,
            existing,
            len(prompts),
        )
        return {"model": plan_item["name"], "status": "skipped", "count": existing}

    # Imported lazily so --dry-run and plan-building work without a GPU / the
    # transformers+torch stack installed (e.g. Windows dev box, CI).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_cfg = cfg["eval"]
    gen_cfg = eval_cfg.get("generation", {})
    apply_chat_template = bool(eval_cfg.get("apply_chat_template", True))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))

    logger.info("loading %s from %s", plan_item["name"], plan_item["hf_path"])
    tokenizer = AutoTokenizer.from_pretrained(plan_item["hf_path"])
    model = AutoModelForCausalLM.from_pretrained(
        plan_item["hf_path"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in prompts:
            prompt_text = row["prompt"]
            if apply_chat_template:
                # return_dict=False: newer transformers default apply_chat_template(tokenize=True)
                # to BatchEncoding, breaking .to(device) + model.generate() (see format_chat.py:54).
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                ).to(model.device)
            else:
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    use_cache=True,
                )
            response_ids = output_ids[0][input_ids.shape[-1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "prompt": prompt_text,
                        "response": response,
                        "instruction_id_list": row["instruction_id_list"],
                        "kwargs": row["kwargs"],
                        "category": row["category"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"model": plan_item["name"], "status": "generated", "count": len(prompts)}


def run_generation(
    config_path: str | Path | None = None,
    models: list[str] | None = None,
    resume_from: str | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = load_ifeval_config(config_path)
    plan = build_generation_plan(cfg, models=models, resume_from=resume_from, limit=limit)

    if dry_run:
        return {"status": "dry_run", "plan": plan}

    prompts = load_prompts(cfg["eval"]["dataset_path"], limit=limit)
    results = []
    for item in plan:
        results.append(generate_for_model(item, prompts, cfg, force=force))
    return {"status": "executed", "plan": plan, "runs": results}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate ifeval_ja responses (Issue #104)")
    parser.add_argument("--config", default=None, help="ifeval_ja eval config YAML path")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names to run")
    parser.add_argument("--resume-from", default=None, help="Resume starting at this model name")
    parser.add_argument("--limit", type=int, default=None, help="Only generate first N prompts")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only")
    args = parser.parse_args()

    results = run_generation(
        config_path=args.config,
        models=args.models,
        resume_from=args.resume_from,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"status={results['status']}")
    for item in results["plan"]:
        print(f"  [{item['name']}] hf_path={item['hf_path']} num_prompts={item['num_prompts']}")
    for run in results.get("runs", []):
        print(f"  -> {run['model']}: {run['status']} ({run['count']} generations)")


if __name__ == "__main__":
    main()
