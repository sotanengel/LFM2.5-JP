"""dpo-001 base on-policy K-sample generation (Issue #115, Phase G, GPU).

Samples the *base* model (``LiquidAI/LFM2.5-1.2B-JP-202606``) K times per
prompt-pool row (:mod:`lfm25_ja.data.pref_prompts`) with temperature>0 --
these on-policy generations are what Phase V (rule verdicts,
:mod:`lfm25_ja.data.pref_verify`) and Phase J (LLM judge,
:mod:`lfm25_ja.eval.judge_swallow`) later label, and Phase P
(:mod:`lfm25_ja.data.pref_pairs`) pairs into chosen/rejected DPO rows.

Output (``generations.jsonl``, one row per sample):
``{"prompt_id", "k", "prompt", "response"}``.

Idempotency (WSL2-restart-safe): existing (prompt_id, k) keys are counted on
startup and only the still-missing samples are generated and *appended* --
rerunning the same command resumes where the previous run died. Resumed
samples are drawn fresh (sampling is not keyed to (prompt_id, k)); that's
fine for preference-pair building, which never assumes sample k is
reproducible, only that it is on-policy.

Batching: prompts are rendered via ``apply_chat_template(...,
tokenize=False)`` and batch-encoded with left padding (the
``BatchEncoding``/``return_dict`` trap documented in docs/agent_ops.md only
bites the tokenize=True path). Pilot-measured throughput on the RTX 3060 Ti:
~340 tok/s at batch 16, VRAM peak 2.3 GiB.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)


def _existing_generation_keys(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add((row["prompt_id"], row["k"]))
    return keys


def build_generation_plan(
    prompts: list[dict[str, Any]],
    num_samples: int,
    existing_keys: set[tuple[str, int]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Expand the prompt pool into the flat list of still-missing
    (prompt_id, k) work items. ``limit`` truncates the *prompt pool* (not the
    work items) so a limited smoke run generates all K samples for the first
    N prompts rather than k=0 only for K*N prompts."""
    if limit is not None:
        prompts = prompts[:limit]
    plan = []
    for row in prompts:
        for k in range(num_samples):
            if (row["id"], k) in existing_keys:
                continue
            plan.append({"prompt_id": row["id"], "k": k, "prompt": row["prompt"]})
    return plan


def run_generation(
    config_path: str | Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Phase G as described by ``config_path`` (see
    ``configs/data/dpo_pairs_001.yaml``'s ``pref_generate`` section):
    plan the missing (prompt_id, k) samples, then batch-sample the base
    model, appending each finished batch to ``output_path`` so a mid-run
    crash loses at most one batch. ``--dry-run`` reports the plan without
    importing torch or touching a GPU."""
    config = load_config(config_path)
    cfg = config.get("pref_generate", config)

    prompts_path = cfg["prompts_path"]
    output_path = Path(cfg["output_path"])
    model_path = cfg.get("model_path", "LiquidAI/LFM2.5-1.2B-JP-202606")
    num_samples = int(cfg.get("num_samples", 4))
    temperature = float(cfg.get("temperature", 0.8))
    top_p = float(cfg.get("top_p", 0.95))
    max_new_tokens = int(cfg.get("max_new_tokens", 384))
    batch_size = int(cfg.get("batch_size", 16))

    prompts = _read_jsonl(prompts_path)
    existing_keys = _existing_generation_keys(output_path)
    plan = build_generation_plan(prompts, num_samples, existing_keys, limit=limit)

    if dry_run:
        return {
            "status": "dry_run",
            "model_path": model_path,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "total_prompts": len(prompts) if limit is None else min(limit, len(prompts)),
            "already_generated": len(existing_keys),
            "pending": len(plan),
        }

    if not plan:
        logger.info("nothing pending: %d samples already generated", len(existing_keys))
        return {"status": "skipped", "generated": 0, "already_generated": len(existing_keys)}

    # Imported lazily so plan-building and --dry-run work without the GPU
    # stack (same pattern as generate_ifeval_ja / judge_swallow).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading base model %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    with output_path.open("a", encoding="utf-8") as f:
        for batch_start in range(0, len(plan), batch_size):
            batch = plan[batch_start : batch_start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["prompt"]}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for item in batch
            ]
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            input_len = encoded["input_ids"].shape[-1]
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                )
            for i, item in enumerate(batch):
                response = tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
                f.write(
                    json.dumps(
                        {
                            "prompt_id": item["prompt_id"],
                            "k": item["k"],
                            "prompt": item["prompt"],
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                generated += 1
            f.flush()
            if (batch_start // batch_size) % 20 == 0:
                logger.info("generated %d/%d pending samples", generated, len(plan))

    return {"status": "executed", "generated": generated, "pending_was": len(plan)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="dpo-001 base on-policy K-sample generation (Issue #115)"
    )
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_001.yaml",
        help="Path to configs/data/dpo_pairs_001.yaml",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only generate for the first N pool prompts"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only")
    args = parser.parse_args()

    result = run_generation(args.config, limit=args.limit, dry_run=args.dry_run)
    print(result)


if __name__ == "__main__":
    main()
