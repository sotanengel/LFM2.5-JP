"""K3 factual DPO: JKB train-split preference prompt pool (Issue #124 / #145).

Builds the on-policy prompt pool for K3 factual DPO from
``datasets/eval/jkb/train.jsonl`` only. Eval split IDs must never appear
in the pool (physical separation for K3 training vs K3 gate evaluation).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _write_jsonl
from lfm25_ja.eval.japan_probe import FEWSHOT
from lfm25_ja.eval.jkb import load_jkb_jsonl
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)


def format_jkb_prompt(row: dict[str, Any], fewshot: str = FEWSHOT) -> str:
    """Build the 1-shot JKB prompt (same contract as scripts/run_jkb.py)."""
    if row["format"] == "mcq":
        choice_lines = "\n".join(f"{c['label']}: {c['text']}" for c in row["choices"])
        return f"{fewshot}質問: {row['prompt']}\n{choice_lines}\n答え:"
    return f"{fewshot}質問: {row['prompt']}\n答え:"


def jkb_row_to_pool_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one JKB train row into a K3 pref pool row."""
    return {
        "id": f"k3pref-{index:05d}",
        "category": "jkb_fact",
        "instruction_id_list": "",
        "constraint_detail": {
            "jkb_id": row["id"],
            "format": row["format"],
            "answers": row["answers"],
            "choices": row["choices"],
            "correct_choice": row["correct_choice"],
            "source_quote": row["source_quote"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        },
        "topic": row["domain"],
        "prompt": format_jkb_prompt(row),
    }


def check_jkb_pool_non_duplication(
    pool: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Hard gate: no pool row's jkb_id may appear in the eval split."""
    eval_ids = {r["id"] for r in eval_rows}
    pool_ids = {r["constraint_detail"]["jkb_id"] for r in pool}
    overlap = sorted(pool_ids & eval_ids)
    if overlap:
        raise ValueError(
            "jkb prompt pool non-duplication violated: eval id overlap: "
            f"{overlap[:5]}{'...' if len(overlap) > 5 else ''}"
        )
    eval_prompts = {r["prompt"] for r in eval_rows}
    prompt_hits = sorted(
        r["constraint_detail"]["jkb_id"]
        for r in pool
        if any(ep in r["prompt"] for ep in eval_prompts)
    )
    return {
        "eval_ids": len(eval_ids),
        "pool_jkb_ids": len(pool_ids),
        "id_overlap": overlap,
        "prompt_hits": prompt_hits,
    }


def build_jkb_prompt_pool(
    *,
    train_path: str | Path,
    eval_path: str | Path,
    output_path: str | Path,
    stats_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build K3 pref pool from JKB train split and write JSONL."""
    train_rows = load_jkb_jsonl(train_path)
    eval_rows = load_jkb_jsonl(eval_path)
    pool = [jkb_row_to_pool_row(row, index=i + 1) for i, row in enumerate(train_rows)]
    dup = check_jkb_pool_non_duplication(pool, eval_rows)
    _write_jsonl(output_path, pool)

    stats: dict[str, Any] = {
        "total": len(pool),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "output_path": str(output_path),
        "domain_counts": dict(Counter(r["topic"] for r in pool)),
        "difficulty_counts": dict(
            Counter(r["constraint_detail"]["difficulty"] for r in pool)
        ),
        "non_duplication": dup,
    }

    if stats_report_path:
        lines = [
            "# K3 factual DPO prompt pool stats (Issue #124)",
            "",
            f"- Total prompts: {stats['total']}",
            f"- Train source: {stats['train_path']}",
            f"- Eval guard: {stats['eval_path']}",
            f"- Output: {stats['output_path']}",
            "",
            "## Domain counts",
            "",
        ]
        for domain, count in sorted(stats["domain_counts"].items()):
            lines.append(f"- {domain}: {count}")
        lines.append("")
        Path(stats_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["report_path"] = str(stats_report_path)

    logger.info("K3 JKB prompt pool: %d rows -> %s", len(pool), output_path)
    return stats


def build_prompt_pool(config_path: str | Path) -> dict[str, Any]:
    """Orchestrator entry matching dpo-001 config section ``pref_prompts_jkb``."""
    config = load_config(config_path)
    cfg = config.get("pref_prompts_jkb", config)
    return build_jkb_prompt_pool(
        train_path=cfg["train_path"],
        eval_path=cfg["eval_path"],
        output_path=cfg["output_path"],
        stats_report_path=cfg.get("stats_report"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build K3 JKB train-split pref pool")
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_k3_facts.yaml",
        help="Path to configs/data/dpo_pairs_k3_facts.yaml",
    )
    args = parser.parse_args()
    result = build_prompt_pool(args.config)
    logger.info("Done: %d prompts -> %s", result["total"], result["output_path"])


if __name__ == "__main__":
    main()
