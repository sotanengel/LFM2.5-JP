"""K3 factual rule verification for on-policy samples (Issue #124 / #145).

Phase V for K3: each generation is scored against the JKB reference answer
via substring matching (jkb.score_row). Degenerate outputs are flagged.
Factual quality ranking is delegated to Phase J (Swallow factual judge).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.data.pref_verify import _check_degenerate
from lfm25_ja.eval.jkb import score_row
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

_ESCAPE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"わかりません",
        r"分かりません",
        r"不明です",
        r"答え(?:られ|が)ません",
        r"お答え(?:でき|がで)きません",
    )
)


def _detail_to_jkb_row(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": detail["jkb_id"],
        "format": detail["format"],
        "answers": detail.get("answers") or [],
        "choices": detail.get("choices"),
        "correct_choice": detail.get("correct_choice"),
    }


def is_escape_response(response: str) -> bool:
    text = response.strip()
    if not text:
        return False
    return any(p.search(text) for p in _ESCAPE_PATTERNS)


def verify_sample(prompt_row: dict[str, Any], response: str) -> dict[str, Any]:
    """Return factual verdict dict for one (prompt, response) pair."""
    detail = prompt_row.get("constraint_detail") or {}
    response = response or ""
    degenerate, degenerate_reason = _check_degenerate(prompt_row["category"], response)
    jkb_row = _detail_to_jkb_row(detail)
    fact_correct = score_row(jkb_row, response)
    escape = is_escape_response(response)
    rule_pass = fact_correct and not escape
    rule_reason = ""
    if escape:
        rule_reason = "escape_response"
    elif not fact_correct:
        rule_reason = "fact_incorrect"
    else:
        rule_reason = "fact_correct"
    return {
        "rule_pass": rule_pass,
        "rule_reason": rule_reason,
        "degenerate": degenerate,
        "degenerate_reason": degenerate_reason,
        "fact_correct": fact_correct,
        "escape": escape,
    }


def verify_generations(
    prompts_by_id: dict[str, dict[str, Any]], generations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    verdicts = []
    for gen in generations:
        prompt_row = prompts_by_id.get(gen["prompt_id"])
        if prompt_row is None:
            raise KeyError(f"unknown prompt_id: {gen['prompt_id']!r}")
        verdict = verify_sample(prompt_row, gen.get("response", ""))
        verdicts.append({"prompt_id": gen["prompt_id"], "k": gen["k"], **verdict})
    return verdicts


def run_verification(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    cfg = config.get("pref_verify_facts", config)

    prompts = _read_jsonl(cfg["prompts_path"])
    prompts_by_id = {p["id"]: p for p in prompts}
    generations = _read_jsonl(cfg["generations_path"])
    verdicts = verify_generations(prompts_by_id, generations)
    _write_jsonl(cfg["output_path"], verdicts)

    result = {
        "total": len(verdicts),
        "rule_pass": sum(1 for v in verdicts if v["rule_pass"]),
        "fact_correct": sum(1 for v in verdicts if v["fact_correct"]),
        "escape": sum(1 for v in verdicts if v["escape"]),
        "degenerate": sum(1 for v in verdicts if v["degenerate"]),
        "output_path": str(cfg["output_path"]),
    }
    logger.info(
        "K3 factual verdicts: %d samples (%d rule_pass) -> %s",
        result["total"],
        result["rule_pass"],
        cfg["output_path"],
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Verify K3 factual on-policy samples")
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_k3_facts.yaml",
    )
    args = parser.parse_args()
    result = run_verification(args.config)
    print(result)


if __name__ == "__main__":
    main()
