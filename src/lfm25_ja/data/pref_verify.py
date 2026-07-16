"""dpo-001 rule-based verdicts for base on-policy samples (Issue #115, Phase V, CPU).

Each (prompt, response) sample produced by Phase G
(:mod:`lfm25_ja.data.pref_generate`) gets a strict pass/fail verdict against
its prompt's ``category``/``constraint_detail``, plus a degenerate-output
flag. Dispatch mirrors ``distill_select.select_row``'s per-category logic,
but deliberately differs in two ways:

- No tight length-margin band is applied to char_count/compound: DPO's
  ``chosen`` only needs a strict rule pass (Issue #115 spec explicitly says
  the tight margin is not required here; whether a sample sits inside the
  60-90%-of-max band the sft-005 distill pipeline required is recorded by
  the caller from ``constraint_detail`` if needed, not gated here).
- Operates on a prompt row + a separately-supplied response string (Phase G's
  generations are stored independently of the prompt pool), not on a single
  CSV row that already carries its own response.

Idempotency: this module recomputes verdicts for every generation on every
run (CPU, seconds even at pool scale) rather than skip-if-exists -- cheap
enough that resuming after a WSL2 restart is just "run it again".
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.data.distill_select import (
    _REPETITION_EXEMPT_CATEGORIES,
    _has_repetition,
    verify_format_json_detail,
    verify_no_constraint,
    verify_paragraph_count,
    verify_start_word,
)
from lfm25_ja.eval.instruction_verifiers import (
    verify_bullet_count,
    verify_char_count,
    verify_keyword,
    verify_polite_form,
)
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

_MIN_RESPONSE_CHARS = 20


def _check_degenerate(category: str, response: str) -> tuple[bool, str]:
    stripped = response.strip()
    if not stripped:
        return True, "空の応答です"
    if len(stripped) < _MIN_RESPONSE_CHARS:
        return True, f"応答が短すぎます({len(stripped)}字)"
    if category not in _REPETITION_EXEMPT_CATEGORIES and _has_repetition(response):
        return True, "反復が検出されました"
    return False, ""


def _dispatch(category: str, detail: dict[str, Any], response: str) -> tuple[bool, str]:
    if category == "char_count":
        return verify_char_count(response, {"max": detail.get("max"), "min": detail.get("min")})

    if category == "compound":
        ok, msg = verify_char_count(response, {"max": detail.get("max")})
        if not ok:
            return False, msg
        return verify_keyword(response, {"include": detail.get("include") or []})

    if category == "bullet_count":
        return verify_bullet_count(response, {"count": detail.get("count")})

    if category == "format_json":
        return verify_format_json_detail(response, detail)

    if category == "keyword_include":
        return verify_keyword(response, {"include": detail.get("include") or []})

    if category == "paragraph_count":
        return verify_paragraph_count(response, detail)

    if category == "forbidden_word":
        return verify_keyword(response, {"exclude": detail.get("exclude") or []})

    if category == "start_word":
        return verify_start_word(response, detail)

    if category == "polite_form":
        return verify_polite_form(response, {"style": detail.get("style", "polite")})

    if category == "no_constraint":
        return verify_no_constraint(response, {})

    return False, f"未知の category です: {category!r}"


def verify_sample(prompt_row: dict[str, Any], response: str) -> dict[str, Any]:
    """Return the rule-verdict + degenerate-flag dict for one (prompt,
    response) pair. ``prompt_row`` must have ``category`` and
    ``constraint_detail`` (a dict, as produced by
    :mod:`lfm25_ja.data.pref_prompts`)."""
    category = prompt_row["category"]
    detail = prompt_row.get("constraint_detail") or {}
    response = response or ""

    degenerate, degenerate_reason = _check_degenerate(category, response)
    rule_pass, rule_reason = _dispatch(category, detail, response)

    return {
        "rule_pass": rule_pass,
        "rule_reason": rule_reason,
        "degenerate": degenerate,
        "degenerate_reason": degenerate_reason,
    }


def verify_generations(
    prompts_by_id: dict[str, dict[str, Any]], generations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verify every generation row (each having ``prompt_id``/``k``/
    ``response``) against its prompt's constraint, returning one verdict row
    per generation (schema: ``prompt_id``/``k``/``rule_pass``/``rule_reason``/
    ``degenerate``/``degenerate_reason``)."""
    verdicts = []
    for gen in generations:
        prompt_row = prompts_by_id.get(gen["prompt_id"])
        if prompt_row is None:
            raise KeyError(f"unknown prompt_id: {gen['prompt_id']!r}")
        verdict = verify_sample(prompt_row, gen.get("response", ""))
        verdicts.append({"prompt_id": gen["prompt_id"], "k": gen["k"], **verdict})
    return verdicts


def run_verification(config_path: str | Path) -> dict[str, Any]:
    """Run the end-to-end verification described by ``config_path`` (see
    ``configs/data/dpo_pairs_001.yaml``'s ``pref_verify`` section): load the
    prompt pool and generations, verify every sample, and write
    ``verdicts.jsonl``."""
    config = load_config(config_path)
    cfg = config.get("pref_verify", config)

    prompts_path = cfg["prompts_path"]
    generations_path = cfg["generations_path"]
    output_path = cfg["output_path"]

    prompts = _read_jsonl(prompts_path)
    prompts_by_id = {p["id"]: p for p in prompts}
    generations = _read_jsonl(generations_path)

    verdicts = verify_generations(prompts_by_id, generations)
    _write_jsonl(output_path, verdicts)

    result = {
        "total": len(verdicts),
        "rule_pass": sum(1 for v in verdicts if v["rule_pass"]),
        "degenerate": sum(1 for v in verdicts if v["degenerate"]),
        "output_path": str(output_path),
    }
    logger.info(
        "dpo-001 verdicts written: %d samples (%d rule_pass, %d degenerate) -> %s",
        result["total"],
        result["rule_pass"],
        result["degenerate"],
        output_path,
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Verify dpo-001 base on-policy samples against rule verifiers (Issue #115)"
    )
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_001.yaml",
        help="Path to configs/data/dpo_pairs_001.yaml",
    )
    args = parser.parse_args()

    result = run_verification(args.config)
    print(
        f"total={result['total']} rule_pass={result['rule_pass']} "
        f"degenerate={result['degenerate']} -> {result['output_path']}"
    )


if __name__ == "__main__":
    main()
