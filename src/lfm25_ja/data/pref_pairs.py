"""dpo-001 preference pair construction + hard gates (Issue #115, Phase P, CPU).

Merges Phase G generations, Phase V rule verdicts, and Phase J judge scores
per prompt, then selects one chosen/rejected pair per prompt (at most --
prompts with no eligible chosen or no eligible rejected sample are skipped,
counted in ``skip_reasons``):

chosen
    rule-pass AND non-degenerate AND has a (non-null) judge score. Among
    those, max judge score; ties broken by *shorter* response (matches
    base's own ~176-char natural length, the same rationale as
    ``distill_select``'s response-length guard). ``polite_form`` prompts
    additionally require ``score >= polite_min_score`` (config, default 3) --
    a bare rule pass on polite_form doesn't guarantee the business-writing
    register is actually good, so a quality floor is enforced even for
    ``chosen``.

rejected
    rule-fail AND non-degenerate. A null judge score is allowed here (unlike
    chosen) since a clean-but-unscored failing sample is still a usable
    negative; among scored candidates, max judge score is preferred (a
    well-written-but-rule-violating response makes a *harder* contrastive
    pair than a garbled one); ties (or an all-null pool) are broken by
    *smallest* :func:`_violation_margin` -- e.g. the char_count sample that
    overshoots ``max`` by the fewest characters -- again favoring the
    hardest, most-instructive negative.

Two hard gates run after pairing, mirroring ``distill_select``'s pattern of
reusing the *same* gate functions across pipeline stages as defense in
depth:

(a) chosen response length guard (``distill_select.check_response_length_guard``,
    reused as-is against ``{"response": pair["chosen"]}`` rows) -- mean
    chosen length must fall within ``base_mean_chars * (1 +/- tolerance)``.
(b) eval non-duplication (``distill_select.check_eval_non_duplication``,
    reused as-is against the *paired* prompts' category/detail/topic) --
    should be a no-op given Phase P0 already enforced this on the whole pool,
    but re-asserted here since the paired subset is what actually reaches
    the trainer.

Output schema (``dpo_pairs.jsonl``, one row per pair):
``{"prompt": str, "chosen": str, "rejected": str, "meta": {...}}`` -- the TRL
``DPOTrainer`` preference-pair contract.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.data.distill_select import check_eval_non_duplication, check_response_length_guard
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_POLITE_MIN_SCORE = 3


# ---------------------------------------------------------------------------
# Per-prompt sample assembly
# ---------------------------------------------------------------------------


def merge_samples(
    prompt_id: str,
    generations_by_prompt: dict[str, list[dict[str, Any]]],
    verdicts_by_key: dict[tuple[str, int], dict[str, Any]],
    judgments_by_key: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join one prompt's generations with their rule verdicts and (if judged)
    judge scores into a flat per-sample dict: ``k``/``response``/
    ``rule_pass``/``rule_reason``/``degenerate``/``score``/``judge_reason``.
    A sample with no matching verdict defaults to ``rule_pass=False``
    (fail-closed); a sample with no matching judgment gets ``score=None``."""
    samples = []
    for gen in generations_by_prompt.get(prompt_id, []):
        key = (gen["prompt_id"], gen["k"])
        verdict = verdicts_by_key.get(key, {})
        judgment = judgments_by_key.get(key, {})
        samples.append(
            {
                "k": gen["k"],
                "response": gen.get("response", ""),
                "rule_pass": verdict.get("rule_pass", False),
                "rule_reason": verdict.get("rule_reason", ""),
                "degenerate": verdict.get("degenerate", False),
                "score": judgment.get("score"),
                "judge_reason": judgment.get("reason", ""),
            }
        )
    return samples


# ---------------------------------------------------------------------------
# rejected tie-break: distance from constraint satisfaction
# ---------------------------------------------------------------------------


def _violation_margin(prompt_row: dict[str, Any], response: str) -> float:
    """Numeric "how far from passing" distance, used only to tie-break among
    rejected candidates -- smaller is a "closer miss" (preferred: it makes
    the harder negative). Only meaningful for char_count/compound, where a
    length delta is well-defined; every other category returns 0.0 (no
    signal, so earlier candidates in iteration order win any remaining tie)."""
    category = prompt_row["category"]
    if category not in ("char_count", "compound"):
        return 0.0
    detail = prompt_row.get("constraint_detail") or {}
    length = len(response)
    margin = 0.0
    mx = detail.get("max")
    if mx is not None and length > mx:
        margin += length - mx
    mn = detail.get("min")
    if mn is not None and length < mn:
        margin += mn - length
    return margin


# ---------------------------------------------------------------------------
# chosen / rejected selection
# ---------------------------------------------------------------------------


def select_chosen(
    prompt_row: dict[str, Any],
    samples: list[dict[str, Any]],
    polite_min_score: int = _DEFAULT_POLITE_MIN_SCORE,
) -> dict[str, Any] | None:
    candidates = [
        s for s in samples if s["rule_pass"] and not s["degenerate"] and s.get("score") is not None
    ]
    if prompt_row["category"] == "polite_form":
        candidates = [s for s in candidates if s["score"] >= polite_min_score]
    if not candidates:
        return None
    max_score = max(s["score"] for s in candidates)
    candidates = [s for s in candidates if s["score"] == max_score]
    candidates.sort(key=lambda s: len(s["response"]))
    return candidates[0]


def select_rejected(
    prompt_row: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidates = [s for s in samples if not s["rule_pass"] and not s["degenerate"]]
    if not candidates:
        return None

    scored = [s for s in candidates if s.get("score") is not None]
    pool = scored if scored else candidates
    if scored:
        max_score = max(s["score"] for s in pool)
        pool = [s for s in pool if s["score"] == max_score]
    pool.sort(key=lambda s: _violation_margin(prompt_row, s["response"]))
    return pool[0]


def build_pair(
    prompt_row: dict[str, Any],
    samples: list[dict[str, Any]],
    polite_min_score: int = _DEFAULT_POLITE_MIN_SCORE,
) -> tuple[dict[str, Any] | None, str]:
    """Return ``(pair_or_None, status)``. ``status`` is ``"ok"`` on success,
    else one of ``"no_chosen_candidate"``/``"no_rejected_candidate"``."""
    chosen = select_chosen(prompt_row, samples, polite_min_score=polite_min_score)
    if chosen is None:
        return None, "no_chosen_candidate"

    rejected = select_rejected(prompt_row, samples)
    if rejected is None:
        return None, "no_rejected_candidate"

    pair = {
        "prompt": prompt_row["prompt"],
        "chosen": chosen["response"],
        "rejected": rejected["response"],
        "meta": {
            "prompt_id": prompt_row["id"],
            "category": prompt_row["category"],
            "chosen_k": chosen["k"],
            "chosen_score": chosen["score"],
            "rejected_k": rejected["k"],
            "rejected_score": rejected.get("score"),
            "rejected_rule_reason": rejected.get("rule_reason", ""),
        },
    }
    return pair, "ok"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _length_summary(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def render_pairs_stats_report(stats: dict[str, Any]) -> str:
    """Render the dpo-001 pairing stats markdown report (Issue #115)."""
    lg = stats["length_guard"]
    dup = stats["non_duplication"]

    lines = ["# dpo-001 preference pair stats report (Issue #115)", ""]
    lines.append(f"- Total pairs: {stats['total_pairs']}")
    lines.append(f"- Total candidate prompts: {stats['total_prompts']}")
    lines.append("")

    lines.append("## カテゴリ別ペア数")
    lines.append("")
    lines.append("| category | pairs |")
    lines.append("|---|---|")
    for cat, count in stats["category_pair_counts"].items():
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    lines.append("## ペア不成立理由内訳")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for reason, count in stats["skip_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    cl = stats["chosen_length"]
    rl = stats["rejected_length"]
    lines.append("## chosen / rejected 長さ分布")
    lines.append("")
    lines.append(
        f"- chosen: mean={cl['mean']:.1f} median={cl['median']} min={cl['min']} max={cl['max']}"
    )
    lines.append(
        f"- rejected: mean={rl['mean']:.1f} median={rl['median']} min={rl['min']} max={rl['max']}"
    )
    lines.append("")

    sd = stats["score_distribution"]
    lines.append("## judge スコア分布")
    lines.append("")
    lines.append(f"- chosen score mean: {sd.get('chosen_mean', 0):.2f}")
    lines.append(f"- rejected score mean: {sd.get('rejected_mean', 0):.2f}")
    lines.append("")

    lines.append("## 応答長ガード(ハードゲート)")
    lines.append("")
    lines.append(
        f"- chosen: mean={lg['mean']:.1f} median={lg['median']:.1f} min={lg['min']} max={lg['max']}"
    )
    lines.append(
        f"- base_mean={lg['base_mean']} tolerance={lg['tolerance']} -> "
        f"band=[{lg['lower_bound']:.1f}, {lg['upper_bound']:.1f}] -> "
        f"判定: {'PASS' if lg['within_band'] else 'FAIL'}"
    )
    lines.append("")

    lines.append("## 評価非重複アサーション(ハードゲート)")
    lines.append("")
    lines.append(f"- 評価 char_count 値集合: {dup['eval_char_values']}")
    lines.append(f"- ペア化プロンプト char_count/compound max 値集合: {dup['distill_max_values']}")
    lines.append(f"- 値の重複: {dup['value_overlap'] or '(なし)'}")
    lines.append(f"- topic の評価プロンプトへの出現: {dup['topic_hits'] or '(なし)'}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_preference_pairs(config_path: str | Path) -> dict[str, Any]:
    """Run the end-to-end dpo-001 pairing described by ``config_path`` (see
    ``configs/data/dpo_pairs_001.yaml``'s ``pref_pairs`` section): merge
    generations/verdicts/judgments per prompt, select at most one
    chosen/rejected pair per prompt, enforce the two hard gates, write
    ``dpo_pairs.jsonl``, and (if configured) write a markdown stats report."""
    config = load_config(config_path)
    cfg = config.get("pref_pairs", config)

    prompts_path = cfg["prompts_path"]
    generations_path = cfg["generations_path"]
    verdicts_path = cfg["verdicts_path"]
    judgments_path = cfg["judgments_path"]
    output_path = cfg["output_path"]
    stats_report_path = cfg.get("stats_report")
    polite_min_score = int(cfg.get("polite_min_score", _DEFAULT_POLITE_MIN_SCORE))
    eval_prompts_path = cfg.get("eval_prompts_path")

    length_guard_cfg = cfg.get("length_guard", {})
    base_mean = float(length_guard_cfg.get("base_mean_chars", 176))
    tolerance = float(length_guard_cfg.get("tolerance", 0.20))

    prompts = _read_jsonl(prompts_path)
    prompts_by_id = {p["id"]: p for p in prompts}
    generations = _read_jsonl(generations_path)
    verdicts = _read_jsonl(verdicts_path)
    judgments = _read_jsonl(judgments_path)

    generations_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for gen in generations:
        generations_by_prompt.setdefault(gen["prompt_id"], []).append(gen)
    verdicts_by_key = {(v["prompt_id"], v["k"]): v for v in verdicts}
    judgments_by_key = {(j["prompt_id"], j["k"]): j for j in judgments}

    pairs: list[dict[str, Any]] = []
    skip_reasons: Counter = Counter()
    category_pair_counts: Counter = Counter()

    for prompt_row in prompts:
        samples = merge_samples(
            prompt_row["id"], generations_by_prompt, verdicts_by_key, judgments_by_key
        )
        if not samples:
            skip_reasons["no_generations"] += 1
            continue
        pair, status = build_pair(prompt_row, samples, polite_min_score=polite_min_score)
        if pair is None:
            skip_reasons[status] += 1
            continue
        pairs.append(pair)
        category_pair_counts[prompt_row["category"]] += 1

    chosen_rows_for_guard = [{"response": p["chosen"]} for p in pairs]
    length_guard_stats = check_response_length_guard(chosen_rows_for_guard, base_mean, tolerance)

    if eval_prompts_path:
        eval_prompts = _read_jsonl(eval_prompts_path)
        used_prompt_rows = [
            {
                "category": prompts_by_id[p["meta"]["prompt_id"]]["category"],
                "detail": prompts_by_id[p["meta"]["prompt_id"]].get("constraint_detail") or {},
                "topic": prompts_by_id[p["meta"]["prompt_id"]].get("topic", ""),
            }
            for p in pairs
        ]
        dup_check = check_eval_non_duplication(used_prompt_rows, eval_prompts)
    else:
        dup_check = {
            "eval_char_values": [],
            "distill_max_values": [],
            "value_overlap": [],
            "topic_hits": [],
        }

    _write_jsonl(output_path, pairs)

    chosen_lengths = [len(p["chosen"]) for p in pairs]
    rejected_lengths = [len(p["rejected"]) for p in pairs]
    chosen_scores = [
        p["meta"]["chosen_score"] for p in pairs if p["meta"]["chosen_score"] is not None
    ]
    rejected_scores = [
        p["meta"]["rejected_score"] for p in pairs if p["meta"]["rejected_score"] is not None
    ]

    stats: dict[str, Any] = {
        "total_pairs": len(pairs),
        "total_prompts": len(prompts),
        "category_pair_counts": dict(category_pair_counts),
        "skip_reasons": dict(skip_reasons),
        "chosen_length": _length_summary(chosen_lengths),
        "rejected_length": _length_summary(rejected_lengths),
        "score_distribution": {
            "chosen_mean": statistics.fmean(chosen_scores) if chosen_scores else 0.0,
            "rejected_mean": statistics.fmean(rejected_scores) if rejected_scores else 0.0,
        },
        "length_guard": length_guard_stats,
        "non_duplication": dup_check,
        "output_path": str(output_path),
    }

    if stats_report_path:
        report = render_pairs_stats_report(stats)
        Path(stats_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_report_path).write_text(report, encoding="utf-8")
        stats["report_path"] = str(stats_report_path)
        logger.info("Report written to %s", stats_report_path)

    logger.info("dpo-001 preference pairs written: %d pairs -> %s", len(pairs), output_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build the dpo-001 chosen/rejected preference pairs (Issue #115)"
    )
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_001.yaml",
        help="Path to configs/data/dpo_pairs_001.yaml",
    )
    args = parser.parse_args()

    result = build_preference_pairs(args.config)
    logger.info(
        "Done: %d pairs -> %s (report: %s)",
        result["total_pairs"],
        result["output_path"],
        result.get("report_path"),
    )


if __name__ == "__main__":
    main()
