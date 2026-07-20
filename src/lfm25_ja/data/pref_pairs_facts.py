"""K3 factual preference pair construction (Issue #124 / #145).

Pairs chosen (judge 4-5, non-degenerate) against rejected (judge 1-2 or
escape). When no pass/fail contrast exists, uses pass-vs-pass score gap
pairs (dpo-001 pattern).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.data.distill_select import check_response_length_guard
from lfm25_ja.data.pref_pairs import merge_samples
from lfm25_ja.data.pref_verify_facts import is_escape_response
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

_CHOSEN_MIN_SCORE = 4
_REJECTED_MAX_SCORE = 2
_MIN_PAIRS_DEFAULT = 500


def select_chosen_factual(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        s
        for s in samples
        if not s["degenerate"]
        and s.get("score") is not None
        and s["score"] >= _CHOSEN_MIN_SCORE
    ]
    if not candidates:
        return None
    max_score = max(s["score"] for s in candidates)
    pool = [s for s in candidates if s["score"] == max_score]
    pool.sort(key=lambda s: len(s["response"]))
    return pool[0]


def select_rejected_factual(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        s
        for s in samples
        if not s["degenerate"]
        and (
            (s.get("score") is not None and s["score"] <= _REJECTED_MAX_SCORE)
            or is_escape_response(s.get("response", ""))
        )
    ]
    if not candidates:
        return None
    scored = [s for s in candidates if s.get("score") is not None]
    pool = scored if scored else candidates
    if scored:
        min_score = min(s["score"] for s in pool)
        pool = [s for s in pool if s["score"] == min_score]
    pool.sort(key=lambda s: -len(s["response"]))
    return pool[0]


def select_gap_pair(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """pass-vs-pass: highest vs lowest judge score among non-degenerate samples."""
    candidates = [
        s for s in samples if not s["degenerate"] and s.get("score") is not None
    ]
    if len(candidates) < 2:
        return None
    hi = max(candidates, key=lambda s: (s["score"], -len(s["response"])))
    lo = min(candidates, key=lambda s: (s["score"], len(s["response"])))
    if hi["score"] <= lo["score"]:
        return None
    return hi, lo


def build_all_factual_pairs(
    prompt_row: dict[str, Any],
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Build all valid factual pairs for one prompt (may be 0..N)."""
    if not samples:
        return [], "no_generations"

    pairs: list[dict[str, Any]] = []
    chosen_pool = [
        s
        for s in samples
        if not s["degenerate"]
        and s.get("score") is not None
        and s["score"] >= _CHOSEN_MIN_SCORE
    ]
    rejected_pool = [
        s
        for s in samples
        if not s["degenerate"]
        and (
            (s.get("score") is not None and s["score"] <= _REJECTED_MAX_SCORE)
            or is_escape_response(s.get("response", ""))
        )
    ]

    seen: set[tuple[int, int]] = set()
    for chosen in chosen_pool:
        for rejected in rejected_pool:
            if chosen["k"] == rejected["k"]:
                continue
            key = (chosen["k"], rejected["k"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "prompt": prompt_row["prompt"],
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                    "meta": {
                        "prompt_id": prompt_row["id"],
                        "category": prompt_row["category"],
                        "jkb_id": prompt_row["constraint_detail"]["jkb_id"],
                        "pair_type": "pass_fail",
                        "chosen_k": chosen["k"],
                        "chosen_score": chosen["score"],
                        "rejected_k": rejected["k"],
                        "rejected_score": rejected.get("score"),
                    },
                }
            )

    if not pairs:
        gap = select_gap_pair(samples)
        if gap is None:
            return [], "no_pair_candidates"
        chosen, rejected = gap
        pairs.append(
            {
                "prompt": prompt_row["prompt"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "meta": {
                    "prompt_id": prompt_row["id"],
                    "category": prompt_row["category"],
                    "jkb_id": prompt_row["constraint_detail"]["jkb_id"],
                    "pair_type": "pass_pass_gap",
                    "chosen_k": chosen["k"],
                    "chosen_score": chosen["score"],
                    "rejected_k": rejected["k"],
                    "rejected_score": rejected.get("score"),
                },
            }
        )
    return pairs, "ok"


def build_factual_pair(
    prompt_row: dict[str, Any],
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    chosen = select_chosen_factual(samples)
    rejected = select_rejected_factual(samples)
    pair_type = "pass_fail"

    if chosen is None or rejected is None:
        gap = select_gap_pair(samples)
        if gap is None:
            if chosen is None and rejected is None:
                return None, "no_pair_candidates"
            if chosen is None:
                return None, "no_chosen_candidate"
            return None, "no_rejected_candidate"
        chosen, rejected = gap
        pair_type = "pass_pass_gap"

    pair = {
        "prompt": prompt_row["prompt"],
        "chosen": chosen["response"],
        "rejected": rejected["response"],
        "meta": {
            "prompt_id": prompt_row["id"],
            "category": prompt_row["category"],
            "jkb_id": prompt_row["constraint_detail"]["jkb_id"],
            "pair_type": pair_type,
            "chosen_k": chosen["k"],
            "chosen_score": chosen["score"],
            "rejected_k": rejected["k"],
            "rejected_score": rejected.get("score"),
        },
    }
    return pair, "ok"


def build_factual_preference_pairs(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    cfg = config.get("pref_pairs_facts", config)

    prompts = _read_jsonl(cfg["prompts_path"])
    prompts_by_id = {p["id"]: p for p in prompts}
    generations = _read_jsonl(cfg["generations_path"])
    verdicts = _read_jsonl(cfg["verdicts_path"])
    judgments = _read_jsonl(cfg["judgments_path"])

    generations_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for gen in generations:
        generations_by_prompt.setdefault(gen["prompt_id"], []).append(gen)
    verdicts_by_key = {(v["prompt_id"], v["k"]): v for v in verdicts}
    judgments_by_key = {(j["prompt_id"], j["k"]): j for j in judgments}

    min_pairs = int(cfg.get("min_pairs", _MIN_PAIRS_DEFAULT))
    length_guard_cfg = cfg.get("length_guard", {})
    base_mean = float(length_guard_cfg.get("base_mean_chars", 40))
    tolerance = float(length_guard_cfg.get("tolerance", 0.20))

    pairs: list[dict[str, Any]] = []
    skip_reasons: Counter = Counter()
    pair_type_counts: Counter = Counter()

    for prompt_row in prompts:
        samples = merge_samples(
            prompt_row["id"], generations_by_prompt, verdicts_by_key, judgments_by_key
        )
        new_pairs, status = build_all_factual_pairs(prompt_row, samples)
        if not new_pairs:
            skip_reasons[status] += 1
            continue
        for pair in new_pairs:
            pairs.append(pair)
            pair_type_counts[pair["meta"]["pair_type"]] += 1

    chosen_rows = [{"response": p["chosen"]} for p in pairs]
    length_guard_stats = check_response_length_guard(chosen_rows, base_mean, tolerance)

    if len(pairs) < min_pairs:
        raise ValueError(
            f"K3 factual pairs below minimum: {len(pairs)} < {min_pairs}. "
            f"Skip reasons: {dict(skip_reasons)}"
        )

    output_path = cfg["output_path"]
    _write_jsonl(output_path, pairs)

    stats: dict[str, Any] = {
        "total_pairs": len(pairs),
        "min_pairs_required": min_pairs,
        "total_prompts": len(prompts),
        "pair_type_counts": dict(pair_type_counts),
        "skip_reasons": dict(skip_reasons),
        "length_guard": length_guard_stats,
        "output_path": str(output_path),
    }
    stats_report_path = cfg.get("stats_report")
    if stats_report_path:
        lines = [
            "# K3 factual DPO pair stats (Issue #124)",
            "",
            f"- Total pairs: {stats['total_pairs']}",
            f"- Min required: {stats['min_pairs_required']}",
            f"- Pair types: {stats['pair_type_counts']}",
            f"- Skip reasons: {stats['skip_reasons']}",
            "",
        ]
        Path(stats_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["report_path"] = str(stats_report_path)

    logger.info("K3 factual pairs: %d -> %s", len(pairs), output_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build K3 factual DPO pairs")
    parser.add_argument("--config", default="configs/data/dpo_pairs_k3_facts.yaml")
    args = parser.parse_args()
    result = build_factual_preference_pairs(args.config)
    print(result)


if __name__ == "__main__":
    main()
