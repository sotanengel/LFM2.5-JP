"""dpo-001 preference-pair prompt pool construction (Issue #115, Phase P0, CPU).

Phase 4 trains DPO directly on the base model
(``LiquidAI/LFM2.5-1.2B-JP-202606``): preference pairs are built from the
base model's own on-policy K-sample generations (Phase G,
:mod:`lfm25_ja.data.pref_generate`), rule-verified (Phase V,
:mod:`lfm25_ja.data.pref_verify`) and LLM-judged (Phase J,
:mod:`lfm25_ja.eval.judge_swallow`) before pairing (Phase P,
:mod:`lfm25_ja.data.pref_pairs`). This module builds the *prompt* pool those
later phases sample against -- no model inference happens here.

Two sources feed the pool:

Source A
    All 4,000 rows of the sft-005 distillation CSV
    (``datasets/sft/sft005_distill_candidateB_prompts.csv``), read via
    :func:`lfm25_ja.data.distill_select.read_distill_csv`. Only
    prompt/category/instruction_id_list/constraint_detail/topic are kept --
    the CSV's ``response`` column is a rule-based sentence-bank composer
    output (see :mod:`lfm25_ja.data.distill_select`'s docstring), not
    something we want to reuse as a preference-pair *response*, since dpo-001
    scores the base model's *own* on-policy samples instead.

Source B
    Deterministic (seed=42), programmatically generated prompts that target
    ifeval_ja coverage gaps identified from base's sft-005 eval run: small-N
    char_count "以内" prompts, char_count "以上" (lower-bound-only) prompts
    (a constraint shape the CSV never produces), min-max both-sided char
    prompts, and plain-style (常体, ``polite_form`` style=plain) prompts
    (base's weakest ifeval_ja category at 0.783). Built over the same 125
    topics the CSV uses (``gen.py``'s topic bank), reusing the CSV's own
    ``topic`` column rather than a new list, so Source B prompts inherit the
    same is-this-a-real-Japanese-topic vetting the CSV author already did.

Hard gate (checked once against the full combined pool, both here and in
:func:`build_prompt_pool`):

    no ``char_count``/``compound`` prompt's ``min``/``max`` may collide with
    the exact ``{50,60,70,80,90,100,120,150}`` value set used by
    ``datasets/eval/ifeval_ja/prompts.jsonl``'s own char_count kwargs, and no
    prompt's ``topic`` may appear as a substring of any eval prompt's text.

Manual inspection of the real CSV (see ``git log`` for this issue's PR
description) found 12 ``compound`` rows with ``max=90`` -- an eval-set
value -- among the raw 4,000; :func:`build_source_a_prompts` drops those
proactively (mirroring ``distill_select.select_row``'s identical proactive
exclusion), so Source A's *pool* count is 3,988, not 4,000, even though all
4,000 CSV rows are read. :func:`check_prompt_pool_non_duplication` is
therefore expected to be a no-op safety net on the real data, exactly like
its ``distill_select.check_eval_non_duplication`` counterpart.

``format_markdown_table`` / ``numeric_only`` constraint types are
intentionally never generated for Source B (held out, per Issue #115 spec);
the real CSV also never produces them, so no explicit filtering is needed for
Source A either.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.data.distill_select import _eval_char_count_values, read_distill_csv
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

# Source B bucket definitions (Issue #115 spec, seed=42 deterministic). Per-
# bucket sample sizes are this module's own choice of exact numbers landing
# on the spec's "~N" targets:
#   char_count "以内" (5 values x 60 topics = 300)
#   char_count "以上" (5 values x 40 topics = 200)
#   char_count min-max (4 pairs x 38 topics = 152, target ~150)
#   plain style (3 templates x 67 topics = 201, target ~200)
# None of these values collide with the eval set {50,60,70,80,90,100,120,150}
# by construction (validated in tests/test_pref_pipeline.py).
_CHAR_LTE_VALUES: tuple[int, ...] = (40, 45, 55, 65, 75)
_CHAR_LTE_PER_VALUE = 60

_CHAR_GTE_VALUES: tuple[int, ...] = (30, 40, 55, 65, 75)
_CHAR_GTE_PER_VALUE = 40

_CHAR_RANGE_PAIRS: tuple[tuple[int, int], ...] = ((40, 110), (55, 130), (65, 140), (75, 160))
_CHAR_RANGE_PER_PAIR = 38

_PLAIN_TEMPLATES: tuple[str, ...] = (
    "{topic}について常体(だ・である調)で簡潔に説明してください。",
    "{topic}とは何か、常体(だ・である調)で述べてください。",
    "{topic}について、だ・である調で説明してください。",
)
_PLAIN_PER_TEMPLATE = 67

_POOL_ROW_FIELDS = ("id", "category", "instruction_id_list", "constraint_detail", "topic", "prompt")


# ---------------------------------------------------------------------------
# Source A: reuse the sft-005 distillation CSV (prompts only, no response)
# ---------------------------------------------------------------------------


def build_source_a_prompts(
    csv_rows: list[dict[str, Any]],
    eval_char_values: set[int],
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    """Convert CSV rows (as returned by ``distill_select.read_distill_csv``,
    i.e. each having a parsed ``detail`` dict) into pool rows, dropping the
    ``response``/``response_char_count`` columns entirely.

    Proactively drops any ``char_count``/``compound`` row whose ``min`` or
    ``max`` collides with ``eval_char_values`` (see module docstring) --
    returns ``(kept_rows, dropped_count)``.
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    idx = start_index
    for row in csv_rows:
        detail = row["detail"]
        if row["category"] in ("char_count", "compound"):
            values = {detail.get("min"), detail.get("max")} - {None}
            if values & eval_char_values:
                dropped += 1
                continue
        kept.append(
            {
                "id": f"pref-{idx:05d}",
                "category": row["category"],
                "instruction_id_list": row["instruction_id_list"],
                "constraint_detail": detail,
                "topic": row["topic"],
                "prompt": row["prompt"],
            }
        )
        idx += 1
    return kept, dropped


# ---------------------------------------------------------------------------
# Source B: programmatic coverage-gap prompts (deterministic, seed=42)
# ---------------------------------------------------------------------------


def build_source_b_prompts(
    topics: list[str], start_index: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Build the coverage-gap prompt buckets described in the module
    docstring. ``topics`` is sampled deterministically (seeded RNG, sorted
    input first so the sample is reproducible independent of input order)
    without replacement per bucket-value. Per-bucket sample sizes are capped
    at the number of unique topics available, so small synthetic pools
    (tests) shrink proportionally while the real 125-topic pool always gets
    the full spec counts."""
    rng = random.Random(seed)
    topics_sorted = sorted(set(topics))
    rows: list[dict[str, Any]] = []
    idx = start_index

    def _pick(count: int) -> list[str]:
        return rng.sample(topics_sorted, min(count, len(topics_sorted)))

    for n in _CHAR_LTE_VALUES:
        for topic in _pick(_CHAR_LTE_PER_VALUE):
            rows.append(
                {
                    "id": f"pref-{idx:05d}",
                    "category": "char_count",
                    "instruction_id_list": "char_count",
                    "constraint_detail": {"max": n},
                    "topic": topic,
                    "prompt": f"{topic}について{n}字以内で説明してください。",
                }
            )
            idx += 1

    for mn in _CHAR_GTE_VALUES:
        for topic in _pick(_CHAR_GTE_PER_VALUE):
            rows.append(
                {
                    "id": f"pref-{idx:05d}",
                    "category": "char_count",
                    "instruction_id_list": "char_count",
                    "constraint_detail": {"min": mn},
                    "topic": topic,
                    "prompt": f"{topic}について{mn}字以上で説明してください。",
                }
            )
            idx += 1

    for mn, mx in _CHAR_RANGE_PAIRS:
        for topic in _pick(_CHAR_RANGE_PER_PAIR):
            rows.append(
                {
                    "id": f"pref-{idx:05d}",
                    "category": "char_count",
                    "instruction_id_list": "char_count",
                    "constraint_detail": {"min": mn, "max": mx},
                    "topic": topic,
                    "prompt": f"{topic}について{mn}文字以上{mx}文字以内で説明してください。",
                }
            )
            idx += 1

    for template in _PLAIN_TEMPLATES:
        for topic in _pick(_PLAIN_PER_TEMPLATE):
            rows.append(
                {
                    "id": f"pref-{idx:05d}",
                    "category": "polite_form",
                    "instruction_id_list": "polite_form",
                    "constraint_detail": {"style": "plain"},
                    "topic": topic,
                    "prompt": template.format(topic=topic),
                }
            )
            idx += 1

    return rows


# ---------------------------------------------------------------------------
# Hard gate: pool <-> eval non-duplication
# ---------------------------------------------------------------------------


def check_prompt_pool_non_duplication(
    rows: list[dict[str, Any]], eval_prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Hard gate: no pool row's ``char_count``/``compound`` ``min``/``max``
    may equal a value in the eval set's own ``char_count`` kwargs (min or
    max), and no ``topic`` may appear as a substring of any eval prompt's
    text. Raises ``ValueError`` on violation.

    Extends ``distill_select.check_eval_non_duplication`` by also checking
    ``min`` (that function only ever needed to check ``max``, since the CSV
    never produces a bare-min-only char_count row -- Source B's "以上" bucket
    here does)."""
    eval_char_values = _eval_char_count_values(eval_prompts)
    eval_prompt_texts = [r["prompt"] for r in eval_prompts]

    pool_char_values: set[int] = set()
    for row in rows:
        if row["category"] in ("char_count", "compound"):
            detail = row["constraint_detail"]
            for key in ("min", "max"):
                v = detail.get(key)
                if v is not None:
                    pool_char_values.add(v)
    value_overlap = pool_char_values & eval_char_values

    topic_hits: list[str] = []
    topics = {row["topic"] for row in rows if row.get("topic")}
    for topic in topics:
        if any(topic in text for text in eval_prompt_texts):
            topic_hits.append(topic)

    if value_overlap:
        raise ValueError(
            "prompt pool non-duplication violated: char_count/compound min/max "
            f"values overlap with eval char_count kwargs: {sorted(value_overlap)}"
        )
    if topic_hits:
        raise ValueError(
            "prompt pool non-duplication violated: pool topics appear in eval prompts: "
            f"{sorted(topic_hits)}"
        )

    return {
        "eval_char_values": sorted(eval_char_values),
        "pool_char_values": sorted(pool_char_values),
        "value_overlap": sorted(value_overlap),
        "topic_hits": sorted(topic_hits),
    }


# ---------------------------------------------------------------------------
# Stats report
# ---------------------------------------------------------------------------


def render_pool_stats_report(stats: dict[str, Any]) -> str:
    dup = stats["non_duplication"]
    lines = ["# dpo-001 preference prompt pool stats report (Issue #115)", ""]
    lines.append(f"- Seed: {stats['seed']}")
    lines.append(f"- Total pool prompts: {stats['total']}")
    lines.append(
        f"- Source A (distill CSV, eval-collision drops): {stats['source_a_count']} kept, "
        f"{stats['source_a_dropped_eval_collision']} dropped"
    )
    lines.append(f"- Source B (coverage-gap, programmatic): {stats['source_b_count']}")
    lines.append(f"- Output: {stats['output_path']}")
    lines.append("")

    lines.append("## カテゴリ別件数")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---|")
    for cat, count in stats["category_counts"].items():
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    lines.append("## 評価非重複アサーション(ハードゲート)")
    lines.append("")
    lines.append(f"- 評価 char_count 値集合: {dup['eval_char_values']}")
    lines.append(f"- プール char_count/compound min/max 値集合(一部): {dup['pool_char_values']}")
    lines.append(f"- 値の重複: {dup['value_overlap'] or '(なし)'}")
    lines.append(f"- topic の評価プロンプトへの出現: {dup['topic_hits'] or '(なし)'}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_prompt_pool(config_path: str | Path) -> dict[str, Any]:
    """Run the end-to-end dpo-001 prompt pool construction described by
    ``config_path`` (see ``configs/data/dpo_pairs_001.yaml``'s ``pref_prompts``
    section): read the distill CSV (Source A), generate the coverage-gap
    prompts (Source B), enforce the non-duplication hard gate, write the pool
    JSONL, and (if configured) write a markdown stats report."""
    config = load_config(config_path)
    cfg = config.get("pref_prompts", config)

    seed = int(cfg.get("seed", 42))
    source_csv = cfg["source_csv"]
    eval_prompts_path = cfg["eval_prompts_path"]
    output_path = cfg["output_path"]
    stats_report_path = cfg.get("stats_report")

    eval_prompts = _read_jsonl(eval_prompts_path)
    eval_char_values = _eval_char_count_values(eval_prompts)

    csv_rows = read_distill_csv(source_csv)
    source_a_rows, dropped_a = build_source_a_prompts(csv_rows, eval_char_values, start_index=1)

    topics = sorted({row["topic"] for row in csv_rows if row.get("topic")})
    source_b_rows = build_source_b_prompts(topics, start_index=len(source_a_rows) + 1, seed=seed)

    pool = source_a_rows + source_b_rows
    dup_check = check_prompt_pool_non_duplication(pool, eval_prompts)

    _write_jsonl(output_path, pool)

    stats: dict[str, Any] = {
        "seed": seed,
        "total": len(pool),
        "source_a_count": len(source_a_rows),
        "source_a_dropped_eval_collision": dropped_a,
        "source_b_count": len(source_b_rows),
        "category_counts": dict(Counter(row["category"] for row in pool)),
        "non_duplication": dup_check,
        "output_path": str(output_path),
    }

    if stats_report_path:
        report = render_pool_stats_report(stats)
        Path(stats_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_report_path).write_text(report, encoding="utf-8")
        stats["report_path"] = str(stats_report_path)
        logger.info("Report written to %s", stats_report_path)

    logger.info("dpo-001 prompt pool written: %d rows -> %s", len(pool), output_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build the dpo-001 preference-pair prompt pool (Issue #115)"
    )
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_001.yaml",
        help="Path to configs/data/dpo_pairs_001.yaml",
    )
    args = parser.parse_args()

    result = build_prompt_pool(args.config)
    logger.info(
        "Done: %d prompts -> %s (report: %s)",
        result["total"],
        result["output_path"],
        result.get("report_path"),
    )


if __name__ == "__main__":
    main()
