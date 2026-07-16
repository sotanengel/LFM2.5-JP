"""sft-005 distillation-CSV selection + mix pipeline (Issue #109).

sft-002-mix (Issue #105) regressed hard on ifeval_ja prompt_strict (base
-18pt): the root cause identified was an off-policy style shift (the mixed
training data averaged 284 chars/response vs. the base model's own ~176
chars/response). sft-005 re-attempts SFT with a mixture of (a) constrained
distillation responses that already pass the *same* rule-based verifiers used
by the ifeval_ja eval harness, so training signal is on-distribution with
what the eval actually scores, and (b) ichikara only (no llm-jp-instruct /
aya-ja / other free-form data that drove the style shift).

The distillation responses are **not** newly generated here -- per explicit
user instruction, this pipeline only *selects* rows from an already-prepared
CSV (``datasets/sft/sft005_distill_candidateB_prompts.csv``, a prior-session
artifact; see the docstring of :func:`render_distill_stats_report` for the
exact provenance wording that must appear in the report). No model inference
happens anywhere in this module.

Verifier selection reuses ``lfm25_ja.eval.instruction_verifiers.VERIFIERS``
as-is (frozen, do-not-touch -- same constraint as Issue #105's
``format_constraints.py``) for char_count / bullet_count / polite_form /
keyword / format_json. Constraint types the eval harness doesn't cover
(paragraph_count / start_word / forbidden_word / no_constraint) get
lightweight verifiers defined locally in this module instead of being added
to the frozen eval module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.eval.instruction_verifiers import (
    verify_bullet_count,
    verify_char_count,
    verify_format_json,
    verify_keyword,
    verify_polite_form,
)
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

# Tight length margin for char_count/compound: a strict verify_char_count
# pass (len <= max) isn't enough on its own -- Issue #109 wants responses
# that sit close to their limit (60-90% of max), not short/slack answers
# that would teach the model to under-use its budget. Validated against the
# real 4,000-row CSV: 932/932 char_count and 92/92 compound rows already
# satisfy this band (the CSV's own generator ("gen.py") appears to target it),
# so the gate is not expected to reject real data.
_BAND_MIN_RATIO = 0.60
_BAND_MAX_RATIO = 0.90

_MIN_RESPONSE_CHARS = 20
_LINE_REPEAT_THRESHOLD = 3
_NGRAM_SIZE = 10
_NGRAM_REPEAT_THRESHOLD = 4

# format_json responses legitimately repeat short substrings across list
# items (shared JSON key names like '", "分類": "'). Validated against the
# real CSV: the char n-gram check below flags 271/466 format_json rows at
# these thresholds and 0 rows in every other category -- a false-positive
# rate that would gut an otherwise-clean category, so the repetition check
# is skipped for format_json only (dedup/duplicate-response filters below
# still apply to it).
_REPETITION_EXEMPT_CATEGORIES: frozenset[str] = frozenset({"format_json"})

_CSV_FIELDNAMES = (
    "id",
    "category",
    "instruction_id_list",
    "constraint_detail",
    "topic",
    "prompt",
    "response",
    "response_char_count",
)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def read_distill_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read the sft-005 distillation candidate CSV.

    The file is UTF-8 with a BOM (``utf-8-sig``) and contains quoted fields
    with embedded newlines (``response``), so it must be parsed with
    ``csv.DictReader`` rather than line-by-line. Each row's
    ``constraint_detail`` JSON string is parsed and stored under the row key
    ``"detail"`` (the raw ``constraint_detail`` string is kept too, since it's
    used verbatim as a dedup key).
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = dict(raw_row)
            row["detail"] = json.loads(row["constraint_detail"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# In-module verifiers for constraint types not covered by the eval harness
# ---------------------------------------------------------------------------


def verify_paragraph_count(response: str, params: dict) -> tuple[bool, str]:
    """Paragraph count = number of non-empty blocks separated by a blank
    line (``\\n\\s*\\n``). Matches the real CSV's actual formatting (blank
    line between paragraphs, validated 93/93 on the real data) rather than
    the more literal "改行で区切る" (single newline) reading of the prompt
    text, which would under/over-count when a paragraph's own sentences also
    wrap onto multiple lines."""
    count = params.get("count")
    if count is None:
        raise ValueError("verify_paragraph_count requires params['count']")
    blocks = [b for b in re.split(r"\n\s*\n+", response) if b.strip()]
    if len(blocks) != count:
        return False, f"段落数 {len(blocks)} が期待値 {count} と一致しません"
    return True, ""


def verify_start_word(response: str, params: dict) -> tuple[bool, str]:
    start = params.get("start")
    if not start:
        raise ValueError("verify_start_word requires params['start']")
    if not response.strip().startswith(start):
        return False, f"書き出しが「{start}」で始まっていません"
    return True, ""


def verify_no_constraint(response: str, params: dict) -> tuple[bool, str]:
    """No-constraint rows still need a sane length: non-empty, >=20 chars,
    <=300 chars (params can override via 'min'/'max')."""
    length = len(response.strip())
    mn = params.get("min", _MIN_RESPONSE_CHARS)
    mx = params.get("max", 300)
    if not (mn <= length <= mx):
        return False, f"文字数 {length} が {mn}-{mx} の範囲外です"
    return True, ""


def verify_format_json_detail(response: str, detail: dict) -> tuple[bool, str]:
    """``verify_format_json`` only checks that the payload parses as JSON.
    This adds a lightweight structural check: if ``detail`` has ``keys``,
    every element of the (list, or the first list-valued entry of a dict)
    payload must contain all of them; if ``detail`` has ``count``, that list's
    length must match exactly. Deliberately not stricter than that (e.g. no
    type-checking of values) -- see Issue #109 design note on keeping this
    check decisive but not brittle."""
    ok, msg = verify_format_json(response, {})
    if not ok:
        return False, msg
    text = response.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:  # pragma: no cover - verify_format_json already checked
        return False, f"JSON として解析できません: {e}"

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        items = list_values[0] if list_values else None
    else:
        items = None
    if items is None:
        return False, "トップレベルが list、または list を含む dict ではありません"

    count = detail.get("count")
    if count is not None and len(items) != count:
        return False, f"要素数 {len(items)} が期待値 {count} と一致しません"

    keys = detail.get("keys") or []
    for item in items:
        if not isinstance(item, dict):
            return False, "要素が object ではありません"
        missing = [k for k in keys if k not in item]
        if missing:
            return False, f"必須キーがありません: {missing}"

    return True, ""


def _tight_margin_ok(
    response: str, max_chars: int, band: tuple[float, float] = (_BAND_MIN_RATIO, _BAND_MAX_RATIO)
) -> bool:
    length = len(_nfkc(response))
    lo, hi = band
    return lo * max_chars <= length <= hi * max_chars


# ---------------------------------------------------------------------------
# Per-row category dispatch
# ---------------------------------------------------------------------------


def select_row(
    row: dict[str, Any],
    eval_char_values: set[int] | None = None,
    tight_margin_band: tuple[float, float] = (_BAND_MIN_RATIO, _BAND_MAX_RATIO),
) -> tuple[bool, str]:
    """Return ``(accepted, reason)`` for a single CSV row (as produced by
    :func:`read_distill_csv`, i.e. must have a parsed ``"detail"`` key).

    ``eval_char_values`` (the set of ``char_count`` max/min values used by
    ``datasets/eval/ifeval_ja/prompts.jsonl``) is checked proactively for the
    char_count/compound categories so an accepted row's ``max`` never
    collides with an eval prompt's exact threshold -- this makes
    :func:`check_eval_non_duplication`'s later hard-gate a defense-in-depth
    check rather than something expected to reject an entire prepared mix.
    """
    category = row["category"]
    detail = row["detail"]
    response = row["response"]
    eval_char_values = eval_char_values or set()

    if category == "char_count":
        mx = detail.get("max")
        if mx is None:
            return False, "max が constraint_detail にありません"
        if mx in eval_char_values:
            return False, f"max={mx} は評価データの char_count 値集合と重複します"
        ok, msg = verify_char_count(response, {"max": mx, "min": detail.get("min")})
        if not ok:
            return False, msg
        if not _tight_margin_ok(response, mx, tight_margin_band):
            return False, "60-90%タイトマージンを満たしません"
        return True, ""

    if category == "compound":
        mx = detail.get("max")
        if mx is None:
            return False, "max が constraint_detail にありません"
        if mx in eval_char_values:
            return False, f"max={mx} は評価データの char_count 値集合と重複します"
        ok, msg = verify_char_count(response, {"max": mx})
        if not ok:
            return False, msg
        if not _tight_margin_ok(response, mx, tight_margin_band):
            return False, "60-90%タイトマージンを満たしません"
        include = detail.get("include") or []
        ok, msg = verify_keyword(response, {"include": include})
        if not ok:
            return False, msg
        return True, ""

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


# ---------------------------------------------------------------------------
# Degenerate filters
# ---------------------------------------------------------------------------


def _has_repetition(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if any(c >= _LINE_REPEAT_THRESHOLD for c in Counter(lines).values()):
        return True
    if len(text) >= _NGRAM_SIZE:
        grams = Counter(text[i : i + _NGRAM_SIZE] for i in range(len(text) - _NGRAM_SIZE + 1))
        if any(c >= _NGRAM_REPEAT_THRESHOLD for c in grams.values()):
            return True
    return False


def apply_degenerate_filters(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply, in order, to an already verifier-accepted row list:

    (a) response shorter than 20 chars (stripped, raw length) -- excluded.
    (b) repetition (same line >=3 times, or any 10-gram >=4 times) --
        excluded, except for the ``format_json`` category (see
        ``_REPETITION_EXEMPT_CATEGORIES``).
    (c) exact ``(prompt, response)`` duplicate -- first occurrence kept.
    (d) same ``(category, constraint_detail, response)`` duplicate -- first
        occurrence kept. Note this does NOT catch same-topic *prefix*
        relationships across different ``constraint_detail`` values (e.g. a
        220-char and a 130-char answer about the same topic) -- those are
        intended training signal (different length constraints correctly
        yielding different responses), not duplicates.

    Returns ``(kept_rows, reject_counts)``.
    """
    kept: list[dict[str, Any]] = []
    reject_counts = {
        "too_short": 0,
        "repetition": 0,
        "prompt_response_dup": 0,
        "category_detail_dup": 0,
    }
    seen_prompt_response: set[tuple[str, str]] = set()
    seen_category_detail_response: set[tuple[str, str, str]] = set()

    for row in rows:
        response = row["response"]
        if len(response.strip()) < _MIN_RESPONSE_CHARS:
            reject_counts["too_short"] += 1
            continue
        if row["category"] not in _REPETITION_EXEMPT_CATEGORIES and _has_repetition(response):
            reject_counts["repetition"] += 1
            continue
        pr_key = (row["prompt"], response)
        if pr_key in seen_prompt_response:
            reject_counts["prompt_response_dup"] += 1
            continue
        cd_key = (row["category"], row["constraint_detail"], response)
        if cd_key in seen_category_detail_response:
            reject_counts["category_detail_dup"] += 1
            continue
        seen_prompt_response.add(pr_key)
        seen_category_detail_response.add(cd_key)
        kept.append(row)

    return kept, reject_counts


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def check_response_length_guard(
    rows: list[dict[str, Any]], base_mean: float, tolerance: float
) -> dict[str, Any]:
    """Hard gate: the mean response length (raw ``len``, no NFKC) of the
    accepted distillation rows must fall within ``base_mean * (1 +/-
    tolerance)``. Raises ``ValueError`` on violation (Issue #109: sft-002-mix
    failed because its training data averaged 284 chars/response vs. the
    base model's own ~176 chars/response -- this gate stops that mistake
    from repeating silently)."""
    if not rows:
        raise ValueError("response length guard: no rows to check")
    lengths = [len(r["response"]) for r in rows]
    mean = statistics.fmean(lengths)
    lower_bound = base_mean * (1 - tolerance)
    upper_bound = base_mean * (1 + tolerance)
    within_band = lower_bound <= mean <= upper_bound

    stats = {
        "mean": mean,
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "base_mean": base_mean,
        "tolerance": tolerance,
        "within_band": within_band,
    }
    if not within_band:
        raise ValueError(
            "response length guard failed: mean="
            f"{mean:.1f} outside [{lower_bound:.1f}, {upper_bound:.1f}] "
            f"(base_mean={base_mean}, tolerance={tolerance})"
        )
    return stats


def _eval_char_count_values(eval_prompts: list[dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for row in eval_prompts:
        cc = (row.get("kwargs") or {}).get("char_count")
        if cc:
            for key in ("max", "min"):
                if key in cc and cc[key] is not None:
                    values.add(cc[key])
    return values


def check_eval_non_duplication(
    rows: list[dict[str, Any]], eval_prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Hard gate: the accepted distillation rows must not overlap with the
    ifeval_ja eval set, so training never directly reinforces the exact
    prompts being scored. Raises ``ValueError`` on violation. Checks:

    (a) no ``char_count``/``compound`` row's ``max`` equals a value in the
        eval set's ``char_count`` kwargs (min or max);
    (b) no CSV ``topic`` string appears as a substring of any eval prompt
        text.

    ``select_row`` already excludes (a) proactively during selection (see its
    docstring), so this is expected to be a no-op safety net on real data,
    not a rejection path.
    """
    eval_char_values = _eval_char_count_values(eval_prompts)
    eval_prompt_texts = [r["prompt"] for r in eval_prompts]

    distill_max_values: set[int] = set()
    for row in rows:
        if row["category"] in ("char_count", "compound"):
            mx = row["detail"].get("max")
            if mx is not None:
                distill_max_values.add(mx)
    value_overlap = distill_max_values & eval_char_values

    topic_hits: list[str] = []
    topics = {row["topic"] for row in rows if row.get("topic")}
    for topic in topics:
        if any(topic in text for text in eval_prompt_texts):
            topic_hits.append(topic)

    if value_overlap:
        raise ValueError(
            "eval non-duplication violated: char_count/compound max values overlap "
            f"with eval char_count kwargs: {sorted(value_overlap)}"
        )
    if topic_hits:
        raise ValueError(
            "eval non-duplication violated: distill topics appear in eval prompts: "
            f"{sorted(topic_hits)}"
        )

    return {
        "eval_char_values": sorted(eval_char_values),
        "distill_max_values": sorted(distill_max_values),
        "value_overlap": sorted(value_overlap),
        "topic_hits": sorted(topic_hits),
    }


# ---------------------------------------------------------------------------
# Topic reuse stats
# ---------------------------------------------------------------------------


def compute_topic_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The CSV is built from a topic sentence bank: the same topic is reused
    across multiple categories/constraints on purpose (e.g. a 220-char and a
    130-char answer about the same topic are both valid, distinct training
    signal for different length constraints), so topic reuse itself is
    expected and reported here rather than filtered out."""
    counts = Counter(row["topic"] for row in rows if row.get("topic"))
    reused = {topic: c for topic, c in counts.items() if c > 1}
    return {
        "unique_topics": len(counts),
        "topics_reused": len(reused),
        "max_reuse": counts.most_common(5),
    }


# ---------------------------------------------------------------------------
# messages conversion + mixing
# ---------------------------------------------------------------------------


def row_to_messages(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }


def mix_distill_with_ichikara(
    distill_rows: list[dict[str, Any]], ichikara_rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Concatenate the (already ``messages``-format) distill and ichikara
    rows and shuffle deterministically (seeded)."""
    combined = list(distill_rows) + list(ichikara_rows)
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


def _response_lengths(rows: list[dict[str, Any]]) -> list[int]:
    lengths = []
    for row in rows:
        messages = row.get("messages", [])
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if assistant_msgs:
            lengths.append(len(assistant_msgs[-1].get("content", "")))
    return lengths


def _length_summary(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def _md5_file(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Stats report
# ---------------------------------------------------------------------------


def render_distill_stats_report(stats: dict[str, Any]) -> str:
    """Render the sft-005 distill-selection stats markdown report.

    Data provenance MUST be stated accurately (Issue #109 explicit
    requirement): the distillation responses are a pre-generated CSV (a
    prior-session artifact, adopted per user instruction) built by a
    rule-based sentence-bank composer (``gen.py``), NOT Qwen3-Swallow 8B
    inference output -- no model generation happens anywhere in this
    pipeline.
    """
    lg = stats["length_guard"]
    dup = stats["eval_non_duplication"]
    topic = stats["topic_stats"]

    lines = ["# sft-005 distill selection stats report (Issue #109)", ""]
    lines.append(f"- Seed: {stats['seed']}")
    lines.append(f"- Total mixed rows: {stats['total']}")
    lines.append(
        f"- Distill accepted / CSV total: {stats['distill_accepted']} / "
        f"{stats['distill_total_csv']}"
    )
    lines.append(f"- ichikara rows: {stats['ichikara_count']}")
    lines.append(f"- Output: {stats['output_path']}")
    lines.append("")

    lines.append("## カテゴリ別 採択/CSV件数")
    lines.append("")
    lines.append("| category | csv_total |")
    lines.append("|---|---|")
    for cat, total in stats["category_totals"].items():
        lines.append(f"| {cat} | {total} |")
    lines.append("")

    lines.append("## 棄却理由内訳(verifier 選抜)")
    lines.append("")
    lines.append("| category | reason | count |")
    lines.append("|---|---|---|")
    for cat, reasons in stats["reject_reasons"].items():
        for reason, count in reasons.items():
            lines.append(f"| {cat} | {reason} | {count} |")
    lines.append("")

    lines.append("## 棄却理由内訳(退化フィルタ)")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for reason, count in stats["degenerate_rejects"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append("## 応答長ガード(ハードゲート)")
    lines.append("")
    lines.append(
        f"- 蒸留部分: mean={lg['mean']:.1f} median={lg['median']:.1f} "
        f"min={lg['min']} max={lg['max']}"
    )
    lines.append(
        f"- base_mean={lg['base_mean']} tolerance={lg['tolerance']} -> "
        f"band=[{lg['lower_bound']:.1f}, {lg['upper_bound']:.1f}] -> "
        f"判定: {'PASS' if lg['within_band'] else 'FAIL'}"
    )
    mix_len = stats.get("mix_response_length", {})
    ichikara_len = stats.get("ichikara_response_length", {})
    lines.append(f"- mix 全体 mean={mix_len.get('mean', 0):.1f}")
    lines.append(f"- ichikara 部分 mean={ichikara_len.get('mean', 0):.1f}")
    lines.append("")

    lines.append("## topic 再利用統計")
    lines.append("")
    lines.append(f"- unique topics: {topic['unique_topics']}")
    lines.append(f"- 複数回使われた topics: {topic['topics_reused']}")
    lines.append(f"- 上位再利用: {topic['max_reuse']}")
    lines.append("")

    lines.append("## 評価非重複アサーション(ハードゲート)")
    lines.append("")
    lines.append(f"- 評価 char_count 値集合: {dup['eval_char_values']}")
    lines.append(f"- 蒸留 char_count/compound max 値集合: {dup['distill_max_values']}")
    lines.append(f"- 値の重複: {dup['value_overlap'] or '(なし)'}")
    lines.append(f"- topic の評価プロンプトへの出現: {dup['topic_hits'] or '(なし)'}")
    lines.append("")

    lines.append("## データ来歴")
    lines.append("")
    lines.append(f"- CSV md5: {stats['csv_md5']}")
    lines.append(
        "- 応答は事前生成 CSV(前セッション成果物、ユーザー指示により採用)。"
        "CSV 生成スクリプトはトピック文章バンクからの規則ベース構成(gen.py、"
        "Qwen3-Swallow 8B の推論出力ではない)。"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare_distill_mix(config_path: str | Path) -> dict[str, Any]:
    """Run the end-to-end sft-005 distill selection + mix pipeline described
    by ``config_path`` (see ``configs/data/mix_005.yaml``): load the CSV,
    apply verifier selection + degenerate filters, enforce the response
    length guard and eval non-duplication hard gates, mix with ichikara,
    write the mixture JSONL, and write a markdown stats report.
    """
    config = load_config(config_path)
    mix_cfg = config.get("mix", config)

    seed = int(mix_cfg.get("seed", 42))
    source_csv = mix_cfg["source_csv"]
    ichikara_path = mix_cfg["ichikara_path"]
    eval_prompts_path = mix_cfg["eval_prompts_path"]
    output_path = mix_cfg["output_path"]
    stats_report_path = mix_cfg.get("stats_report")

    length_guard_cfg = mix_cfg.get("length_guard", {})
    base_mean = float(length_guard_cfg.get("base_mean_chars", 176))
    tolerance = float(length_guard_cfg.get("tolerance", 0.20))

    tight_margin_cfg = mix_cfg.get("tight_margin", {})
    tight_margin_band = (
        float(tight_margin_cfg.get("min_ratio", _BAND_MIN_RATIO)),
        float(tight_margin_cfg.get("max_ratio", _BAND_MAX_RATIO)),
    )

    rows = read_distill_csv(source_csv)
    eval_prompts = _read_jsonl(eval_prompts_path)
    eval_char_values = _eval_char_count_values(eval_prompts)

    category_totals = dict(Counter(row["category"] for row in rows))

    accepted: list[dict[str, Any]] = []
    reject_reasons: dict[str, Counter] = {}
    for row in rows:
        ok, reason = select_row(row, eval_char_values, tight_margin_band)
        if ok:
            accepted.append(row)
        else:
            reject_reasons.setdefault(row["category"], Counter())[reason or "rejected"] += 1

    accepted, degenerate_rejects = apply_degenerate_filters(accepted)

    length_guard_stats = check_response_length_guard(accepted, base_mean, tolerance)
    dup_check = check_eval_non_duplication(accepted, eval_prompts)
    topic_stats = compute_topic_stats(rows)

    ichikara_rows = _read_jsonl(ichikara_path)
    distill_messages = [row_to_messages(row) for row in accepted]
    combined = mix_distill_with_ichikara(distill_messages, ichikara_rows, seed)

    _write_jsonl(output_path, combined)

    stats: dict[str, Any] = {
        "seed": seed,
        "total": len(combined),
        "distill_accepted": len(accepted),
        "distill_total_csv": len(rows),
        "category_totals": category_totals,
        "reject_reasons": {cat: dict(counter) for cat, counter in reject_reasons.items()},
        "degenerate_rejects": degenerate_rejects,
        "length_guard": length_guard_stats,
        "eval_non_duplication": dup_check,
        "topic_stats": topic_stats,
        "ichikara_count": len(ichikara_rows),
        "mix_response_length": _length_summary(_response_lengths(combined)),
        "ichikara_response_length": _length_summary(_response_lengths(ichikara_rows)),
        "output_path": str(output_path),
        "csv_md5": _md5_file(source_csv),
    }

    if stats_report_path:
        report = render_distill_stats_report(stats)
        Path(stats_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_report_path).write_text(report, encoding="utf-8")
        stats["report_path"] = str(stats_report_path)
        logger.info("Report written to %s", stats_report_path)

    logger.info("sft-005 distill mix written: %d rows -> %s", len(combined), output_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build the sft-005 distillation-selection + mix dataset (Issue #109)"
    )
    parser.add_argument(
        "--config", default="configs/data/mix_005.yaml", help="Path to configs/data/mix_005.yaml"
    )
    args = parser.parse_args()

    result = prepare_distill_mix(args.config)
    logger.info(
        "Done: %d examples -> %s (report: %s)",
        result["total"],
        result["output_path"],
        result.get("report_path"),
    )


if __name__ == "__main__":
    main()
