"""JKB v1 (Japan Knowledge Bench) scorer + I/O (Issue #121 / Epic #120)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from lfm25_ja.eval.japan_probe import extract_answer_segment

JKB_DOMAINS: tuple[str, ...] = (
    "地理",
    "歴史",
    "文学",
    "食文化",
    "伝統文化",
    "政治・制度",
    "生活・慣習",
    "地域・観光",
    "スポーツ",
    "科学技術・産業",
    "宗教・信仰",
    "言語",
)

JKB_DIFFICULTIES: tuple[str, ...] = ("core", "standard", "advanced")

_FORMATS: tuple[str, ...] = ("short_answer", "mcq")

_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "domain",
    "difficulty",
    "format",
    "prompt",
    "answers",
    "choices",
    "correct_choice",
    "source_url",
    "source_quote",
)

_MCQ_ANSWER_RE = re.compile(r"^[\s（(]*([A-E])[\s)）:：.．、,]*")
_MCQ_FALLBACK_RE = re.compile(r"答え[:：]\s*([A-E])")

_Z_95 = 1.959963985  # two-sided 95% normal quantile, used by the Wilson interval


def _row_label(row: dict[str, Any], line_no: int) -> str:
    """Best-effort identifier for an error message: the row's id, else its line number."""
    return str(row["id"]) if row.get("id") else f"<line {line_no}>"


def _validate_row(row: dict[str, Any], line_no: int) -> None:
    label = _row_label(row, line_no)
    for field in _REQUIRED_FIELDS:
        if field not in row:
            raise ValueError(f"jkb row {label!r}: missing required field {field!r}")

    if row["domain"] not in JKB_DOMAINS:
        raise ValueError(f"jkb row {label!r}: unknown domain {row['domain']!r}")
    if row["difficulty"] not in JKB_DIFFICULTIES:
        raise ValueError(f"jkb row {label!r}: unknown difficulty {row['difficulty']!r}")

    fmt = row["format"]
    if fmt not in _FORMATS:
        raise ValueError(f"jkb row {label!r}: format must be one of {_FORMATS}, got {fmt!r}")

    if fmt == "short_answer":
        if not row["answers"]:
            raise ValueError(f"jkb row {label!r}: short_answer format requires non-empty answers")
    elif fmt == "mcq":
        choices = row["choices"]
        correct_choice = row["correct_choice"]
        if not choices:
            raise ValueError(f"jkb row {label!r}: mcq format requires non-empty choices")
        if not correct_choice:
            raise ValueError(f"jkb row {label!r}: mcq format requires correct_choice")
        labels = [c["label"] for c in choices]
        if correct_choice not in labels:
            raise ValueError(
                f"jkb row {label!r}: correct_choice {correct_choice!r} not in "
                f"choice labels {labels}"
            )


def load_jkb_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JKB JSONL dataset, validating every row against the schema."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _validate_row(row, line_no)
            rows.append(row)
    return rows


def score_short_answer(raw_text: str, answers: list[str]) -> bool:
    """True when any expected answer is a substring of raw_text's extracted answer segment."""
    segment = extract_answer_segment(raw_text)
    return any(a in segment for a in answers)


def score_mcq(raw_text: str, correct_choice: str) -> bool:
    """Extract the model's chosen A-E label and compare it to correct_choice."""
    segment = extract_answer_segment(raw_text)
    match = _MCQ_ANSWER_RE.match(segment)
    if match:
        return match.group(1) == correct_choice
    # Leading-label extraction can come up empty (e.g. the continuation echoes
    # a "B:"-style choice list, which extract_answer_segment's boundary cut
    # strips away) -- fall back to an explicit "答え: X" tag anywhere in the
    # raw, unextracted text.
    fallback = _MCQ_FALLBACK_RE.search(raw_text)
    if fallback:
        return fallback.group(1) == correct_choice
    return False


def score_row(row: dict[str, Any], raw_text: str) -> bool:
    """Dispatch to score_short_answer or score_mcq based on row['format']."""
    fmt = row["format"]
    if fmt == "short_answer":
        return score_short_answer(raw_text, row["answers"])
    if fmt == "mcq":
        return score_mcq(raw_text, row["correct_choice"])
    raise ValueError(f"unknown format {fmt!r}")


def _stat(n: int, correct: int) -> dict[str, Any]:
    return {"n": n, "correct": correct, "accuracy": (correct / n) if n else 0.0}


def aggregate(rows: list[dict[str, Any]], raw_texts: dict[str, str]) -> dict[str, Any]:
    """Score every row against raw_texts and roll up overall/domain/difficulty/cell stats.

    Ids missing from raw_texts count as unanswered (correct=False, still counted in n).
    """
    overall_n = 0
    overall_correct = 0
    domain_n: dict[str, int] = {}
    domain_correct: dict[str, int] = {}
    difficulty_n: dict[str, int] = {}
    difficulty_correct: dict[str, int] = {}
    cell_n: dict[tuple[str, str], int] = {}
    cell_correct: dict[tuple[str, str], int] = {}
    per_row: list[dict[str, Any]] = []

    for row in rows:
        rid = row["id"]
        domain = row["domain"]
        difficulty = row["difficulty"]
        correct = bool(rid in raw_texts and score_row(row, raw_texts[rid]))

        overall_n += 1
        overall_correct += int(correct)
        domain_n[domain] = domain_n.get(domain, 0) + 1
        domain_correct[domain] = domain_correct.get(domain, 0) + int(correct)
        difficulty_n[difficulty] = difficulty_n.get(difficulty, 0) + 1
        difficulty_correct[difficulty] = difficulty_correct.get(difficulty, 0) + int(correct)
        cell_key = (domain, difficulty)
        cell_n[cell_key] = cell_n.get(cell_key, 0) + 1
        cell_correct[cell_key] = cell_correct.get(cell_key, 0) + int(correct)
        per_row.append({"id": rid, "domain": domain, "difficulty": difficulty, "correct": correct})

    return {
        "overall": _stat(overall_n, overall_correct),
        "by_domain": {
            d: _stat(domain_n[d], domain_correct[d]) for d in JKB_DOMAINS if d in domain_n
        },
        "by_difficulty": {
            d: _stat(difficulty_n[d], difficulty_correct[d])
            for d in JKB_DIFFICULTIES
            if d in difficulty_n
        },
        "by_cell": {
            (d, diff): _stat(cell_n[(d, diff)], cell_correct[(d, diff)])
            for d in JKB_DOMAINS
            for diff in JKB_DIFFICULTIES
            if (d, diff) in cell_n
        },
        "per_row": per_row,
    }


def _wilson_ci(n: int, correct: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion, as (lower, upper) fractions."""
    if n == 0:
        return (0.0, 0.0)
    phat = correct / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return (lower, upper)


def _ci_cell(n: int, correct: int) -> str:
    lo, hi = _wilson_ci(n, correct)
    half = (hi - lo) / 2 * 100
    return f"{lo * 100:.1f}-{hi * 100:.1f}% (±{half:.1f}pt)"


def render_report_markdown(agg: dict[str, Any], model_label: str) -> str:
    """Render aggregate() output as a Markdown report for one model."""
    overall = agg["overall"]
    lines: list[str] = [f"# JKB v1 レポート: {model_label}", ""]

    lines += [
        "## 全体",
        "",
        f"- N = {overall['n']}",
        f"- 正答数 = {overall['correct']}",
        f"- 正答率 = {overall['accuracy'] * 100:.1f}% "
        f"(95% CI: {_ci_cell(overall['n'], overall['correct'])})",
        "",
    ]

    lines += ["## 分野 x 難度 クロス集計", ""]
    header = ["分野", *JKB_DIFFICULTIES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    by_cell = agg["by_cell"]
    for domain in JKB_DOMAINS:
        if not any((domain, diff) in by_cell for diff in JKB_DIFFICULTIES):
            continue
        cells = [domain]
        for diff in JKB_DIFFICULTIES:
            stat = by_cell.get((domain, diff))
            if stat is None:
                cells.append("-")
            else:
                cells.append(f"{stat['accuracy'] * 100:.0f}% ({stat['correct']}/{stat['n']})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 分野別", "", "| 分野 | N | 正答率 | 95% CI |", "|---|---|---|---|"]
    for domain, stat in agg["by_domain"].items():
        lines.append(
            f"| {domain} | {stat['n']} | {stat['accuracy'] * 100:.1f}% "
            f"| {_ci_cell(stat['n'], stat['correct'])} |"
        )
    lines.append("")

    lines += ["## 難度別", "", "| 難度 | N | 正答率 | 95% CI |", "|---|---|---|---|"]
    for difficulty, stat in agg["by_difficulty"].items():
        lines.append(
            f"| {difficulty} | {stat['n']} | {stat['accuracy'] * 100:.1f}% "
            f"| {_ci_cell(stat['n'], stat['correct'])} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"
