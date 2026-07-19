"""Wikipedia-ja subset targeting K1's worst-10 domain x difficulty cells (Issue #130).

Issue #122 (K1 asset re-evaluation, see experiments/reports/k1_asset_reassessment.md)
measured the base model on JKB v1 and identified the "worst 10 cells" (domain x
difficulty, accuracy) that K2's knowledge-injection CPT (Issue #123) must target.
This module builds a keyword-filtered, oversampled Wikipedia-ja subset that
concentrates training signal on those weak cells.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

from lfm25_ja.data.download import download_corpus

# Keyword lists per worst-10 cell (Issue #122 K1 asset reassessment). A Wikipedia
# article "matches" a cell if any of its keywords is a substring of the article's
# title or text (see filter_and_weight_documents). These lists encode an upstream
# design decision (experiments/reports/k1_asset_reassessment.md) and must not be
# invented or altered here.
WORST10_CELL_KEYWORDS: dict[str, list[str]] = {
    "生活・慣習_advanced": [
        "冠婚葬祭", "初七日", "四十九日", "一周忌", "新盆", "彼岸",
        "還暦", "古希", "喜寿", "傘寿", "米寿",
    ],
    "言語_advanced": [
        "日本語の歴史", "上代日本語", "中古日本語", "中世日本語", "近世日本語",
        "近代日本語", "日本語の方言", "係り結び", "連濁",
    ],
    "政治・制度_advanced": [
        "日本国憲法", "大正・昭和の政治事件", "五・一五事件", "二・二六事件",
    ],
    "歴史_advanced": [
        "白村江の戦い", "承久の乱", "建武式目", "南北朝時代", "寛政の改革", "天保の改革",
    ],
    "食文化_advanced": [
        "郷土料理", "へしこ", "ふなずし", "しもつかれ", "ほうとう",
        "五節句", "京料理", "加賀料理",
    ],
    "地域・観光_standard": [
        "世界遺産", "白川郷", "知床", "屋久島", "富岡製糸場", "中尊寺",
    ],
    "食文化_standard": [
        "懐石料理", "精進料理", "雑煮", "みりん", "焼酎",
    ],
    "科学技術・産業_standard": [
        "ノーベル賞", "カミオカンデ", "はやぶさ", "H-IIAロケット", "H-IIA",
    ],
    "生活・慣習_core": [
        "日本の祝日", "建国記念の日", "成人の日", "勤労感謝の日",
    ],
    "生活・慣習_standard": [
        "社会保障制度", "戸籍", "住民票", "マイナンバー", "国民健康保険",
        "健康保険", "国民年金", "厚生年金", "介護保険",
    ],
}


def filter_matching_documents(
    docs: list[dict[str, Any]],
    keyword_groups: dict[str, list[str]] = WORST10_CELL_KEYWORDS,
) -> list[dict[str, Any]]:
    """Filter ``docs`` to those matching a worst-10-cell keyword and tag them.

    A doc matches if any keyword (from any cell's list) is a substring of
    ``doc["text"]`` or ``doc.get("title", "")``. Non-matching docs are dropped
    entirely (this builds a *subset*, not a full reweighted corpus). Matching
    docs are tagged with ``"matched_cells"`` (sorted list of matching cell
    ids). Input order is preserved among matched docs (stable filter).

    This function does *not* duplicate documents -- see ``oversample_documents``
    for that step. The two are kept separate (Issue #123 bugfix) so that
    duplication can be applied *after* ``clean_corpus``'s MinHash dedup stage
    instead of before it: duplicating first produces near-identical copies
    that dedup then correctly (and unhelpfully) removes, cancelling out the
    oversampling entirely.
    """
    result: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.get("text", "")
        title = doc.get("title", "")
        matched_cells = sorted(
            cell
            for cell, keywords in keyword_groups.items()
            if any(kw in text or kw in title for kw in keywords)
        )
        if not matched_cells:
            continue

        new_doc = dict(doc)
        new_doc["matched_cells"] = matched_cells
        result.append(new_doc)

    return result


def oversample_documents(
    docs: list[dict[str, Any]],
    weight: int = 3,
) -> list[dict[str, Any]]:
    """Repeat each doc in ``docs`` ``weight`` times for training-signal oversampling.

    Repeated copies get their ``"id"`` suffixed with ``-dup{n}`` (n =
    1..weight-1) to keep ids unique, while the first occurrence keeps the
    original id unchanged. Works on any docs carrying an ``"id"`` key --
    independent of keyword-matching logic, so it applies equally to any
    already-filtered document list.

    Must run *after* dedup (``clean_corpus``), not before: see
    ``filter_matching_documents`` for why running it beforehand cancels out
    the oversampling (Issue #123 bugfix).
    """
    result: list[dict[str, Any]] = []
    for doc in docs:
        base_id = doc["id"]
        for n in range(weight):
            new_doc = dict(doc)
            new_doc["id"] = base_id if n == 0 else f"{base_id}-dup{n}"
            result.append(new_doc)

    return result


def filter_and_weight_documents(
    docs: list[dict[str, Any]],
    keyword_groups: dict[str, list[str]] = WORST10_CELL_KEYWORDS,
    oversample_weight: int = 3,
) -> list[dict[str, Any]]:
    """Filter ``docs`` to those matching a worst-10-cell keyword, then oversample.

    Thin wrapper composing ``filter_matching_documents`` +
    ``oversample_documents``, kept for callers (e.g.
    ``load_and_filter_wikipedia_ja``) that want filter+oversample as a single
    step with no dedup in between. ``prepare.py``'s pipeline calls the two
    halves separately instead, with ``clean_corpus`` run in between (Issue
    #123 bugfix -- see the two functions above for why that ordering matters).
    """
    return oversample_documents(
        filter_matching_documents(docs, keyword_groups), weight=oversample_weight
    )


def _extract_text_title_rows(
    dataset: Iterable[Any], sample_limit: int | None
) -> list[dict[str, Any]]:
    """Convert a loaded dataset (HF ``Dataset`` or any iterable of mapping rows)
    into a list of ``{"id": ..., "text": ..., "title": ...}`` dicts.

    Mirrors ``prepare._extract_text_rows``'s contract (id/text extraction,
    ``sample_limit`` capping for streaming safety) but additionally propagates
    ``title`` (defaulting to ``""`` if absent), which keyword matching needs.
    A local variant rather than importing ``prepare``'s private helper, since
    that helper does not carry ``title`` through.
    """
    rows: list[dict[str, Any]] = []
    iterator: Iterable[Any] = iter(dataset)
    if sample_limit is not None:
        iterator = itertools.islice(iterator, sample_limit)
    for i, row in enumerate(iterator):
        if "text" not in row:
            raise ValueError(f"Row {i} is missing a 'text' field (keys={list(row.keys())})")
        rows.append({"id": str(i), "text": row["text"], "title": row.get("title", "")})
    return rows


def load_and_filter_wikipedia_ja(
    entry: dict[str, Any],
    cache_dir: str,
    sample_limit: int | None = None,
    streaming: bool = False,
    oversample_weight: int = 3,
) -> list[dict[str, Any]]:
    """Download a Wikipedia-ja corpus entry and return its worst-10-cell subset.

    Wraps ``download.download_corpus`` + row extraction (id/text/title) +
    ``filter_and_weight_documents``.
    """
    dataset = download_corpus(entry, cache_dir=cache_dir, streaming=streaming)
    docs = _extract_text_title_rows(dataset, sample_limit)
    return filter_and_weight_documents(docs, oversample_weight=oversample_weight)
