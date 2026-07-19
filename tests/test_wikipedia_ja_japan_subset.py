"""Tests for the K1-worst-10-cell-targeted Wikipedia-ja subset builder (Issue #130).

Covers ``filter_and_weight_documents`` (keyword match + oversample duplication)
and ``load_and_filter_wikipedia_ja`` (download_corpus + extraction + filter wiring),
following the fixture-based, no-network-call testing style used in
tests/test_data_pipeline.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lfm25_ja.data.wikipedia_ja_japan_subset import (
    WORST10_CELL_KEYWORDS,
    filter_and_weight_documents,
    filter_matching_documents,
    load_and_filter_wikipedia_ja,
    oversample_documents,
)

# ---------------------------------------------------------------------------
# WORST10_CELL_KEYWORDS: sanity checks on the transcribed constant
# ---------------------------------------------------------------------------


def test_worst10_cell_keywords_has_ten_cells() -> None:
    assert len(WORST10_CELL_KEYWORDS) == 10


def test_worst10_cell_keywords_values_are_nonempty_keyword_lists() -> None:
    for cell_id, keywords in WORST10_CELL_KEYWORDS.items():
        assert isinstance(cell_id, str)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(kw, str) for kw in keywords)


# ---------------------------------------------------------------------------
# filter_and_weight_documents
# ---------------------------------------------------------------------------


def test_filter_matches_on_text_substring() -> None:
    docs = [{"id": "1", "text": "冠婚葬祭のマナーについて解説する記事です。"}]
    result = filter_and_weight_documents(docs, oversample_weight=1)
    assert len(result) == 1
    assert result[0]["matched_cells"] == ["生活・慣習_advanced"]


def test_filter_matches_on_title_substring() -> None:
    docs = [{"id": "1", "title": "還暦祝いの由来", "text": "本文には無関係な内容のみ含む。"}]
    result = filter_and_weight_documents(docs, oversample_weight=1)
    assert len(result) == 1
    assert result[0]["matched_cells"] == ["生活・慣習_advanced"]


def test_filter_drops_non_matching_documents() -> None:
    docs = [
        {"id": "1", "text": "関係のない一般的な文章です。"},
        {"id": "2", "text": "これも無関係な文章です。"},
    ]
    result = filter_and_weight_documents(docs, oversample_weight=1)
    assert result == []


def test_filter_empty_docs_list_returns_empty() -> None:
    assert filter_and_weight_documents([], oversample_weight=3) == []


def test_filter_doc_without_title_key_matches_on_text_only_no_keyerror() -> None:
    docs = [{"id": "1", "text": "冠婚葬祭について説明します。"}]
    result = filter_and_weight_documents(docs, oversample_weight=1)
    assert len(result) == 1
    assert result[0]["matched_cells"] == ["生活・慣習_advanced"]


def test_filter_oversample_weight_one_means_no_duplication() -> None:
    docs = [{"id": "1", "text": "冠婚葬祭について説明します。"}]
    result = filter_and_weight_documents(docs, oversample_weight=1)
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_oversample_weight_three_duplicates_with_suffixed_ids() -> None:
    docs = [{"id": "1", "text": "冠婚葬祭について説明します。"}]
    result = filter_and_weight_documents(docs, oversample_weight=3)
    assert len(result) == 3
    ids = [d["id"] for d in result]
    assert ids == ["1", "1-dup1", "1-dup2"]
    # All copies carry the same tag and text content.
    for d in result:
        assert d["matched_cells"] == ["生活・慣習_advanced"]
        assert d["text"] == docs[0]["text"]


def test_filter_matching_multiple_cells_lists_all_but_duplicates_once_per_weight() -> None:
    # "日本国憲法" -> 政治・制度_advanced ; "冠婚葬祭" -> 生活・慣習_advanced
    docs = [{"id": "1", "text": "日本国憲法と冠婚葬祭についての記事。"}]
    result = filter_and_weight_documents(docs, oversample_weight=3)
    assert len(result) == 3  # not 3 * 2 matched cells
    for d in result:
        assert d["matched_cells"] == ["政治・制度_advanced", "生活・慣習_advanced"]


def test_filter_preserves_input_order_among_matched_docs() -> None:
    docs = [
        {"id": "a", "text": "無関係な文章。"},
        {"id": "b", "text": "冠婚葬祭についての記事。"},
        {"id": "c", "text": "還暦のお祝いについての記事。"},
        {"id": "d", "text": "また無関係な文章。"},
    ]
    result = filter_and_weight_documents(docs, oversample_weight=1)
    assert [d["id"] for d in result] == ["b", "c"]


def test_filter_default_keyword_groups_is_worst10() -> None:
    docs = [{"id": "1", "text": "冠婚葬祭について説明します。"}]
    result = filter_and_weight_documents(docs)
    # default oversample_weight is 3 per signature
    assert len(result) == 3


# ---------------------------------------------------------------------------
# filter_matching_documents (Issue #123 bugfix: split out of
# filter_and_weight_documents so duplication can move to *after* clean_corpus's
# dedup stage -- see oversample_documents below).
# ---------------------------------------------------------------------------


def test_filter_matching_documents_filters_and_tags_without_duplicating() -> None:
    docs = [
        {"id": "1", "text": "冠婚葬祭について説明します。"},
        {"id": "2", "text": "関係のない一般的な文章です。"},
        {"id": "3", "title": "還暦祝いの由来", "text": "本文には無関係な内容のみ含む。"},
    ]
    result = filter_matching_documents(docs)
    # Only the 2 matching docs are kept, and each appears exactly once (no
    # oversampling / duplication at this stage).
    assert [d["id"] for d in result] == ["1", "3"]
    assert result[0]["matched_cells"] == ["生活・慣習_advanced"]
    assert result[1]["matched_cells"] == ["生活・慣習_advanced"]


def test_filter_matching_documents_preserves_input_order() -> None:
    docs = [
        {"id": "a", "text": "無関係な文章。"},
        {"id": "b", "text": "冠婚葬祭についての記事。"},
        {"id": "c", "text": "還暦のお祝いについての記事。"},
        {"id": "d", "text": "また無関係な文章。"},
    ]
    result = filter_matching_documents(docs)
    assert [d["id"] for d in result] == ["b", "c"]


def test_filter_matching_documents_empty_docs_list_returns_empty() -> None:
    assert filter_matching_documents([]) == []


def test_filter_matching_documents_has_no_oversample_weight_param() -> None:
    import inspect

    sig = inspect.signature(filter_matching_documents)
    assert "oversample_weight" not in sig.parameters


def test_filter_matching_documents_default_keyword_groups_is_worst10() -> None:
    docs = [{"id": "1", "text": "冠婚葬祭について説明します。"}]
    result = filter_matching_documents(docs)
    assert len(result) == 1
    assert result[0]["matched_cells"] == ["生活・慣習_advanced"]


# ---------------------------------------------------------------------------
# oversample_documents (pure duplication, independent of keyword logic)
# ---------------------------------------------------------------------------


def test_oversample_documents_default_weight_is_three() -> None:
    docs = [{"id": "1", "text": "anything"}]
    result = oversample_documents(docs)
    assert len(result) == 3
    assert [d["id"] for d in result] == ["1", "1-dup1", "1-dup2"]


def test_oversample_documents_weight_one_is_no_duplication() -> None:
    docs = [{"id": "1", "text": "anything"}]
    result = oversample_documents(docs, weight=1)
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_oversample_documents_works_on_arbitrary_docs_without_keyword_fields() -> None:
    # No "matched_cells"/"title"/keyword-matching involved at all -- pure
    # duplication over any docs that carry an "id".
    docs = [{"id": "x", "text": "foo"}, {"id": "y", "text": "bar"}]
    result = oversample_documents(docs, weight=2)
    assert [d["id"] for d in result] == ["x", "x-dup1", "y", "y-dup1"]
    assert [d["text"] for d in result] == ["foo", "foo", "bar", "bar"]


def test_oversample_documents_empty_list_returns_empty() -> None:
    assert oversample_documents([], weight=3) == []


def test_filter_and_weight_documents_equals_filter_then_oversample_composed() -> None:
    """filter_and_weight_documents is now a thin wrapper around the two new
    functions -- assert it produces exactly the same result as calling them
    in sequence."""
    docs = [
        {"id": "1", "text": "冠婚葬祭について説明します。"},
        {"id": "2", "text": "サッカーは無関係。"},
        {"id": "3", "text": "還暦のお祝い。"},
    ]
    combined = filter_and_weight_documents(docs, oversample_weight=2)
    manual = oversample_documents(filter_matching_documents(docs), weight=2)
    assert combined == manual


# ---------------------------------------------------------------------------
# load_and_filter_wikipedia_ja
# ---------------------------------------------------------------------------


def _fake_wikipedia_rows() -> list[dict]:
    return [
        {"title": "冠婚葬祭", "text": "冠婚葬祭に関する日本の慣習について説明する。"},
        {"title": "サッカー", "text": "サッカーは世界中で人気のスポーツである。"},
        {"title": "還暦", "text": "還暦は数え年61歳を祝う日本の伝統行事である。"},
    ]


@patch("lfm25_ja.data.wikipedia_ja_japan_subset.download_corpus")
def test_load_and_filter_wikipedia_ja_filters_and_weights(mock_download: MagicMock) -> None:
    mock_download.return_value = _fake_wikipedia_rows()
    entry = {
        "name": "wikipedia_ja_japan_subset",
        "hf_id": "wikimedia/wikipedia",
        "hf_config": "20231101.ja",
        "split": "train",
    }
    result = load_and_filter_wikipedia_ja(
        entry, cache_dir="data/raw", oversample_weight=2
    )
    # 2 of the 3 fake rows match (冠婚葬祭, 還暦); サッカー doesn't -> dropped.
    matched_ids = {d["id"].split("-dup")[0] for d in result}
    assert matched_ids == {"0", "2"}
    assert len(result) == 4  # 2 matching docs * oversample_weight 2
    mock_download.assert_called_once()
    _, kwargs = mock_download.call_args
    assert kwargs["cache_dir"] == "data/raw"


@patch("lfm25_ja.data.wikipedia_ja_japan_subset.download_corpus")
def test_load_and_filter_wikipedia_ja_respects_sample_limit(mock_download: MagicMock) -> None:
    mock_download.return_value = _fake_wikipedia_rows()
    entry = {"name": "wikipedia_ja_japan_subset", "hf_id": "wikimedia/wikipedia", "split": "train"}
    result = load_and_filter_wikipedia_ja(entry, cache_dir="data/raw", sample_limit=1)
    # Only the first row (冠婚葬祭, matching) is consumed.
    assert {d["id"].split("-dup")[0] for d in result} == {"0"}


@patch("lfm25_ja.data.wikipedia_ja_japan_subset.download_corpus")
def test_load_and_filter_wikipedia_ja_forwards_streaming_flag(mock_download: MagicMock) -> None:
    mock_download.return_value = _fake_wikipedia_rows()
    entry = {"name": "wikipedia_ja_japan_subset", "hf_id": "wikimedia/wikipedia", "split": "train"}
    load_and_filter_wikipedia_ja(entry, cache_dir="data/raw", streaming=True, sample_limit=2)
    _, kwargs = mock_download.call_args
    assert kwargs["streaming"] is True
