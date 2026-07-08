"""Data pipeline tests: download (Issue #16), clean + contamination (Issue #17 / #22)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lfm25_ja.data.clean import (
    MinHashDeduplicator,
    clean_corpus,
    detect_language,
    length_filter,
    ngram_contamination_checker,
    normalize_nfkc,
    remove_control_chars,
    render_stats_report,
)
from lfm25_ja.data.download import download_all, download_corpus, load_corpus_config

CORPUS_YAML = """
cache_dir: data/raw

corpora:
  - name: wikipedia_ja
    hf_id: wikimedia/wikipedia
    hf_config: 20231101.ja
    split: train
    language: ja
  - name: aozora
    hf_id: globis-university/aozorabunko-clean
    split: train
    language: ja

clean:
  min_chars: 10
  max_chars: 1000
  lang_threshold: 0.5
  minhash:
    num_perm: 64
    threshold: 0.7
    ngram: 5
  contamination:
    ngram: 5
    threshold: 0.5
"""


@pytest.fixture
def corpus_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.yaml"
    path.write_text(CORPUS_YAML, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# download.py (Issue #16)
# ---------------------------------------------------------------------------


def test_load_corpus_config_reads_corpora_and_clean_sections(corpus_config_path: Path) -> None:
    cfg = load_corpus_config(corpus_config_path)
    assert cfg["cache_dir"] == "data/raw"
    assert len(cfg["corpora"]) == 2
    assert cfg["clean"]["min_chars"] == 10


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_passes_hf_config_as_second_positional_arg(mock_load: MagicMock) -> None:
    entry = {
        "name": "wikipedia_ja",
        "hf_id": "wikimedia/wikipedia",
        "hf_config": "20231101.ja",
        "split": "train",
    }
    download_corpus(entry, cache_dir="data/raw")
    args, kwargs = mock_load.call_args
    assert args[0] == "wikimedia/wikipedia"
    assert args[1] == "20231101.ja"
    assert kwargs["split"] == "train"
    assert kwargs["cache_dir"] == "data/raw"


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_without_hf_config_omits_second_arg(mock_load: MagicMock) -> None:
    entry = {"name": "aozora", "hf_id": "globis-university/aozorabunko-clean", "split": "train"}
    download_corpus(entry, cache_dir="data/raw")
    args, kwargs = mock_load.call_args
    assert args == ("globis-university/aozorabunko-clean",)
    assert kwargs["split"] == "train"


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_streaming_flag_forwarded(mock_load: MagicMock) -> None:
    entry = {"name": "aozora", "hf_id": "globis-university/aozorabunko-clean", "split": "train"}
    download_corpus(entry, cache_dir="data/raw", streaming=True)
    _, kwargs = mock_load.call_args
    assert kwargs["streaming"] is True


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_error_includes_corpus_name(mock_load: MagicMock) -> None:
    mock_load.side_effect = OSError("network unreachable")
    entry = {"name": "wikipedia_ja", "hf_id": "wikimedia/wikipedia", "hf_config": "20231101.ja"}
    with pytest.raises(RuntimeError, match="wikipedia_ja"):
        download_corpus(entry, cache_dir="data/raw")


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_all_downloads_only_requested_names(
    mock_load: MagicMock, corpus_config_path: Path
) -> None:
    mock_load.return_value = "dataset-stub"
    result = download_all(corpus_config_path, names=["aozora"])
    assert list(result.keys()) == ["aozora"]
    assert mock_load.call_count == 1


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_all_downloads_all_when_names_is_none(
    mock_load: MagicMock, corpus_config_path: Path
) -> None:
    mock_load.return_value = "dataset-stub"
    result = download_all(corpus_config_path)
    assert set(result.keys()) == {"wikipedia_ja", "aozora"}
    assert mock_load.call_count == 2


def test_download_all_unknown_name_raises(corpus_config_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown_corpus"):
        download_all(corpus_config_path, names=["unknown_corpus"])


# ---------------------------------------------------------------------------
# clean.py: normalization / control chars (Issue #17)
# ---------------------------------------------------------------------------


def test_normalize_nfkc_fullwidth_alnum_to_halfwidth() -> None:
    assert normalize_nfkc("ABC１２３") == "ABC123"


def test_normalize_nfkc_halfwidth_katakana_to_fullwidth() -> None:
    assert normalize_nfkc("ﾃｽﾄ") == "テスト"


def test_remove_control_chars_keeps_newline_and_tab() -> None:
    text = "line1\nline2\tindented"
    assert remove_control_chars(text) == text


def test_remove_control_chars_strips_control_and_private_use() -> None:
    text = "clean" + chr(0) + "text" + chr(7) + chr(0xE000) + "withjunk"
    result = remove_control_chars(text)
    assert chr(0) not in result
    assert chr(7) not in result
    assert chr(0xE000) not in result
    assert "clean" in result and "text" in result and "junk" in result


def test_detect_language_japanese_sentence() -> None:
    assert detect_language("これは日本語の文章です。テストを行います。") == "ja"


def test_detect_language_english_sentence() -> None:
    assert detect_language("This is a plain English sentence for testing purposes.") == "en"


def test_detect_language_empty_string_is_other() -> None:
    assert detect_language("") == "other"


# ---------------------------------------------------------------------------
# clean.py: length filter (Issue #17)
# ---------------------------------------------------------------------------


def test_length_filter_boundaries() -> None:
    assert length_filter("a" * 50, min_chars=50, max_chars=100) is True
    assert length_filter("a" * 100, min_chars=50, max_chars=100) is True
    assert length_filter("a" * 49, min_chars=50, max_chars=100) is False
    assert length_filter("a" * 101, min_chars=50, max_chars=100) is False


# ---------------------------------------------------------------------------
# clean.py: MinHash near-duplicate removal (Issue #17)
# ---------------------------------------------------------------------------


def test_minhash_deduplicator_flags_near_duplicate_and_keeps_unrelated() -> None:
    dedup = MinHashDeduplicator(num_perm=64, threshold=0.7, ngram=5)
    base = "吾輩は猫である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。" * 3
    near_duplicate = base + "。"  # trivially near-identical
    unrelated = "本日は晴天なり。遠足のお知らせをいたします。持ち物を確認してください。" * 3

    assert dedup.add_and_check("doc1", base) is False
    assert dedup.add_and_check("doc2", near_duplicate) is True
    assert dedup.add_and_check("doc3", unrelated) is False


# ---------------------------------------------------------------------------
# clean.py: n-gram contamination check (Issue #22)
# ---------------------------------------------------------------------------


def test_ngram_contamination_checker_flags_eval_overlap() -> None:
    eval_texts = ["日本の首都は東京都です。人口は約1400万人です。"]
    checker = ngram_contamination_checker(eval_texts, ngram=5)

    contaminated = (
        "日本の首都は東京都です。人口は約1400万人です。" + "これは学習データに混入した文です。"
    )
    clean_doc = "猫は液体である説が有名だ。柔軟な体を持つ動物として知られている。"

    assert checker.check(contaminated) > 0.5
    assert checker.check(clean_doc) < 0.1


def test_ngram_contamination_checker_empty_text_is_zero() -> None:
    checker = ngram_contamination_checker(["some eval text"], ngram=5)
    assert checker.check("") == 0.0


# ---------------------------------------------------------------------------
# clean.py: clean_corpus end-to-end (Issue #17 + #22)
# ---------------------------------------------------------------------------


def test_clean_corpus_end_to_end_stats_are_consistent() -> None:
    docs = [
        {"id": "1", "text": "これはテスト文書です。" * 5},  # valid ja, long enough
        {"id": "2", "text": "これはテスト文書です。" * 5},  # near-duplicate of #1
        {"id": "3", "text": "short"},  # too short
        {"id": "4", "text": "This is a valid English document for the mixing corpus." * 2},
        {"id": "5", "text": "1234567890"},  # digits only -> not ja/en, likely "other"
    ]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.7, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.5},
    }
    clean_docs, stats = clean_corpus(docs, cfg)

    assert stats["input_count"] == 5
    assert stats["output_count"] == len(clean_docs)
    assert stats["output_count"] < stats["input_count"]
    total_removed = sum(stage["removed"] for stage in stats["stages"])
    assert stats["input_count"] - total_removed == stats["output_count"]


def test_clean_corpus_removes_eval_contaminated_documents() -> None:
    eval_texts = ["日本の首都は東京都です。人口は約1400万人です。"]
    docs = [
        {"id": "1", "text": "日本の首都は東京都です。人口は約1400万人です。" * 2},
        {"id": "2", "text": "猫は液体である説が有名だ。柔軟な体を持つ動物として知られている。" * 2},
    ]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.9, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.3},
    }
    clean_docs, stats = clean_corpus(docs, cfg, eval_texts=eval_texts)
    remaining_ids = {d["id"] for d in clean_docs}
    assert "1" not in remaining_ids
    assert "2" in remaining_ids
    assert any(stage["name"] == "contamination_filter" for stage in stats["stages"])


def test_clean_corpus_language_mix_ratio_in_stats() -> None:
    """Stats should make it possible to compute the ja/en mixing ratio downstream."""
    docs = [
        {"id": "1", "text": "これは日本語のテスト文書です。" * 4},
        {"id": "2", "text": "This is an English test document used for the mix." * 3},
    ]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.9, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.5},
    }
    clean_docs, stats = clean_corpus(docs, cfg)
    languages = {d["language"] for d in clean_docs}
    assert languages == {"ja", "en"}
    assert stats["output_count"] == 2


def test_clean_corpus_output_docs_retain_text_field_for_chatml_stage() -> None:
    """Downstream ChatML formatting (format_chat.py) consumes the 'text' field untouched."""
    docs = [{"id": "1", "text": "これはテスト文書です。" * 5}]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.7, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.5},
    }
    clean_docs, _ = clean_corpus(docs, cfg)
    assert clean_docs[0]["text"] == docs[0]["text"]


# ---------------------------------------------------------------------------
# clean.py: markdown stats report
# ---------------------------------------------------------------------------


def test_render_stats_report_contains_key_numbers() -> None:
    stats = {
        "input_count": 10,
        "output_count": 6,
        "stages": [
            {"name": "length_filter", "removed": 2, "remaining": 8, "removal_rate": 0.2},
            {"name": "language_filter", "removed": 1, "remaining": 7, "removal_rate": 0.125},
            {"name": "dedup", "removed": 1, "remaining": 6, "removal_rate": 0.142857},
        ],
    }
    report = render_stats_report(stats)
    assert "length_filter" in report
    assert "10" in report
    assert "6" in report
    assert "|" in report  # markdown table
