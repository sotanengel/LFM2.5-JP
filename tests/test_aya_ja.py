"""aya_dataset (Japanese subset) acquisition + conversion tests (Issue #105).

``CohereForAI/aya_dataset`` is confirmed non-gated (``gated: false``, license
``apache-2.0``) via the HF API -- see the doc header of
``lfm25_ja.data.aya_ja`` for the verification trail. These tests mock
``datasets.load_dataset`` the same way ``tests/test_ichikara.py`` does -- no
network calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lfm25_ja.data.aya_ja import (
    AYA_HF_REPO,
    build_aya_ja_dataset,
    convert_aya_record,
    download_aya_raw,
    filter_japanese_records,
    prepare_aya_ja,
    sample_rows,
)

from lfm25_ja.data.clean import _read_jsonl

# ---------------------------------------------------------------------------
# filter_japanese_records
# ---------------------------------------------------------------------------


def test_filter_japanese_records_keeps_only_jpn_language_code() -> None:
    records = [
        {"inputs": "hi", "targets": "hello", "language": "English", "language_code": "eng"},
        {
            "inputs": "こんにちは",
            "targets": "こんにちは！",
            "language": "Japanese",
            "language_code": "jpn",
        },
    ]
    result = filter_japanese_records(records)
    assert len(result) == 1
    assert result[0]["language_code"] == "jpn"


def test_filter_japanese_records_falls_back_to_language_field() -> None:
    records = [{"inputs": "a", "targets": "b", "language": "Japanese"}]
    result = filter_japanese_records(records)
    assert len(result) == 1


def test_filter_japanese_records_empty_input_returns_empty_list() -> None:
    assert filter_japanese_records([]) == []


# ---------------------------------------------------------------------------
# convert_aya_record
# ---------------------------------------------------------------------------


def test_convert_aya_record_maps_inputs_targets_to_messages() -> None:
    record = {"inputs": "日本の首都はどこですか？", "targets": "東京です。", "language_code": "jpn"}
    result = convert_aya_record(record)
    assert result == {
        "messages": [
            {"role": "user", "content": "日本の首都はどこですか？"},
            {"role": "assistant", "content": "東京です。"},
        ]
    }


def test_convert_aya_record_missing_inputs_raises() -> None:
    with pytest.raises(ValueError, match="inputs"):
        convert_aya_record({"targets": "東京です。"})


def test_convert_aya_record_missing_targets_raises() -> None:
    with pytest.raises(ValueError, match="targets"):
        convert_aya_record({"inputs": "日本の首都は？"})


# ---------------------------------------------------------------------------
# build_aya_ja_dataset
# ---------------------------------------------------------------------------


def test_build_aya_ja_dataset_filters_and_converts() -> None:
    raw_records = [
        {"inputs": "hi", "targets": "hello", "language_code": "eng"},
        {"inputs": "質問1", "targets": "回答1", "language_code": "jpn"},
        {"inputs": "質問2", "targets": "回答2", "language_code": "jpn"},
    ]
    rows = build_aya_ja_dataset(raw_records)
    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "質問1"


def test_build_aya_ja_dataset_empty_input_returns_empty_list() -> None:
    assert build_aya_ja_dataset([]) == []


# ---------------------------------------------------------------------------
# sample_rows
# ---------------------------------------------------------------------------


def test_sample_rows_deterministic_and_bounded() -> None:
    rows = [{"id": i} for i in range(50)]
    a = sample_rows(rows, n_samples=10, seed=42)
    b = sample_rows(rows, n_samples=10, seed=42)
    assert len(a) == 10
    assert a == b


# ---------------------------------------------------------------------------
# download_aya_raw
# ---------------------------------------------------------------------------


@patch("lfm25_ja.data.aya_ja.datasets.load_dataset")
def test_download_aya_raw_passes_repo_split_and_cache_dir(mock_load: MagicMock) -> None:
    mock_load.return_value = [{"inputs": "q", "targets": "a", "language_code": "jpn"}]
    result = download_aya_raw(cache_dir="data/raw/aya_ja")

    args, kwargs = mock_load.call_args
    assert args[0] == AYA_HF_REPO
    assert kwargs["split"] == "train"
    assert kwargs["cache_dir"] == "data/raw/aya_ja"
    assert len(result) == 1


@patch("lfm25_ja.data.aya_ja.datasets.load_dataset")
def test_download_aya_raw_error_wraps_with_context(mock_load: MagicMock) -> None:
    mock_load.side_effect = OSError("network unreachable")
    with pytest.raises(RuntimeError, match="aya_dataset"):
        download_aya_raw(cache_dir="data/raw/aya_ja")


# ---------------------------------------------------------------------------
# prepare_aya_ja (end-to-end, download mocked)
# ---------------------------------------------------------------------------


@patch("lfm25_ja.data.aya_ja.download_aya_raw")
def test_prepare_aya_ja_writes_sampled_japanese_chat_format_jsonl(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = [
        {"inputs": f"q{i}", "targets": f"a{i}", "language_code": "eng"} for i in range(10)
    ] + [{"inputs": f"質問{i}", "targets": f"回答{i}", "language_code": "jpn"} for i in range(10)]
    output_path = tmp_path / "aya_ja.jsonl"

    result = prepare_aya_ja(
        output_path=output_path, cache_dir=str(tmp_path / "raw"), n_samples=5, seed=42
    )

    assert result["input_count"] == 20
    assert result["ja_count"] == 10
    assert result["output_count"] == 5
    rows = _read_jsonl(output_path)
    assert len(rows) == 5
    for row in rows:
        assert row["messages"][0]["content"].startswith("質問")


@patch("lfm25_ja.data.aya_ja.download_aya_raw")
def test_prepare_aya_ja_uses_all_when_fewer_than_requested(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = [{"inputs": "質問1", "targets": "回答1", "language_code": "jpn"}]
    output_path = tmp_path / "aya_ja.jsonl"

    result = prepare_aya_ja(
        output_path=output_path, cache_dir=str(tmp_path / "raw"), n_samples=1500, seed=42
    )

    assert result["output_count"] == 1
