"""ichikara-instruction dataset acquisition + conversion tests (Issue #33).

``kinokokoro/ichikara-instruction-003`` is a non-gated HuggingFace Hub
mirror of the CC-BY-NC-SA-licensed ichikara-instruction release (verified
via the HF API: ``gated: false``), so it can be pulled with the ordinary
``datasets.load_dataset`` flow -- these tests mock that call the same way
``tests/test_data_pipeline.py`` mocks ``lfm25_ja.data.download``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lfm25_ja.data.clean import _read_jsonl
from lfm25_ja.data.ichikara import (
    ICHIKARA_FILES,
    ICHIKARA_HF_REPO,
    build_ichikara_dataset,
    convert_ichikara_record,
    download_ichikara_raw,
    prepare_ichikara,
)

# ---------------------------------------------------------------------------
# convert_ichikara_record
# ---------------------------------------------------------------------------


def test_convert_ichikara_record_maps_text_and_output_to_messages() -> None:
    record = {"ID": "ichikara-003-1", "text": "日本の首都はどこですか？", "output": "東京です。"}
    result = convert_ichikara_record(record)
    assert result == {
        "messages": [
            {"role": "user", "content": "日本の首都はどこですか？"},
            {"role": "assistant", "content": "東京です。"},
        ]
    }


def test_convert_ichikara_record_missing_text_raises() -> None:
    with pytest.raises(ValueError, match="text"):
        convert_ichikara_record({"output": "東京です。"})


def test_convert_ichikara_record_missing_output_raises() -> None:
    with pytest.raises(ValueError, match="output"):
        convert_ichikara_record({"text": "日本の首都はどこですか？"})


# ---------------------------------------------------------------------------
# build_ichikara_dataset
# ---------------------------------------------------------------------------


def test_build_ichikara_dataset_converts_every_record() -> None:
    raw_records = [
        {"text": "質問1", "output": "回答1"},
        {"text": "質問2", "output": "回答2"},
    ]
    rows = build_ichikara_dataset(raw_records)
    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "質問1"
    assert rows[1]["messages"][1]["content"] == "回答2"


def test_build_ichikara_dataset_empty_input_returns_empty_list() -> None:
    assert build_ichikara_dataset([]) == []


# ---------------------------------------------------------------------------
# download_ichikara_raw
# ---------------------------------------------------------------------------


@patch("lfm25_ja.data.ichikara.datasets.load_dataset")
def test_download_ichikara_raw_passes_resolved_urls_and_cache_dir(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = [
        {"ID": "1", "text": "q1", "output": "a1"},
        {"ID": "2", "text": "q2", "output": "a2"},
    ]
    result = download_ichikara_raw(cache_dir="data/raw/ichikara")

    args, kwargs = mock_load.call_args
    assert args[0] == "json"
    urls = kwargs["data_files"]
    assert len(urls) == len(ICHIKARA_FILES)
    assert all(ICHIKARA_HF_REPO in url for url in urls)
    assert all(url.startswith("https://huggingface.co/datasets/") for url in urls)
    assert kwargs["split"] == "train"
    assert kwargs["cache_dir"] == "data/raw/ichikara"
    assert result == [
        {"ID": "1", "text": "q1", "output": "a1"},
        {"ID": "2", "text": "q2", "output": "a2"},
    ]


@patch("lfm25_ja.data.ichikara.datasets.load_dataset")
def test_download_ichikara_raw_error_wraps_with_context(mock_load: MagicMock) -> None:
    mock_load.side_effect = OSError("network unreachable")
    with pytest.raises(RuntimeError, match="ichikara"):
        download_ichikara_raw(cache_dir="data/raw/ichikara")


# ---------------------------------------------------------------------------
# prepare_ichikara (end-to-end, download mocked)
# ---------------------------------------------------------------------------


@patch("lfm25_ja.data.ichikara.download_ichikara_raw")
def test_prepare_ichikara_writes_chat_format_jsonl(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = [
        {"ID": "1", "text": "質問1", "output": "回答1"},
        {"ID": "2", "text": "質問2", "output": "回答2"},
    ]
    output_path = tmp_path / "ichikara.jsonl"

    result = prepare_ichikara(output_path=output_path, cache_dir=str(tmp_path / "raw"))

    assert result["input_count"] == 2
    assert result["output_count"] == 2
    assert result["output_path"] == str(output_path)

    rows = _read_jsonl(output_path)
    assert len(rows) == 2
    assert rows[0]["messages"][0]["role"] == "user"
    assert rows[0]["messages"][1]["role"] == "assistant"


@patch("lfm25_ja.data.ichikara.download_ichikara_raw")
def test_prepare_ichikara_skips_malformed_records_and_reports_counts(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = [
        {"text": "質問1", "output": "回答1"},
        {"text": "質問だけで回答なし"},  # malformed: missing "output"
    ]
    output_path = tmp_path / "ichikara.jsonl"

    result = prepare_ichikara(output_path=output_path, cache_dir=str(tmp_path / "raw"))

    assert result["input_count"] == 2
    assert result["output_count"] == 1
    rows = _read_jsonl(output_path)
    assert len(rows) == 1
