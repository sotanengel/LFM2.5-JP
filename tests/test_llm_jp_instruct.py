"""llm-jp-instruct dataset acquisition + conversion tests (Issue #105).

``llm-jp/oasst1-21k-ja`` is the fallback repo actually used here (the
originally-specified `llm-jp/llm-jp-instruct-v1` does not exist on the HF
Hub -- see the doc header of ``lfm25_ja.data.llm_jp_instruct`` for the
verification trail). These tests mock ``datasets.load_dataset`` the same way
``tests/test_ichikara.py`` does -- no network calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lfm25_ja.data.clean import _read_jsonl
from lfm25_ja.data.llm_jp_instruct import (
    LLM_JP_INSTRUCT_FILE,
    LLM_JP_INSTRUCT_HF_REPO,
    build_llm_jp_instruct_dataset,
    convert_llm_jp_instruct_record,
    download_llm_jp_instruct_raw,
    prepare_llm_jp_instruct,
    sample_rows,
)

# ---------------------------------------------------------------------------
# convert_llm_jp_instruct_record
# ---------------------------------------------------------------------------


def test_convert_conversations_record_maps_human_gpt_to_user_assistant() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "こんにちは"},
            {"from": "gpt", "value": "こんにちは！ご質問はありますか？"},
        ]
    }
    result = convert_llm_jp_instruct_record(record)
    assert result == {
        "messages": [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは！ご質問はありますか？"},
        ]
    }


def test_convert_conversations_record_preserves_multi_turn_order() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "質問1"},
            {"from": "gpt", "value": "回答1"},
            {"from": "human", "value": "質問2"},
            {"from": "gpt", "value": "回答2"},
        ]
    }
    result = convert_llm_jp_instruct_record(record)
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert result["messages"][-1]["content"] == "回答2"


def test_convert_conversations_record_unknown_role_raises() -> None:
    record = {"conversations": [{"from": "system", "value": "x"}, {"from": "gpt", "value": "y"}]}
    with pytest.raises(ValueError, match="system"):
        convert_llm_jp_instruct_record(record)


def test_convert_conversations_record_empty_raises() -> None:
    with pytest.raises(ValueError):
        convert_llm_jp_instruct_record({"conversations": []})


def test_convert_instruction_record_maps_to_messages() -> None:
    record = {"instruction": "日本の首都は？", "input": "", "output": "東京です。"}
    result = convert_llm_jp_instruct_record(record)
    assert result == {
        "messages": [
            {"role": "user", "content": "日本の首都は？"},
            {"role": "assistant", "content": "東京です。"},
        ]
    }


def test_convert_instruction_record_includes_input_when_present() -> None:
    record = {"instruction": "要約してください", "input": "長い文章です", "output": "短い要約"}
    result = convert_llm_jp_instruct_record(record)
    assert "長い文章です" in result["messages"][0]["content"]


def test_convert_record_missing_all_known_schemas_raises() -> None:
    with pytest.raises(ValueError):
        convert_llm_jp_instruct_record({"foo": "bar"})


# ---------------------------------------------------------------------------
# build_llm_jp_instruct_dataset
# ---------------------------------------------------------------------------


def test_build_llm_jp_instruct_dataset_converts_every_record() -> None:
    raw_records = [
        {"conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}]},
        {"conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}]},
    ]
    rows = build_llm_jp_instruct_dataset(raw_records)
    assert len(rows) == 2


def test_build_llm_jp_instruct_dataset_skips_malformed_records() -> None:
    raw_records = [
        {"conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}]},
        {"conversations": [{"from": "human", "value": "no answer"}]},
    ]
    rows = build_llm_jp_instruct_dataset(raw_records)
    assert len(rows) == 1


def test_build_llm_jp_instruct_dataset_empty_input_returns_empty_list() -> None:
    assert build_llm_jp_instruct_dataset([]) == []


# ---------------------------------------------------------------------------
# sample_rows
# ---------------------------------------------------------------------------


def test_sample_rows_returns_requested_count() -> None:
    rows = [{"id": i} for i in range(100)]
    sampled = sample_rows(rows, n_samples=10, seed=42)
    assert len(sampled) == 10


def test_sample_rows_is_deterministic_given_seed() -> None:
    rows = [{"id": i} for i in range(100)]
    sampled_a = sample_rows(rows, n_samples=10, seed=42)
    sampled_b = sample_rows(rows, n_samples=10, seed=42)
    assert sampled_a == sampled_b


def test_sample_rows_returns_all_when_n_samples_exceeds_available() -> None:
    rows = [{"id": i} for i in range(5)]
    sampled = sample_rows(rows, n_samples=100, seed=42)
    assert len(sampled) == 5
    assert {r["id"] for r in sampled} == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# download_llm_jp_instruct_raw
# ---------------------------------------------------------------------------


@patch("lfm25_ja.data.llm_jp_instruct.datasets.load_dataset")
def test_download_llm_jp_instruct_raw_passes_resolved_url_and_cache_dir(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = [
        {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
    ]
    result = download_llm_jp_instruct_raw(cache_dir="data/raw/llm_jp_instruct")

    args, kwargs = mock_load.call_args
    assert args[0] == "json"
    urls = kwargs["data_files"]
    assert len(urls) == 1
    assert LLM_JP_INSTRUCT_HF_REPO in urls[0]
    assert LLM_JP_INSTRUCT_FILE in urls[0]
    assert kwargs["split"] == "train"
    assert kwargs["cache_dir"] == "data/raw/llm_jp_instruct"
    assert len(result) == 1


@patch("lfm25_ja.data.llm_jp_instruct.datasets.load_dataset")
def test_download_llm_jp_instruct_raw_error_wraps_with_context(mock_load: MagicMock) -> None:
    mock_load.side_effect = OSError("network unreachable")
    with pytest.raises(RuntimeError, match="llm-jp-instruct"):
        download_llm_jp_instruct_raw(cache_dir="data/raw/llm_jp_instruct")


# ---------------------------------------------------------------------------
# prepare_llm_jp_instruct (end-to-end, download mocked)
# ---------------------------------------------------------------------------


@patch("lfm25_ja.data.llm_jp_instruct.download_llm_jp_instruct_raw")
def test_prepare_llm_jp_instruct_writes_sampled_chat_format_jsonl(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = [
        {
            "conversations": [
                {"from": "human", "value": f"質問{i}"},
                {"from": "gpt", "value": f"回答{i}"},
            ]
        }
        for i in range(20)
    ]
    output_path = tmp_path / "llm_jp_instruct.jsonl"

    result = prepare_llm_jp_instruct(
        output_path=output_path, cache_dir=str(tmp_path / "raw"), n_samples=5, seed=42
    )

    assert result["input_count"] == 20
    assert result["converted_count"] == 20
    assert result["output_count"] == 5
    rows = _read_jsonl(output_path)
    assert len(rows) == 5
    assert rows[0]["messages"][0]["role"] == "user"
    assert rows[0]["messages"][1]["role"] == "assistant"


@patch("lfm25_ja.data.llm_jp_instruct.download_llm_jp_instruct_raw")
def test_prepare_llm_jp_instruct_uses_all_when_fewer_than_requested(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = [
        {"conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}]},
        {"conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}]},
    ]
    output_path = tmp_path / "llm_jp_instruct.jsonl"

    result = prepare_llm_jp_instruct(
        output_path=output_path, cache_dir=str(tmp_path / "raw"), n_samples=4000, seed=42
    )

    assert result["output_count"] == 2
