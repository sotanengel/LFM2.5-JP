"""Packed CPT dataset cache and package selection (Issues #71 / #72)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from lfm25_ja.data.clean import _write_jsonl
from lfm25_ja.train.packed_cache import (
    PACKAGES,
    apply_package,
    build_or_load_packed,
    cache_is_valid,
    packed_cache_dir,
    save_packed_cache,
)


def _packed_rows(n: int, seq_len: int = 4) -> list[dict[str, list[int]]]:
    return [
        {
            "input_ids": list(range(i * seq_len, (i + 1) * seq_len)),
            "labels": list(range(i * seq_len, (i + 1) * seq_len)),
            "attention_mask": [1] * seq_len,
        }
        for i in range(n)
    ]


def test_packed_cache_dir_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "mixture.jsonl"
    source.write_text("{}", encoding="utf-8")
    a = packed_cache_dir(source, "LiquidAI/LFM2.5-1.2B-Base", 6144, tmp_path / "packed")
    b = packed_cache_dir(source, "LiquidAI/LFM2.5-1.2B-Base", 6144, tmp_path / "packed")
    assert a == b
    assert "mixture" in a.name
    assert "seq6144" in a.name


def test_save_and_cache_is_valid_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "mixture.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")
    cache_dir = packed_cache_dir(source, "model/x", 8, tmp_path / "packed")
    packed = _packed_rows(5, seq_len=8)
    save_packed_cache(cache_dir, packed, source, "model/x", 8)

    assert cache_is_valid(cache_dir, source, "model/x", 8)
    loaded = torch.load(cache_dir / "packed.pt", weights_only=False)
    assert loaded == packed
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_sequences"] == 5
    assert manifest["seq_len"] == 8


def test_cache_invalid_when_source_mtime_changes(tmp_path: Path) -> None:
    source = tmp_path / "mixture.jsonl"
    source.write_text('{"text":"a"}\n', encoding="utf-8")
    cache_dir = packed_cache_dir(source, "model/x", 4, tmp_path / "packed")
    save_packed_cache(cache_dir, _packed_rows(2), source, "model/x", 4)

    source.write_text('{"text":"updated"}\n', encoding="utf-8")
    assert not cache_is_valid(cache_dir, source, "model/x", 4)


def test_apply_package_full_and_centi() -> None:
    packed = _packed_rows(200)
    assert len(apply_package(packed, "full")) == 200
    assert len(apply_package(packed, "centi")) == 2
    assert len(apply_package(_packed_rows(1), "centi")) == 1


def test_apply_package_deci() -> None:
    packed = _packed_rows(200)
    assert len(apply_package(packed, "deci")) == 20
    # at least one row for smoke runs, same rule as centi
    assert len(apply_package(_packed_rows(1), "deci")) == 1
    assert len(apply_package(_packed_rows(5), "deci")) == 1


def test_apply_package_deci_is_deterministic_prefix() -> None:
    packed = _packed_rows(50)
    first = apply_package(packed, "deci")
    second = apply_package(packed, "deci")
    assert first == second == packed[:5]


def test_apply_package_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="package"):
        apply_package(_packed_rows(3), "half")


def test_packages_constant() -> None:
    assert PACKAGES == ("full", "centi", "deci")


class _CountingTokenizer:
    def __init__(self) -> None:
        self.calls = 0
        self.eos_token_id = 0

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        self.calls += 1
        return {"input_ids": [self.calls, self.calls + 1]}


def test_build_or_load_packed_uses_cache_on_second_call(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "mixture.jsonl"
    _write_jsonl(jsonl_path, [{"text": "one"}, {"text": "two"}, {"text": "three four"}])
    tokenizer = _CountingTokenizer()
    cache_root = tmp_path / "packed"

    first = build_or_load_packed(
        jsonl_path, tokenizer, seq_len=4, model_name="model/x", cache_root=cache_root
    )
    assert tokenizer.calls == 3
    assert len(first) >= 1

    tokenizer.calls = 0
    second = build_or_load_packed(
        jsonl_path, tokenizer, seq_len=4, model_name="model/x", cache_root=cache_root
    )
    assert tokenizer.calls == 0
    assert second == first


def test_build_or_load_packed_rebuild_forces_tokenization(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "mixture.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"text": "one"},
            {"text": "two"},
            {"text": "three"},
            {"text": "four"},
        ],
    )
    tokenizer = _CountingTokenizer()
    cache_root = tmp_path / "packed"

    build_or_load_packed(
        jsonl_path, tokenizer, seq_len=4, model_name="model/x", cache_root=cache_root
    )
    tokenizer.calls = 0
    build_or_load_packed(
        jsonl_path,
        tokenizer,
        seq_len=4,
        model_name="model/x",
        cache_root=cache_root,
        rebuild=True,
    )
    assert tokenizer.calls == 4


def test_build_or_load_packed_delegates_to_build_cpt_dataset_when_missing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "mixture.jsonl"
    _write_jsonl(jsonl_path, [{"text": "hello"}])
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.return_value = {"input_ids": [1, 2, 3]}
    expected = _packed_rows(1)
    monkeypatch.setattr(
        "lfm25_ja.train.train_cpt.build_cpt_dataset", lambda *_a, **_k: expected
    )

    loaded = build_or_load_packed(
        jsonl_path, tokenizer, seq_len=4, model_name="m", cache_root=tmp_path / "packed"
    )
    assert loaded == expected
    assert (tmp_path / "packed").exists()
