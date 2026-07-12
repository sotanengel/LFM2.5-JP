"""SFT training tests: TRL SFTTrainer wrapper (Issue #31).

Covers config loading/validation, layer-selection (including the "all"
full-FT reference case), loss-masked dataset construction via
``format_chat.build_sft_example``, run-name composition, and a fast
CPU-only dry-run smoke test that a few training steps reduce loss.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch.nn as nn

from lfm25_ja.data.clean import _write_jsonl
from lfm25_ja.train.train_sft import (
    build_sft_dataset,
    build_sft_run_name,
    resolve_trainable_layer_indices,
    run_sft,
)
from lfm25_ja.utils.config import load_config, load_project_config, merge_configs

# ---------------------------------------------------------------------------
# MockSFTTokenizer -- deterministic whitespace tokenizer (no HF download),
# mirroring tests/test_data_pipeline.py's MockTokenizer for format_chat.
# ---------------------------------------------------------------------------


class MockSFTTokenizer:
    """Deterministic whitespace-level tokenizer with no ``apply_chat_template``
    attribute, so ``build_sft_example`` exercises the ``to_chatml()`` fallback.
    """

    _SPECIAL_RE = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inverse: dict[int, str] = {}
        self.eos_token_id = 0
        self._token_id("<eos>")

    def _token_id(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse[idx] = token
        return self.vocab[token]

    def encode(self, text: str) -> list[int]:
        tokens: list[str] = []
        for chunk in self._SPECIAL_RE.split(text):
            if not chunk:
                continue
            if chunk in ("<|im_start|>", "<|im_end|>"):
                tokens.append(chunk)
            else:
                tokens.extend(chunk.split())
        return [self._token_id(t) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inverse[i] for i in ids)

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}


# ---------------------------------------------------------------------------
# resolve_trainable_layer_indices
# ---------------------------------------------------------------------------


class _InnerStack(nn.Module):
    def __init__(self, n_layers: int = 4, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])


class _DummyHFLikeModel(nn.Module):
    """Mimics the `model.model.layers` layout used by HF causal LMs."""

    def __init__(self, n_layers: int = 4, dim: int = 4) -> None:
        super().__init__()
        self.model = _InnerStack(n_layers, dim)
        self.embed = nn.Embedding(10, dim)


def test_resolve_trainable_layer_indices_passthrough_list() -> None:
    model = _DummyHFLikeModel(n_layers=6)
    assert resolve_trainable_layer_indices([2, 3], model) == [2, 3]


def test_resolve_trainable_layer_indices_all_resolves_to_full_range() -> None:
    model = _DummyHFLikeModel(n_layers=6)
    assert resolve_trainable_layer_indices("all", model) == [0, 1, 2, 3, 4, 5]


def test_resolve_trainable_layer_indices_empty_list_passthrough() -> None:
    model = _DummyHFLikeModel(n_layers=4)
    assert resolve_trainable_layer_indices([], model) == []


# ---------------------------------------------------------------------------
# build_sft_dataset (JSONL of {"messages": [...]} -> loss-masked dataset)
# ---------------------------------------------------------------------------


def test_build_sft_dataset_reads_jsonl_and_masks_loss(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sft_data.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "hi friend"},
                ]
            }
        ],
    )
    tokenizer = MockSFTTokenizer()
    dataset = build_sft_dataset(str(jsonl_path), tokenizer, max_seq_len=100)

    assert len(dataset) == 1
    row = dataset[0]
    assert set(row.keys()) >= {"input_ids", "labels", "attention_mask"}
    assert len(row["labels"]) == len(row["input_ids"])

    # Only the assistant content tokens ("hi", "friend") should be unmasked.
    learned_tokens = {
        tokenizer.inverse[tok_id]
        for tok_id, label in zip(row["input_ids"], row["labels"])
        if label != -100
    }
    assert learned_tokens == {"hi", "friend"}
    # The user turn must stay fully masked.
    assert all(
        label == -100
        for tok_id, label in zip(row["input_ids"], row["labels"])
        if tokenizer.inverse[tok_id] in ("hello", "there")
    )


def test_build_sft_dataset_multiple_rows(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sft_data.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "c"},
                    {"role": "assistant", "content": "d"},
                ]
            },
        ],
    )
    tokenizer = MockSFTTokenizer()
    dataset = build_sft_dataset(str(jsonl_path), tokenizer, max_seq_len=100)
    assert len(dataset) == 2


def test_build_sft_dataset_truncates_to_max_seq_len(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sft_data.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "one two three four five"},
                    {"role": "assistant", "content": "six seven eight nine ten"},
                ]
            }
        ],
    )
    tokenizer = MockSFTTokenizer()
    dataset = build_sft_dataset(str(jsonl_path), tokenizer, max_seq_len=5)
    assert len(dataset[0]["input_ids"]) == 5


def test_build_sft_dataset_empty_input_raises(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        build_sft_dataset(str(jsonl_path), MockSFTTokenizer(), max_seq_len=100)


# ---------------------------------------------------------------------------
# run_sft(dry_run=True) -- CPU-only smoke test, no HF download
# ---------------------------------------------------------------------------


def test_run_sft_dry_run_loss_decreases_and_summary_consistent(tmp_path: Path) -> None:
    sft_config = tmp_path / "sft_test.yaml"
    sft_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: [1]
""",
        encoding="utf-8",
    )
    result = run_sft(str(sft_config), dry_run=True)

    assert result["final_loss"] < result["initial_loss"]
    assert len(result["losses"]) >= 2
    summary = result["trainable_summary"]
    assert 0 < summary["trainable_params"] < summary["total_params"]
    assert summary["trainable_pct"] == pytest.approx(
        summary["trainable_params"] / summary["total_params"] * 100.0
    )


def test_run_sft_dry_run_supports_all_layers(tmp_path: Path) -> None:
    single_layer_config = tmp_path / "sft_single_layer.yaml"
    single_layer_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: [1]
""",
        encoding="utf-8",
    )
    all_layers_config = tmp_path / "sft_full_ft.yaml"
    all_layers_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: all
""",
        encoding="utf-8",
    )
    single_result = run_sft(str(single_layer_config), dry_run=True)
    all_result = run_sft(str(all_layers_config), dry_run=True)

    assert all_result["final_loss"] < all_result["initial_loss"]
    single_summary = single_result["trainable_summary"]
    all_summary = all_result["trainable_summary"]
    # "all" -> every layer (4, in the tiny dry-run model) is trainable
    # instead of just one, so the trainable param count/pct should scale up
    # accordingly relative to the single-layer selection.
    assert all_summary["trainable_params"] == pytest.approx(4 * single_summary["trainable_params"])
    assert all_summary["trainable_pct"] > single_summary["trainable_pct"]


# ---------------------------------------------------------------------------
# build_sft_run_name
# ---------------------------------------------------------------------------


def test_build_sft_run_name_no_override_is_prefix() -> None:
    assert build_sft_run_name("sft-001", [9], layers_overridden=False) == "sft-001"


def test_build_sft_run_name_with_override_includes_layers() -> None:
    assert build_sft_run_name("sft-001", [6, 9], layers_overridden=True) == "sft-001-L6-9"


# ---------------------------------------------------------------------------
# config loading: merges over base.yaml (fixture config, Issue #31 -- real
# configs/sft/*.yaml experiment configs are Issue #32, not created here)
# ---------------------------------------------------------------------------


def test_sft_fixture_config_merges_over_base() -> None:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    sft_cfg = load_config(root / "tests" / "fixtures" / "sft_test.yaml")
    merged = merge_configs(base_cfg, sft_cfg)

    assert merged["model_name"] == "dummy/does-not-matter"
    assert merged["tuning"]["trainable_layer_indices"] == [9]
    assert merged["training"]["num_train_epochs"] == 1
    assert merged["training"]["learning_rate"] == pytest.approx(1.0e-4)
    # not overridden by the fixture config -> inherited from base.yaml
    assert merged["training"]["per_device_train_batch_size"] == 1
    assert merged["dataset"]["train_path"] == "tests/fixtures/sft_test_data.jsonl"
