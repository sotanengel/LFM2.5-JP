"""350M pilot CPT training tests: sequence packing, dataset building, dry-run
training loop, and config merging (Issue #23 / #24 / #25).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lfm25_ja.data.clean import _write_jsonl
from lfm25_ja.train.train_cpt import build_cpt_dataset, pack_sequences, run_cpt
from lfm25_ja.utils.config import load_config, load_project_config, merge_configs

# ---------------------------------------------------------------------------
# pack_sequences (Issue #23)
# ---------------------------------------------------------------------------


def test_pack_sequences_concatenates_with_eos_and_chunks() -> None:
    token_ids_iter = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    eos = 0
    # concatenated stream (eos appended after every doc):
    # 1 2 3 0 4 5 0 6 7 8 9 0  -> length 12
    packed = pack_sequences(token_ids_iter, seq_len=4, eos_token_id=eos)

    assert len(packed) == 3
    assert packed[0]["input_ids"] == [1, 2, 3, 0]
    assert packed[1]["input_ids"] == [4, 5, 0, 6]
    assert packed[2]["input_ids"] == [7, 8, 9, 0]


def test_pack_sequences_labels_equal_input_ids() -> None:
    packed = pack_sequences([[1, 2], [3, 4]], seq_len=3, eos_token_id=9)
    for row in packed:
        assert row["labels"] == row["input_ids"]
        assert row["attention_mask"] == [1] * len(row["input_ids"])


def test_pack_sequences_discards_remainder() -> None:
    # stream: 1 2 3 0 (eos) -> length 4, seq_len=3 -> 1 full chunk, remainder of 1 discarded
    packed = pack_sequences([[1, 2, 3]], seq_len=3, eos_token_id=0)
    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3]


def test_pack_sequences_empty_input_yields_empty_list() -> None:
    assert pack_sequences([], seq_len=4, eos_token_id=0) == []


def test_pack_sequences_invalid_seq_len_raises() -> None:
    with pytest.raises(ValueError):
        pack_sequences([[1, 2]], seq_len=0, eos_token_id=0)


# ---------------------------------------------------------------------------
# build_cpt_dataset (Issue #23)
# ---------------------------------------------------------------------------


class MockCPTTokenizer:
    """Deterministic whitespace tokenizer with an eos token, for tests (no HF download)."""

    _WORD_RE = re.compile(r"\S+")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<eos>": 0}
        self.inverse: dict[int, str] = {0: "<eos>"}
        self.eos_token_id = 0

    def _token_id(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse[idx] = token
        return self.vocab[token]

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        tokens = self._WORD_RE.findall(text)
        return {"input_ids": [self._token_id(t) for t in tokens]}


def test_build_cpt_dataset_reads_jsonl_and_packs(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "mixture.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"text": "one two three four"},
            {"text": "five six seven"},
        ],
    )
    tokenizer = MockCPTTokenizer()
    dataset = build_cpt_dataset(str(jsonl_path), tokenizer, seq_len=4)

    assert len(dataset) >= 1
    for row in dataset:
        assert len(row["input_ids"]) == 4
        assert row["labels"] == row["input_ids"]


def test_build_cpt_dataset_missing_eos_token_id_raises(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "mixture.jsonl"
    _write_jsonl(jsonl_path, [{"text": "hello world"}])

    class _NoEosTokenizer:
        eos_token_id = None

        def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
            return {"input_ids": [1, 2]}

    with pytest.raises(ValueError):
        build_cpt_dataset(str(jsonl_path), _NoEosTokenizer(), seq_len=4)


# ---------------------------------------------------------------------------
# run_cpt(dry_run=True) (Issue #23)
# ---------------------------------------------------------------------------


def test_run_cpt_dry_run_loss_decreases_and_summary_consistent(tmp_path: Path) -> None:
    cpt_config = tmp_path / "cpt_test.yaml"
    cpt_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: [1]
""",
        encoding="utf-8",
    )
    result = run_cpt(str(cpt_config), dry_run=True)

    assert result["final_loss"] < result["initial_loss"]
    assert len(result["losses"]) >= 2
    summary = result["trainable_summary"]
    assert 0 < summary["trainable_params"] < summary["total_params"]
    assert summary["trainable_pct"] == pytest.approx(
        summary["trainable_params"] / summary["total_params"] * 100.0
    )


# ---------------------------------------------------------------------------
# configs/cpt/cpt_350m_pilot.yaml merges over base.yaml (Issue #25)
# ---------------------------------------------------------------------------


def test_cpt_350m_pilot_config_merges_over_base() -> None:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    cpt_cfg = load_config(root / "configs" / "cpt" / "cpt_350m_pilot.yaml")
    merged = merge_configs(base_cfg, cpt_cfg)

    assert merged["model_name"] == "LiquidAI/LFM2-350M"
    assert merged["tuning"]["trainable_layer_indices"] == [7]
    assert merged["training"]["num_train_epochs"] == 1
    assert merged["training"]["learning_rate"] == pytest.approx(1.0e-4)
    # not overridden by the pilot config -> inherited from base.yaml
    assert merged["training"]["per_device_train_batch_size"] == 1
    assert merged["dataset"]["train_path"] == "data/processed/mixture.jsonl"


def test_cpt_1_2b_layerft_config_merges_over_base() -> None:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    cpt_cfg = load_config(root / "configs" / "cpt" / "cpt_1.2b_layerft.yaml")
    merged = merge_configs(base_cfg, cpt_cfg)

    assert merged["model_name"] == "LiquidAI/LFM2.5-1.2B-Base"
    assert "trainable_layer_indices" in merged["tuning"]
