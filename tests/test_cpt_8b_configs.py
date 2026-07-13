"""configs/cpt/cpt_8b_a1b_layerft.yaml merge over base.yaml (Issue #95).

8B-A1B-Base CPT for local 8GB: NF4 4bit frozen weights + central-band
layer FT ([10..13] mirrors 1.2B [6..9] from PR #86 on L=24).
"""

from pathlib import Path

import pytest

from lfm25_ja.utils.config import load_config, load_project_config, merge_configs

_8B_BASE = "LiquidAI/LFM2.5-8B-A1B-Base"


def test_cpt_8b_a1b_layerft_config_merges_over_base() -> None:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    cpt_cfg = load_config(root / "configs" / "cpt" / "cpt_8b_a1b_layerft.yaml")
    merged = merge_configs(base_cfg, cpt_cfg)

    assert merged["model_name"] == _8B_BASE
    assert merged["max_seq_len"] == 1024
    assert merged["tuning"]["method"] == "full_layer"
    assert merged["tuning"]["trainable_layer_indices"] == [10, 11, 12, 13]
    assert merged["tuning"]["load_in_4bit"] is True
    assert merged["training"]["num_train_epochs"] == 1
    assert merged["training"]["learning_rate"] == pytest.approx(1.0e-4)
    assert merged["training"]["per_device_train_batch_size"] == 1
    assert merged["training"]["gradient_accumulation_steps"] == 4
    assert merged["training"]["gradient_checkpointing"] is True
    assert merged["training"]["optim"] == "paged_adamw_8bit"
    assert merged["dataset"]["train_path"] == "data/processed/mixture.jsonl"
    assert merged["dataset"]["packed_cache_dir"] == "data/processed/packed_8b_a1b"
    assert merged["logging"]["run_name_prefix"] == "cpt-8b-a1b-layerft"


def test_cpt_1_2b_layerft_does_not_enable_4bit_by_default() -> None:
    """1.2B path must stay bf16 (load_in_4bit absent or false) for back-compat."""
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    cpt_cfg = load_config(root / "configs" / "cpt" / "cpt_1.2b_layerft.yaml")
    merged = merge_configs(base_cfg, cpt_cfg)

    assert merged["tuning"].get("load_in_4bit", False) is False
    assert merged["model_name"] == "LiquidAI/LFM2.5-1.2B-Base"
