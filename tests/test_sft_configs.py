"""configs/sft/sft_1.2b_layerft_*.yaml merge over base.yaml (Issue #32).

These are the sft-003 layer-ablation configs (Phase 2 layer-profiling center
band: see experiments/reports/phase2_gate_and_next_steps.md sec 4.1). Each
config must resolve tuning.trainable_layer_indices to the exact list
train_sft.py expects and must target the confirmed Phase 3 starting
checkpoint (LiquidAI/LFM2.5-1.2B-JP-202606), not -Base or -Instruct.
"""

from pathlib import Path

import pytest

from lfm25_ja.utils.config import load_config, load_project_config, merge_configs

_JP_MODEL = "LiquidAI/LFM2.5-1.2B-JP-202606"


@pytest.mark.parametrize(
    ("filename", "expected_layers"),
    [
        ("sft_1.2b_layerft_L9.yaml", [9]),
        ("sft_1.2b_layerft_L6.yaml", [6]),
        ("sft_1.2b_layerft_L6L9.yaml", [6, 9]),
        ("sft_1.2b_layerft_L6-9.yaml", [6, 7, 8, 9]),
    ],
)
def test_sft_layerft_config_merges_over_base(filename: str, expected_layers: list[int]) -> None:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    sft_cfg = load_config(root / "configs" / "sft" / filename)
    merged = merge_configs(base_cfg, sft_cfg)

    assert merged["model_name"] == _JP_MODEL
    assert merged["tuning"]["method"] == "full_layer"
    assert merged["tuning"]["trainable_layer_indices"] == expected_layers
    # sft-001 と同条件(2 epoch / ichikara)で層 ablation を成立させるため、
    # 4 layerft config も 2 epoch / ichikara に統一(sft-003 実行時、#35)。
    assert merged["training"]["num_train_epochs"] == 2
    assert merged["dataset"]["train_path"] == "data/processed/sft/ichikara.jsonl"


def test_sft_layerft_full_config_merges_over_base() -> None:
    """フル FT 参照アーム(sft-003 の上限参照)。全 16 層を trainable_layer_indices:
    "all" で指定し、train_sft.py:45 の resolve_trainable_layer_indices が
    モデルの実際の層数に合わせて展開する(単一 config で 350M / 1.2B のどちらでも
    可搬)。データ・エポックは他アームと同一。"""
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    sft_cfg = load_config(root / "configs" / "sft" / "sft_1.2b_layerft_full.yaml")
    merged = merge_configs(base_cfg, sft_cfg)

    assert merged["model_name"] == _JP_MODEL
    assert merged["tuning"]["method"] == "full_layer"
    assert merged["tuning"]["trainable_layer_indices"] == "all"
    assert merged["training"]["num_train_epochs"] == 2
    assert merged["dataset"]["train_path"] == "data/processed/sft/ichikara.jsonl"


def test_sft_001_ichikara_config_merges_over_base() -> None:
    """sft-001 (Issue #33): ichikara のみ・単層 L9・2 epoch。sft-003(#35)の
    「単層 L9」アームを兼ねるため、layers/model は sft_1.2b_layerft_L9.yaml と
    同じだが epoch 数とデータセットが異なる(2 epoch / ichikara 単体)。
    """
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    sft_cfg = load_config(root / "configs" / "sft" / "sft_001_ichikara.yaml")
    merged = merge_configs(base_cfg, sft_cfg)

    assert merged["model_name"] == _JP_MODEL
    assert merged["tuning"]["method"] == "full_layer"
    assert merged["tuning"]["trainable_layer_indices"] == [9]
    assert merged["training"]["num_train_epochs"] == 2
    assert merged["dataset"]["train_path"] == "data/processed/sft/ichikara.jsonl"
