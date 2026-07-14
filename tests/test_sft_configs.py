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


@pytest.mark.parametrize(
    ("filename", "expected_layers", "expected_lr"),
    [
        ("sft_004_L9_lr1e-5.yaml", [9], 1e-5),
        ("sft_004_L9_lr3e-5.yaml", [9], 3e-5),
        ("sft_004_L9_lr1e-4.yaml", [9], 1e-4),
        ("sft_004_L6-9_lr1e-5.yaml", [6, 7, 8, 9], 1e-5),
        ("sft_004_L6-9_lr3e-5.yaml", [6, 7, 8, 9], 3e-5),
        ("sft_004_L6-9_lr1e-4.yaml", [6, 7, 8, 9], 1e-4),
    ],
)
def test_sft_004_lr_epoch_sweep_config_merges_over_base(
    filename: str, expected_layers: list[int], expected_lr: float
) -> None:
    """sft-004 lr/epoch スイープ(Issue #36)。sft-003(#33 #35)で 5 アーム全てが
    pre-SFT base を下回った件を受け、学習強度過剰仮説を直接テストする。sft-003
    の layerft config(L9 / L6-9)と layers・データ・ベースモデルは同一のまま、
    learning_rate を 1e-5/3e-5/1e-4 で振り、num_train_epochs を 2→1 に半減させた
    6 アーム(2 layer configs x 3 lr)を検証する。
    """
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    sft_cfg = load_config(root / "configs" / "sft" / filename)
    merged = merge_configs(base_cfg, sft_cfg)

    assert merged["model_name"] == _JP_MODEL
    assert merged["tuning"]["method"] == "full_layer"
    assert merged["tuning"]["trainable_layer_indices"] == expected_layers
    assert merged["training"]["num_train_epochs"] == 1
    assert merged["training"]["learning_rate"] == pytest.approx(expected_lr)
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
