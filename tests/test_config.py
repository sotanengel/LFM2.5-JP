"""Config loader tests."""

from pathlib import Path

import pytest

from lfm25_ja.utils.config import config_hash, load_config, merge_configs


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
model_name: test-model
seed: 42
precision: bf16
max_seq_len: 1024
tuning:
  method: full_layer
  trainable_layer_indices: [15]
""",
        encoding="utf-8",
    )
    override = tmp_path / "override.yaml"
    override.write_text("max_seq_len: 2048\n", encoding="utf-8")
    return tmp_path


def test_load_config(config_dir: Path) -> None:
    cfg = load_config(config_dir / "base.yaml")
    assert cfg["model_name"] == "test-model"
    assert cfg["seed"] == 42
    assert cfg["tuning"]["trainable_layer_indices"] == [15]


def test_merge_configs(config_dir: Path) -> None:
    base = load_config(config_dir / "base.yaml")
    merged = merge_configs(base, load_config(config_dir / "override.yaml"))
    assert merged["max_seq_len"] == 2048
    assert merged["model_name"] == "test-model"


def test_config_hash_stable(config_dir: Path) -> None:
    cfg = load_config(config_dir / "base.yaml")
    assert config_hash(cfg) == config_hash(load_config(config_dir / "base.yaml"))


def test_project_base_yaml_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "base.yaml")
    assert "LiquidAI" in cfg["model_name"]
    assert "trainable_layer_indices" in cfg["tuning"]
    assert "lora" not in cfg
    assert "qlora" not in cfg
