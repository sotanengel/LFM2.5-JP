"""Training smoke test."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lfm25_ja.train.layer_select import build_lora_config, freeze_all_but_lora
from lfm25_ja.train.smoke import run_smoke_test, run_smoke_training_loop


def test_build_lora_config_layer_indices() -> None:
    cfg = {
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj"],
            "trainable_layer_indices": [14, 15],
        }
    }
    lora_cfg = build_lora_config(cfg)
    assert lora_cfg.r == 16
    assert lora_cfg.layers_to_transform == [14, 15]
    assert set(lora_cfg.target_modules) == {"q_proj", "v_proj"}


def test_freeze_all_but_lora() -> None:
    model = nn.Linear(4, 4)
    model.weight.requires_grad = True
    model.bias.requires_grad = True
    # Simulate LoRA param name
    lora_param = nn.Parameter(torch.randn(2, 2), requires_grad=True)
    model.register_parameter("lora_A", lora_param)
    freeze_all_but_lora(model)
    assert model.weight.requires_grad is False
    assert model.lora_A.requires_grad is True


def test_run_smoke_training_loop_loss_decreases() -> None:
    model = nn.Linear(8, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = run_smoke_training_loop(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_steps=20,
        batch_size=2,
        seq_len=8,
        vocab_size=16,
    )
    assert len(losses) == 20
    assert losses[-1] < losses[0]


def test_smoke_dry_run() -> None:
    result = run_smoke_test(dry_run=True)
    assert result.final_loss < result.initial_loss
    assert result.max_steps >= 3


@pytest.mark.gpu
def test_smoke_test_gpu_integration() -> None:
    """Full 4bit load + QLoRA smoke on CUDA (requires GPU + HF access)."""
    from lfm25_ja.train.smoke import run_smoke_test

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    result = run_smoke_test()
    assert result.max_steps >= 20
    assert result.final_loss < result.initial_loss
    assert result.peak_vram_bytes > 0
