"""Training smoke test."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lfm25_ja.train.layer_select import select_trainable_layers, trainable_param_summary
from lfm25_ja.train.smoke import run_smoke_test, run_smoke_training_loop


class _Block(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _InnerStack(nn.Module):
    def __init__(self, n_layers: int = 4, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim) for _ in range(n_layers)])


class _DummyHFLikeModel(nn.Module):
    """Mimics the `model.model.layers` layout used by HF causal LMs."""

    def __init__(self, n_layers: int = 4, dim: int = 4) -> None:
        super().__init__()
        self.model = _InnerStack(n_layers, dim)
        self.embed = nn.Embedding(10, dim)


def test_select_trainable_layers_only_selected_trainable() -> None:
    model = _DummyHFLikeModel(n_layers=4)
    select_trainable_layers(model, [1, 2])
    for i, layer in enumerate(model.model.layers):
        expected = i in (1, 2)
        for p in layer.parameters():
            assert p.requires_grad is expected
    for p in model.embed.parameters():
        assert p.requires_grad is False


def test_select_trainable_layers_out_of_range_raises() -> None:
    model = _DummyHFLikeModel(n_layers=4)
    with pytest.raises(ValueError):
        select_trainable_layers(model, [10])


def test_select_trainable_layers_negative_index_raises() -> None:
    model = _DummyHFLikeModel(n_layers=4)
    with pytest.raises(ValueError):
        select_trainable_layers(model, [-1])


def test_select_trainable_layers_fallback_to_top_level_layers() -> None:
    class _FlatModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    model = _FlatModel()
    select_trainable_layers(model, [0])
    assert model.layers[0].weight.requires_grad is True
    assert model.layers[1].weight.requires_grad is False


def test_select_trainable_layers_no_layers_raises() -> None:
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        select_trainable_layers(model, [0])


def test_trainable_param_summary() -> None:
    model = _DummyHFLikeModel(n_layers=4, dim=4)
    select_trainable_layers(model, [0])
    summary = trainable_param_summary(model)
    expected_trainable = sum(p.numel() for p in model.model.layers[0].parameters())
    total = sum(p.numel() for p in model.parameters())
    assert summary["trainable_params"] == expected_trainable
    assert summary["total_params"] == total
    assert summary["trainable_pct"] == pytest.approx(expected_trainable / total * 100.0)


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
    """Full bf16 load + layer-selective full-parameter FT smoke on CUDA (GPU + HF access)."""
    from lfm25_ja.train.smoke import run_smoke_test

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    result = run_smoke_test()
    assert result.max_steps >= 20
    assert result.final_loss < result.initial_loss
    assert result.peak_vram_bytes > 0
