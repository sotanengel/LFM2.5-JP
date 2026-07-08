"""Layer-selective full-parameter fine-tuning (layer FT) utilities for LFM2.5.

Freezes the whole model and unfreezes only a config-selected subset of
`model.model.layers` (see arXiv:2607.01232). No LoRA/PEFT adapters are used;
the selected layers are trained with their full parameters.
"""

from __future__ import annotations

import torch.nn as nn


def _resolve_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer layer list on `model`.

    Tries `model.model.layers` first (standard HF causal LM layout), then
    falls back to `model.layers`. Raises ValueError if neither is present.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError(
            "Could not locate a transformer layer list on model: "
            "tried `model.model.layers` and `model.layers`."
        )
    return layers


def select_trainable_layers(model: nn.Module, layer_indices: list[int]) -> None:
    """Freeze all parameters, then unfreeze only the selected transformer layers.

    Args:
        model: A model exposing `model.model.layers` (or `model.layers` as a
            fallback) -- an indexable sequence of layer submodules.
        layer_indices: 0-based indices of layers to keep trainable. All other
            parameters (including layers not listed here) stay frozen.

    Raises:
        ValueError: if the model has no discoverable layer list, or if any
            index in `layer_indices` is out of range for that layer list.
    """
    layers = _resolve_layers(model)
    n_layers = len(layers)

    for idx in layer_indices:
        if idx < 0 or idx >= n_layers:
            raise ValueError(
                f"trainable_layer_indices contains {idx}, which is out of range "
                f"for a model with {n_layers} layers (valid range: 0..{n_layers - 1})."
            )

    for param in model.parameters():
        param.requires_grad = False

    for idx in layer_indices:
        for param in layers[idx].parameters():
            param.requires_grad = True


def trainable_param_summary(model: nn.Module) -> dict:
    """Return a summary of trainable vs. total parameter counts.

    Returns:
        A dict with keys `trainable_params` (int), `total_params` (int), and
        `trainable_pct` (float, 0-100).
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / total_params * 100.0) if total_params else 0.0
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
    }
