"""Layer-selective LoRA configuration for LFM2.5."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from peft import LoraConfig


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    """Build PEFT LoRA config with optional layer index restriction."""
    lora = cfg["lora"]
    layers = lora.get("trainable_layer_indices")
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.05),
        bias=lora.get("bias", "none"),
        target_modules=lora["target_modules"],
        layers_to_transform=layers if layers else None,
        task_type="CAUSAL_LM",
    )


def freeze_all_but_lora(model: nn.Module) -> None:
    """Freeze base weights; keep only LoRA adapter parameters trainable."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
