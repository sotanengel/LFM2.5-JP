"""Phase 0 smoke test: 4bit inference + layer-selective QLoRA training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from lfm25_ja.train.callbacks import LossTrackerCallback, VramMonitorCallback
from lfm25_ja.train.layer_select import build_lora_config, freeze_all_but_lora
from lfm25_ja.utils.config import load_project_config
from lfm25_ja.utils.memory import get_vram_usage, reset_peak_memory
from lfm25_ja.utils.seed import set_seed


@dataclass
class SmokeTrainResult:
    initial_loss: float
    final_loss: float
    losses: list[float]
    max_steps: int
    peak_vram_bytes: int


def _build_bnb_config(cfg: dict[str, Any]) -> BitsAndBytesConfig:
    qlora = cfg["qlora"]
    compute_dtype = getattr(torch, qlora.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=qlora.get("load_in_4bit", True),
        bnb_4bit_quant_type=qlora.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora.get("bnb_4bit_use_double_quant", True),
    )


def _load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any]:
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_build_bnb_config(cfg),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    freeze_all_but_lora(model)
    return model, tokenizer


def _build_dummy_dataset(
    tokenizer: Any,
    num_samples: int,
    seq_len: int,
) -> list[dict[str, list[int]]]:
    texts = [
        "こんにちは、LFM2.5のスモークテストです。",
        "Layer-selective QLoRA training on RTX 3060 Ti.",
    ]
    rows: list[dict[str, list[int]]] = []
    for i in range(num_samples):
        text = texts[i % len(texts)]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors=None,
        )
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["input_ids"].copy(),
            }
        )
    return rows


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {k: torch.tensor(v) for k, v in row.items()}


def run_smoke_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int = 128,
) -> list[float]:
    """Minimal training loop for unit tests (no HF model required)."""
    model.train()
    losses: list[float] = []
    if isinstance(model, nn.Linear):
        x = torch.randn(batch_size, seq_len, device=device)
        target = torch.randn(batch_size, model.out_features, device=device)
        for _ in range(max_steps):
            y = model(x)
            loss = nn.functional.mse_loss(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    for _ in range(max_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        if y.dim() != 2:
            loss = y.mean()
        else:
            labels = torch.randint(0, y.shape[-1], (batch_size,), device=device)
            loss = nn.functional.cross_entropy(y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def run_smoke_test(
    config_path: str | None = None,
    dry_run: bool = False,
) -> SmokeTrainResult:
    """Run full smoke test with optional dry_run for mocked environments."""
    cfg = load_project_config("base.yaml")
    if config_path:
        from lfm25_ja.utils.config import load_config, merge_configs

        cfg = merge_configs(cfg, load_config(config_path))

    smoke = cfg.get("smoke_test", {})
    max_steps = int(smoke.get("max_steps", 20))
    seq_len = int(smoke.get("max_seq_len", 512))
    set_seed(int(cfg.get("seed", 42)))
    reset_peak_memory()

    if dry_run:
        model = nn.Linear(8, 8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = run_smoke_training_loop(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_steps=max_steps,
            batch_size=2,
            seq_len=8,
        )
        return SmokeTrainResult(
            initial_loss=losses[0],
            final_loss=losses[-1],
            losses=losses,
            max_steps=max_steps,
            peak_vram_bytes=0,
        )

    # Inference sanity check
    model, tokenizer = _load_model_and_tokenizer(cfg)
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    dataset = _ListDataset(_build_dummy_dataset(tokenizer, num_samples=8, seq_len=seq_len))
    train_cfg = cfg.get("training", {})
    vram_cb = VramMonitorCallback()
    loss_cb = LossTrackerCallback()

    args = TrainingArguments(
        output_dir=str(cfg.get("output_dir", "outputs")) + "/smoke",
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=cfg.get("precision") == "bf16",
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        callbacks=[vram_cb, loss_cb],
    )
    trainer.train()
    losses = loss_cb.losses or [0.0, 0.0]
    peak = max(get_vram_usage()["max_allocated"], vram_cb.peak_bytes)
    return SmokeTrainResult(
        initial_loss=losses[0],
        final_loss=losses[-1],
        losses=losses,
        max_steps=max_steps,
        peak_vram_bytes=peak,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2.5-JA smoke test")
    parser.add_argument("--config", default=None, help="Optional config override YAML")
    args = parser.parse_args()
    result = run_smoke_test(config_path=args.config)
    print(
        f"Smoke test OK: loss {result.initial_loss:.4f} -> {result.final_loss:.4f}, "
        f"peak_vram={result.peak_vram_bytes} bytes, steps={result.max_steps}"
    )
    if result.final_loss >= result.initial_loss:
        raise SystemExit("Smoke test failed: loss did not decrease")


if __name__ == "__main__":
    main()
