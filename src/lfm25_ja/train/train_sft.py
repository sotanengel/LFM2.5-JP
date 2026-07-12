"""Supervised fine-tuning (SFT) via `trl.SFTTrainer`, config-driven (Issue #31).

Trains via the same layer-selective full-parameter fine-tuning as CPT (see
``lfm25_ja.train.layer_select``): the model is loaded in bf16, every
parameter is frozen, and only ``tuning.trainable_layer_indices`` is
unfrozen -- no LoRA/QLoRA adapters are used. Training data is a JSONL of
chat ``messages`` that gets tokenized and loss-masked (assistant-completion
tokens only) via ``lfm25_ja.data.format_chat.build_sft_example``.

The resulting dataset already carries ``input_ids``/``labels``/
``attention_mask`` columns, so ``trl.SFTTrainer`` detects it as
already-processed and skips its own chat-template rendering / tokenization
step, using its ``DataCollatorForLanguageModeling`` purely to pad batches
while respecting the existing ``-100`` loss mask (verified against the
project's pinned ``trl`` release; see PR description for the version-
specific behavior this relies on).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lfm25_ja.data.clean import _read_jsonl
from lfm25_ja.data.format_chat import build_sft_example
from lfm25_ja.train.callbacks import LossTrackerCallback, VramMonitorCallback
from lfm25_ja.train.layer_select import (
    _resolve_layers,
    select_trainable_layers,
    trainable_param_summary,
)
from lfm25_ja.train.train_cpt import parse_layer_indices, resolve_resume_checkpoint
from lfm25_ja.utils.config import load_config, load_project_config, merge_configs
from lfm25_ja.utils.memory import get_vram_usage, reset_peak_memory
from lfm25_ja.utils.seed import set_seed

logger = logging.getLogger(__name__)


def resolve_trainable_layer_indices(raw: list[int] | str, model: nn.Module) -> list[int]:
    """Resolve ``tuning.trainable_layer_indices`` into a concrete 0-based index list.

    Supports the literal string ``"all"`` (full-fine-tune reference config)
    in addition to an explicit list of indices. ``"all"`` is resolved
    against ``model``'s actual transformer layer count (via the same layer
    list ``select_trainable_layers`` freezes), so it works regardless of how
    many layers the underlying model has.
    """
    if raw == "all":
        return list(range(len(_resolve_layers(model))))
    return list(raw)


def build_sft_dataset(jsonl_path: str | Path, tokenizer: Any, max_seq_len: int) -> Any:
    """Read a JSONL of ``{"messages": [...]}`` rows and build a pre-tokenized,
    loss-masked SFT dataset via :func:`lfm25_ja.data.format_chat.build_sft_example`.

    Returns a ``datasets.Dataset`` with ``input_ids``/``labels``/
    ``attention_mask`` columns. ``trl.SFTTrainer`` detects the ``input_ids``
    column and treats the dataset as already-processed, so the loss mask
    built here (assistant-completion tokens only) is used as-is.
    """
    from datasets import Dataset

    docs = _read_jsonl(jsonl_path)
    if not docs:
        raise ValueError(f"No examples found in {jsonl_path!r}")
    rows = [build_sft_example(doc["messages"], tokenizer, max_seq_len) for doc in docs]
    return Dataset.from_list(rows)


class _TinySFTStack(nn.Module):
    """Stand-in for `model.model.layers` used to exercise select_trainable_layers."""

    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TinySFTModel(nn.Module):
    """Small token-embedding + multi-layer + LM-head model mimicking the HF
    causal LM layout (`model.model.layers`) for the CPU dry_run path.

    Used only so dry_run can exercise the same freeze -> select layers ->
    masked cross-entropy loss pipeline as the real bf16 model + build_sft_example
    loss mask, without downloading any weights.
    """

    def __init__(self, vocab_size: int = 32, dim: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.model = _TinySFTStack(dim, n_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.model(x)
        return self.lm_head(x)


def _run_dry_run_sft(cfg: dict[str, Any], max_steps: int = 20) -> dict[str, Any]:
    """CPU-only dry run: synthetic token data + a masked cross-entropy loss
    (mirroring build_sft_example's -100 loss mask contract) through a tiny
    multi-layer model, exercising freeze -> select layers -> loss decreases,
    without any HF download.
    """
    tuning = cfg.get("tuning", {})
    raw_indices = tuning.get("trainable_layer_indices", [1]) or [1]
    n_layers = 4

    model = _TinySFTModel(vocab_size=32, dim=8, n_layers=n_layers)
    if raw_indices == "all":
        layer_indices = list(range(n_layers))
    else:
        # Map configured (possibly out-of-range for this tiny model) indices
        # into a valid range so the dry run stays fast while still
        # exercising the real select_trainable_layers() call path used
        # against the full model.
        layer_indices = sorted({idx % n_layers for idx in raw_indices})
    select_trainable_layers(model, layer_indices)
    trainable_summary = trainable_param_summary(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.1)

    vocab_size = 32
    seq_len = 8
    batch_size = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    # Mask out the first half of each sequence (mimics prompt/system/user
    # tokens not contributing to loss), leaving only the "completion" half
    # learnable -- exercises the same -100 loss-masking contract as
    # build_sft_example.
    labels[:, : seq_len // 2] = -100

    model.train()
    losses: list[float] = []
    for _ in range(max_steps):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "trainable_summary": trainable_summary,
    }


def _load_sft_config(config_path: str) -> dict[str, Any]:
    """Deep-merge an sft config (e.g. configs/sft/sft-001.yaml) over base.yaml."""
    base_cfg = load_project_config("base.yaml")
    sft_cfg = load_config(config_path)
    return merge_configs(base_cfg, sft_cfg)


def build_sft_run_name(prefix: str, layer_indices: list[int], layers_overridden: bool) -> str:
    """Compose the SFT run name / output subdirectory.

    Mirrors :func:`lfm25_ja.train.train_cpt.build_run_name` but without a
    data-package axis (SFT has no packed-cache full/centi/deci split):
    plain ``prefix`` unless ``--layers`` overrides the config, in which case
    ``{prefix}-L{indices joined by '-'}`` is used so overridden runs don't
    collide with the config's default run directory.
    """
    if not layers_overridden:
        return prefix
    layer_suffix = "-".join(str(idx) for idx in layer_indices)
    return f"{prefix}-L{layer_suffix}"


def run_sft(
    config_path: str,
    dry_run: bool = False,
    layers: list[int] | None = None,
    no_checkpoints: bool = False,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Run TRL-SFTTrainer-based supervised fine-tuning driven by ``config_path``.

    ``config_path`` is deep-merged over ``configs/base.yaml`` (sft config
    wins). With ``dry_run=True``, no HF download happens: a tiny CPU model is
    trained on synthetic token data (with a manual -100 loss mask, mirroring
    build_sft_example) to exercise the layer-select + training loop,
    returning ``{"initial_loss", "final_loss", "losses",
    "trainable_summary"}``.

    ``layers``, when given, overrides ``tuning.trainable_layer_indices`` from
    the config (see :func:`lfm25_ja.train.train_cpt.parse_layer_indices` for
    the CLI form). It also changes the run name (see
    :func:`build_sft_run_name`).

    ``no_checkpoints`` disables intermediate checkpoint saving
    (``save_strategy="no"``); only the final model is written via
    ``trainer.save_model``.

    ``output_root``, when given, overrides ``output_dir`` from the config as
    the root directory the run name is written under (default ``outputs``).
    """
    cfg = _load_sft_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    if dry_run:
        return _run_dry_run_sft(cfg)

    # Imported lazily so dry_run (and CPU-only test/CI environments) never
    # need a working HF download / GPU stack.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tuning = cfg.get("tuning", {})
    layers_overridden = layers is not None
    raw_indices = list(layers) if layers_overridden else tuning.get("trainable_layer_indices", [])
    layer_indices = resolve_trainable_layer_indices(raw_indices, model)
    select_trainable_layers(model, layer_indices)
    summary = trainable_param_summary(model)
    logger.info("Trainable param summary: %s", summary)

    dataset_cfg = cfg.get("dataset", {})
    if "train_path" not in dataset_cfg:
        raise ValueError("sft config must set dataset.train_path (see configs/sft/*.yaml)")
    max_seq_len = int(cfg.get("max_seq_len", 1024))
    dataset = build_sft_dataset(dataset_cfg["train_path"], tokenizer, max_seq_len)

    sample_fraction = dataset_cfg.get("sample_fraction")
    if sample_fraction is not None:
        n = max(1, int(len(dataset) * float(sample_fraction)))
        dataset = dataset.select(range(n))

    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    run_name_prefix = logging_cfg.get("run_name_prefix", "sft")
    run_name = build_sft_run_name(run_name_prefix, layer_indices, layers_overridden)
    output_root_dir = (
        Path(output_root) if output_root is not None else Path(cfg.get("output_dir", "outputs"))
    )
    output_dir = str(output_root_dir / run_name)

    vram_cb = VramMonitorCallback()
    loss_cb = LossTrackerCallback()

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        save_steps=int(training_cfg.get("save_steps", 100)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        save_strategy="no" if no_checkpoints else "steps",
        report_to=[],
        fp16=False,
        bf16=cfg.get("precision") == "bf16",
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        optim=str(training_cfg.get("optim", "paged_adamw_8bit")),
        remove_unused_columns=False,
        max_length=max_seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[vram_cb, loss_cb],
    )
    reset_peak_memory()
    # Resume from the latest checkpoint in output_dir when one exists (e.g. an
    # interrupted run being restarted); otherwise start fresh (see
    # resolve_resume_checkpoint for why the path is resolved explicitly
    # rather than passing resume_from_checkpoint=True).
    resume_checkpoint = resolve_resume_checkpoint(output_dir, no_checkpoints)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)

    losses = loss_cb.losses or [0.0, 0.0]
    peak = max(get_vram_usage()["max_allocated"], vram_cb.peak_bytes)
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "trainable_summary": summary,
        "peak_vram_bytes": peak,
        "output_dir": output_dir,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="LFM2.5-JA SFT training via trl.SFTTrainer (Issue #31)"
    )
    parser.add_argument("--config", required=True, help="Path to configs/sft/*.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="CPU-only dry run, no HF download / GPU required"
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated 0-based layer indices, e.g. '7,8'. Overrides "
        "tuning.trainable_layer_indices from the config and changes the run name.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable intermediate checkpoint saving (save_strategy=no); only the "
        "final model is written via trainer.save_model",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory the run name is written under (default: outputs, "
        "or config output_dir). Use e.g. 'outputs/sweep' for sweep runs.",
    )
    args = parser.parse_args()

    layers = parse_layer_indices(args.layers) if args.layers is not None else None

    result = run_sft(
        args.config,
        dry_run=args.dry_run,
        layers=layers,
        no_checkpoints=args.no_checkpoints,
        output_root=args.output_root,
    )
    print(
        f"SFT run finished: loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f} "
        f"(trainable={result['trainable_summary']['trainable_pct']:.3f}%)"
    )
    if result["final_loss"] >= result["initial_loss"]:
        raise SystemExit("SFT run failed: loss did not decrease")


if __name__ == "__main__":
    main()
