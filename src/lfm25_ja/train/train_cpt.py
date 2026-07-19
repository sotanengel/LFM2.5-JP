"""Packed causal-LM continued pre-training (CPT), config-driven (Issue #23 / #24).

Trains via layer-selective full-parameter fine-tuning (see
``lfm25_ja.train.layer_select``): the model is loaded in bf16, every
parameter is frozen, and only ``tuning.trainable_layer_indices`` is
unfrozen. Training data is a prepare.py-produced JSONL (``text`` field)
that gets tokenized and packed into fixed-length causal-LM sequences.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn

from lfm25_ja.data.clean import _iter_jsonl
from lfm25_ja.train.callbacks import LossTrackerCallback, VramMonitorCallback
from lfm25_ja.train.layer_select import select_trainable_layers, trainable_param_summary
from lfm25_ja.train.packed_cache import PACKAGES, apply_package, build_or_load_packed
from lfm25_ja.utils.config import load_config, load_project_config, merge_configs
from lfm25_ja.utils.memory import get_vram_usage, reset_peak_memory
from lfm25_ja.utils.seed import set_seed

logger = logging.getLogger(__name__)


def iter_packed_sequences(
    token_ids_iter: Iterable[list[int]], seq_len: int, eos_token_id: int
) -> Iterator[dict[str, list[int]]]:
    """Stream-pack tokenized documents into fixed-length causal-LM chunks.

    Documents are concatenated with ``eos_token_id`` separators. Full
    ``seq_len`` chunks are yielded as soon as the rolling buffer fills; only a
    ``< seq_len`` remainder is retained (Issue #136 / #132). Trailing tokens
    that do not fill a chunk are discarded.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    buffer: list[int] = []
    for token_ids in token_ids_iter:
        buffer.extend(token_ids)
        buffer.append(eos_token_id)
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            del buffer[:seq_len]
            yield {
                "input_ids": chunk,
                "labels": list(chunk),
                "attention_mask": [1] * seq_len,
            }


def pack_sequences(
    token_ids_iter: Iterable[list[int]], seq_len: int, eos_token_id: int
) -> list[dict[str, list[int]]]:
    """Concatenate tokenized documents (separated by ``eos_token_id``) and cut
    the resulting stream into fixed-length ``seq_len`` chunks for causal LM
    training.

    Any trailing tokens that don't fill a full ``seq_len`` chunk are
    discarded. Each returned row has ``labels`` equal to ``input_ids`` (causal
    LM: every position predicts the next token) and an all-ones
    ``attention_mask``.

    Implemented via :func:`iter_packed_sequences` so peak buffer stays near
    ``seq_len`` even for large corpora (Issue #136).
    """
    return list(iter_packed_sequences(token_ids_iter, seq_len, eos_token_id))


def build_cpt_dataset(
    jsonl_path: str | Path, tokenizer: Any, seq_len: int
) -> list[dict[str, list[int]]]:
    """Stream a prepare.py-produced JSONL (``text`` field per line), tokenize
    each document, and pack them into ``seq_len``-length causal LM examples.

    Uses line-at-a-time JSONL iteration (not a full in-memory load) so cpt-D
    scale mixtures (~173k docs) can be packed under WSL memory limits
    (Issue #136 / #132).
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id must be set to pack sequences")

    def _token_ids() -> Iterator[list[int]]:
        for doc in _iter_jsonl(jsonl_path):
            yield list(tokenizer(doc["text"])["input_ids"])

    return pack_sequences(_token_ids(), seq_len, eos_token_id)


class _PackedDataset(torch.utils.data.Dataset):
    """torch Dataset wrapper around a list of packed rows (see pack_sequences)."""

    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {k: torch.tensor(v) for k, v in row.items()}


class _TinyCPTStack(nn.Module):
    """Stand-in for `model.model.layers` used to exercise select_trainable_layers."""

    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyCPTModel(nn.Module):
    """Small token-embedding + multi-layer model mimicking the HF causal LM
    layout (`model.model.layers`) for the CPU dry_run path.

    Used only so dry_run can exercise the same synthetic-token ->
    freeze -> select layers -> train pipeline as the real bf16 model,
    without downloading any weights.
    """

    def __init__(self, vocab_size: int = 32, dim: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.model = _TinyCPTStack(dim, n_layers)
        self.head = nn.Linear(dim, dim, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids).mean(dim=1)
        x = self.model(x)
        return self.head(x)


def _run_dry_run_cpt(cfg: dict[str, Any], max_steps: int = 8) -> dict[str, Any]:
    """CPU-only dry run: synthetic token data through a tiny multi-layer model,
    exercising freeze -> select layers -> loss decreases, without any HF
    download.
    """
    tuning = cfg.get("tuning", {})
    configured_indices = list(tuning.get("trainable_layer_indices", [1])) or [1]
    n_layers = 4
    # Map configured (possibly out-of-range for this tiny model) indices into
    # a valid range so the dry run stays fast while still exercising the real
    # select_trainable_layers() call path used against the full model.
    layer_indices = sorted({idx % n_layers for idx in configured_indices})

    vocab_size = 32
    dim = 8
    model = _TinyCPTModel(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
    select_trainable_layers(model, layer_indices)
    trainable_summary = trainable_param_summary(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.1)

    seq_len = 8
    batch_size = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target = torch.randn(batch_size, dim)

    model.train()
    losses: list[float] = []
    for _ in range(max_steps):
        y = model(input_ids)
        loss = nn.functional.mse_loss(y, target)
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


def _load_cpt_config(config_path: str) -> dict[str, Any]:
    """Deep-merge a cpt config (e.g. configs/cpt/cpt_350m_pilot.yaml) over base.yaml."""
    base_cfg = load_project_config("base.yaml")
    cpt_cfg = load_config(config_path)
    return merge_configs(base_cfg, cpt_cfg)


def parse_layer_indices(raw: str) -> list[int]:
    """Parse a ``--layers`` CLI value like ``"7,8"`` into 0-based layer indices.

    Raises:
        ValueError: if ``raw`` is empty, contains a non-integer entry, or a
            negative index.
    """
    parts = [p.strip() for p in raw.split(",")]
    if not raw.strip() or any(not p for p in parts):
        raise ValueError(f"--layers must be a comma-separated list of integers, got {raw!r}")
    try:
        indices = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"--layers must be a comma-separated list of integers, got {raw!r}"
        ) from exc
    if any(idx < 0 for idx in indices):
        raise ValueError(f"--layers indices must be non-negative, got {raw!r}")
    return indices


def build_run_name(
    prefix: str,
    package: str,
    layer_indices: list[int],
    layers_overridden: bool,
) -> str:
    """Compose the training run name / output subdirectory.

    ``{prefix}-{package}-L{indices joined by '-'}`` in the general case, e.g.
    ``cpt-1.2b-layerft-centi-L7-8``. When ``package == "full"`` and no
    ``--layers`` override was given, the plain ``{prefix}`` is returned
    unchanged so existing ``outputs/cpt-1.2b-layerft`` runs keep working.
    """
    if package == "full" and not layers_overridden:
        return prefix
    layer_suffix = "-".join(str(idx) for idx in layer_indices)
    return f"{prefix}-{package}-L{layer_suffix}"


def resolve_resume_checkpoint(output_dir: str | Path, no_checkpoints: bool) -> str | None:
    """Return the checkpoint path :meth:`Trainer.train` should resume from.

    Returns ``None`` (start fresh) when ``no_checkpoints`` is set (no
    intermediate checkpoints are ever written in that mode), when
    ``output_dir`` doesn't exist yet (a brand new run), or when
    ``output_dir`` exists but contains no ``checkpoint-*`` subdirectory yet.
    Otherwise returns the latest checkpoint directory under ``output_dir``
    (via ``transformers.trainer_utils.get_last_checkpoint``).

    Passing ``resume_from_checkpoint=True`` to ``Trainer.train`` unconditionally
    would raise when no checkpoint has ever been written -- always true for a
    brand new ``output_dir`` -- so callers resolve the path explicitly first.
    """
    if no_checkpoints or not Path(output_dir).is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(output_dir))


def run_cpt(
    config_path: str,
    dry_run: bool = False,
    package: str = "full",
    rebuild_cache: bool = False,
    layers: list[int] | None = None,
    no_checkpoints: bool = False,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Run packed causal-LM CPT training driven by ``config_path``.

    ``config_path`` is deep-merged over ``configs/base.yaml`` (cpt config
    wins). With ``dry_run=True``, no HF download happens: a tiny CPU model is
    trained on synthetic token data to exercise the layer-select + training
    loop, returning ``{"initial_loss", "final_loss", "losses",
    "trainable_summary"}``.

    ``layers``, when given, overrides ``tuning.trainable_layer_indices`` from
    the config (see :func:`parse_layer_indices` for the CLI form). It also
    changes the run name (see :func:`build_run_name`).

    ``no_checkpoints`` disables intermediate checkpoint saving
    (``save_strategy="no"``); only the final model is written via
    ``trainer.save_model``.

    ``output_root``, when given, overrides ``output_dir`` from the config as
    the root directory the run name is written under (default ``outputs``).
    """
    cfg = _load_cpt_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    if dry_run:
        return _run_dry_run_cpt(cfg)

    # Imported lazily so dry_run (and CPU-only test/CI environments) never
    # need a working HF download / GPU stack.
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

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
    layer_indices = (
        list(layers) if layers_overridden else list(tuning.get("trainable_layer_indices", []))
    )
    select_trainable_layers(model, layer_indices)
    summary = trainable_param_summary(model)
    logger.info("Trainable param summary: %s", summary)

    dataset_cfg = cfg.get("dataset", {})
    if "train_path" not in dataset_cfg:
        raise ValueError("cpt config must set dataset.train_path (see configs/cpt/*.yaml)")
    seq_len = int(cfg.get("max_seq_len", 1024))
    cache_root = dataset_cfg.get("packed_cache_dir", "data/processed/packed")
    packed = build_or_load_packed(
        dataset_cfg["train_path"],
        tokenizer,
        seq_len,
        model_name,
        cache_root=cache_root,
        rebuild=rebuild_cache,
    )
    packed = apply_package(packed, package)

    sample_fraction = dataset_cfg.get("sample_fraction")
    if sample_fraction is not None:
        n = max(1, int(len(packed) * float(sample_fraction)))
        packed = packed[:n]
    if not packed:
        raise ValueError(
            f"No packed training sequences produced from {dataset_cfg['train_path']!r} "
            f"(seq_len={seq_len}); check the dataset and max_seq_len."
        )

    dataset = _PackedDataset(packed)

    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    run_name_prefix = logging_cfg.get("run_name_prefix", "cpt")
    run_name = build_run_name(run_name_prefix, package, layer_indices, layers_overridden)
    output_root_dir = (
        Path(output_root) if output_root is not None else Path(cfg.get("output_dir", "outputs"))
    )
    output_dir = str(output_root_dir / run_name)

    vram_cb = VramMonitorCallback()
    loss_cb = LossTrackerCallback()

    args = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
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
    parser = argparse.ArgumentParser(description="LFM2.5-JA packed CPT training (Issue #23)")
    parser.add_argument("--config", required=True, help="Path to configs/cpt/*.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="CPU-only dry run, no HF download / GPU required"
    )
    parser.add_argument(
        "--package",
        choices=PACKAGES,
        default="full",
        help="Training data package: full (all packed sequences), centi (1/100 subset), "
        "or deci (1/10 subset)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force re-tokenization and overwrite the packed cache on disk",
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

    result = run_cpt(
        args.config,
        dry_run=args.dry_run,
        package=args.package,
        rebuild_cache=args.rebuild_cache,
        layers=layers,
        no_checkpoints=args.no_checkpoints,
        output_root=args.output_root,
    )
    print(
        f"CPT run finished: loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f} "
        f"(trainable={result['trainable_summary']['trainable_pct']:.3f}%)"
    )
    if result["final_loss"] >= result["initial_loss"]:
        raise SystemExit("CPT run failed: loss did not decrease")


if __name__ == "__main__":
    main()
