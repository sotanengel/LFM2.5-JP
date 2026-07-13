"""OOM probing: seq_len x batch x n_trainable_layers grid exploration.

Two measurement modes are available:

- ``mode="surrogate"`` (default): a lightweight matmul sized by the grid
  parameters, used to approximate memory pressure cheaply. Not an actual
  VRAM measurement of the real LFM2.5 model.
- ``mode="real"``: loads the actual LFM2.5 model once and runs a short HF
  ``Trainer`` training run per grid point, under layer-selective
  full-parameter fine-tuning conditions (bf16, grad checkpointing,
  paged_adamw_8bit — same path as ``lfm25_ja.train.smoke``) to measure true
  peak VRAM. See Issue #57.
"""

from __future__ import annotations

import argparse
import gc
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from lfm25_ja.train.layer_select import select_trainable_layers
from lfm25_ja.utils.config import load_project_config, project_root
from lfm25_ja.utils.memory import (
    format_bytes,
    get_vram_usage,
    is_oom_error,
    probe_result,
    reset_peak_memory,
)

logger = logging.getLogger(__name__)


def iter_probe_grid(cfg: dict[str, Any]) -> Iterator[dict[str, int]]:
    """Yield all probe combinations from config."""
    probe = cfg.get("memory_probe", {})
    for seq_len in probe.get("seq_lengths", [1024]):
        for batch_size in probe.get("batch_sizes", [1]):
            for n_trainable_layers in probe.get("n_trainable_layers", [1]):
                yield {
                    "seq_len": int(seq_len),
                    "batch_size": int(batch_size),
                    "n_trainable_layers": int(n_trainable_layers),
                }


def summarize_probe_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize probe grid outcomes."""
    successful = [r for r in results if r.get("success")]
    max_ok = max(
        successful,
        key=lambda r: (r["seq_len"], r["batch_size"], r["n_trainable_layers"]),
        default=None,
    )
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "max_successful": max_ok,
    }


def render_probe_report(
    results: list[dict[str, Any]],
    mode: str = "surrogate",
    model_name: str | None = None,
    gpu_name: str | None = None,
) -> str:
    """Render probe results as markdown table.

    mode="surrogate" (default): notes that this is a lightweight surrogate
    measurement, not an actual VRAM measurement of the real model (Issue #57).

    mode="real": headers the report with the real model name / GPU name /
    grad-checkpointing state, and always includes a note about Windows WDDM
    memory spilling (peak may exceed physical VRAM without triggering OOM,
    since the effective ceiling is the physical VRAM).
    """
    summary = summarize_probe_results(results)
    lines = ["# Phase 0 Memory Probe Report", ""]
    if mode == "real":
        detail_bits = ["実モデル・GPU 上で HF Trainer による 2 step 学習を実行して計測"]
        if model_name:
            detail_bits.append(f"モデル: {model_name}")
        if gpu_name:
            detail_bits.append(f"GPU: {gpu_name}")
        detail_bits.append("grad ckpt 有効")
        lines.append(f"> **実モデル実測**: {' / '.join(detail_bits)}。")
        lines.append("")
        lines.append(
            "> **注記**: Windows WDDM は物理 VRAM 超過分をシステム RAM にスピルするため、"
            "peak が物理容量を超えていても OOM しない場合がある"
            "(実効上限は物理 VRAM 内)。"
        )
    else:
        lines.append(
            "> **注記**: これは軽量な代理計測であり実モデルの VRAM 実測ではない。"
            "実測は Issue #57 で対応。"
        )
    lines.extend(
        [
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "## Summary",
            "",
        ]
    )
    lines.extend(
        [
            f"- Total trials: {summary['total']}",
            f"- Successful: {summary['successful']}",
            f"- Failed: {summary['failed']}",
            "",
        ]
    )
    if summary["max_successful"]:
        m = summary["max_successful"]
        lines.append(
            f"- Max successful config: seq_len={m['seq_len']}, "
            f"batch={m['batch_size']}, n_layers={m['n_trainable_layers']}"
        )
    lines.extend(
        [
            "",
            "## Grid Results",
            "",
            "| seq_len | batch | n_layers | success | peak_vram | error |",
            "|---:|---:|---:|:---:|---|---|",
        ]
    )
    for r in results:
        lines.append(
            f"| {r['seq_len']} | {r['batch_size']} | {r['n_trainable_layers']} | "
            f"{'OK' if r['success'] else 'OOM'} | {r.get('peak_vram_human', 'N/A')} | "
            f"{r.get('error') or ''} |"
        )
    return "\n".join(lines) + "\n"


def _default_trial(params: dict[str, int], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one probe trial on GPU using a lightweight forward/backward pass."""
    if not torch.cuda.is_available():
        return probe_result(
            params["seq_len"],
            params["batch_size"],
            params["n_trainable_layers"],
            success=True,
            peak_bytes=0,
            error="CPU-only dry probe",
        )

    reset_peak_memory()
    try:
        hidden = 256
        seq_len = params["seq_len"]
        batch = params["batch_size"]
        n_trainable_layers = params["n_trainable_layers"]
        # Lightweight surrogate matmul to approximate memory pressure
        a = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(
            hidden, hidden + n_trainable_layers * 4, device="cuda", dtype=torch.bfloat16
        )
        with torch.no_grad():
            out = a @ b
        torch.cuda.synchronize()
        peak = get_vram_usage()["max_allocated"]
        del a, b, out
        torch.cuda.empty_cache()
        return probe_result(
            params["seq_len"],
            params["batch_size"],
            params["n_trainable_layers"],
            success=True,
            peak_bytes=peak,
        )
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        return probe_result(
            params["seq_len"],
            params["batch_size"],
            params["n_trainable_layers"],
            success=False,
            peak_bytes=get_vram_usage()["max_allocated"],
            error=str(exc) if is_oom_error(exc) else str(exc),
        )


def _build_random_token_dataset(
    vocab_size: int,
    num_samples: int,
    seq_len: int,
) -> torch.utils.data.Dataset:
    """Build a CPU dataset of random token rows (input_ids/attention_mask/labels).

    No tokenizer required: token ids are sampled below ``vocab_size``. Rows
    live on CPU; the Trainer's dataloader moves each batch to the model
    device, mirroring the real training data flow.
    """

    class _RandomTokenDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return num_samples

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": input_ids.clone(),
            }

    return _RandomTokenDataset()


def _release_trial_memory(model: Any) -> None:
    """Drop per-trial GPU references (grads) and free CUDA allocator blocks."""
    for param in model.parameters():
        param.grad = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def make_real_trial(
    cfg: dict[str, Any],
) -> Callable[[dict[str, int], dict[str, Any]], dict[str, Any]]:
    """Build a trial closure that measures real VRAM usage against the actual model.

    Loads the model referenced by ``cfg["model_name"]`` exactly once using
    :func:`lfm25_ja.train.train_cpt.build_from_pretrained_kwargs` (bf16 by
    default, or NF4 4bit when ``tuning.load_in_4bit`` is true) with
    ``device_map="auto"``, and reuses it for every grid point. Each trial
    unfreezes a window of ``n_trainable_layers`` consecutive layers starting
    at ``cfg["memory_probe"]["start_layer"]`` (default 8) and runs a 2-step
    HF ``Trainer`` training run on random-token dummy data, using the same
    ``TrainingArguments`` path as ``lfm25_ja.train.smoke`` (bf16, gradient
    checkpointing, paged_adamw_8bit) whose peak-VRAM readings match known
    manual measurements. The Trainer path also disables the generation cache
    during training, so no KV/conv cache memory is counted.

    After each trial, all per-trial references (Trainer, optimizer state,
    dataset, gradients) are dropped, then ``gc.collect()`` and
    ``torch.cuda.empty_cache()`` run so trials do not leak memory into one
    another.

    Raises:
        RuntimeError: if no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "make_real_trial requires a CUDA device, but none was found. "
            "Real-mode memory probing cannot run on a CPU-only environment."
        )

    # Lazy import: heavy dependency, keep module import light.
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from lfm25_ja.train.layer_select import upcast_trainable_layers
    from lfm25_ja.train.train_cpt import build_from_pretrained_kwargs

    model_name = cfg["model_name"]
    tuning = cfg.get("tuning", {})
    load_kwargs = build_from_pretrained_kwargs(tuning)
    load_mode = "4bit-NF4" if tuning.get("load_in_4bit") else "bf16"
    logger.info(
        "Loading model %s for real memory probe (%s, device_map=auto)",
        model_name,
        load_mode,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    # Training never uses the generation cache; leaving it on both wastes
    # VRAM (KV/conv cache per forward) and conflicts with grad checkpointing.
    model.config.use_cache = False
    if tuning.get("load_in_4bit") and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    probe_cfg = cfg.get("memory_probe", {})
    start_layer = int(probe_cfg.get("start_layer", 8))
    train_cfg = cfg.get("training", {})
    output_dir = str(cfg.get("output_dir", "outputs")) + "/memory_probe"

    def _trial(params: dict[str, int], _cfg: dict[str, Any]) -> dict[str, Any]:
        seq_len = params["seq_len"]
        batch_size = params["batch_size"]
        n_trainable_layers = params["n_trainable_layers"]
        layer_indices = list(range(start_layer, start_layer + n_trainable_layers))
        max_steps = 2

        logger.info(
            "Real probe trial: seq_len=%d batch=%d n_layers=%d layer_indices=%s",
            seq_len,
            batch_size,
            n_trainable_layers,
            layer_indices,
        )

        if tuning.get("load_in_4bit"):
            upcast_trainable_layers(model, layer_indices)
        else:
            select_trainable_layers(model, layer_indices)

        trainer = None
        dataset = None
        try:
            dataset = _build_random_token_dataset(
                vocab_size=model.config.vocab_size,
                num_samples=max_steps * batch_size,
                seq_len=seq_len,
            )
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                max_steps=max_steps,
                learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                fp16=False,
                bf16=True,
                gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
                optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
                remove_unused_columns=False,
                disable_tqdm=True,
            )
            trainer = Trainer(model=model, args=args, train_dataset=dataset)
            reset_peak_memory()
            trainer.train()
            peak = get_vram_usage()["max_allocated"]
            logger.info("Real probe trial OK: peak=%s", format_bytes(peak))
            return probe_result(seq_len, batch_size, n_trainable_layers, True, peak)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            peak = get_vram_usage()["max_allocated"]
            logger.info("Real probe trial OOM: peak=%s error=%s", format_bytes(peak), exc)
            return probe_result(
                seq_len, batch_size, n_trainable_layers, False, peak, error=str(exc)
            )
        finally:
            # Drop every per-trial reference (the Trainer holds the optimizer
            # state and dataloader; gradients hang off the model parameters)
            # BEFORE empty_cache(), otherwise referenced blocks survive and
            # leak into the next trial.
            if trainer is not None:
                trainer.optimizer = None
                trainer.lr_scheduler = None
            del trainer, dataset
            _release_trial_memory(model)

    return _trial


def run_probe_grid(
    cfg: dict[str, Any],
    trial: Callable[[dict[str, int], dict[str, Any]], dict[str, Any]] | None = None,
    mode: str = "surrogate",
) -> list[dict[str, Any]]:
    """Execute full probe grid.

    A ``trial`` callable can be injected regardless of ``mode`` (used by
    tests). If not given, ``mode="real"`` builds a real-model trial via
    ``make_real_trial``; any other mode uses the lightweight surrogate trial.
    """
    if trial is not None:
        fn = trial
    elif mode == "real":
        fn = make_real_trial(cfg)
    else:
        fn = _default_trial
    return [fn(params, cfg) for params in iter_probe_grid(cfg)]


def write_probe_report(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
    mode: str = "surrogate",
    model_name: str | None = None,
    gpu_name: str | None = None,
) -> Path:
    """Write markdown report to experiments/reports/phase0_memory.md."""
    path = output_path or (project_root() / "experiments" / "reports" / "phase0_memory.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_probe_report(results, mode=mode, model_name=model_name, gpu_name=gpu_name),
        encoding="utf-8",
    )
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="OOM memory probe grid")
    parser.add_argument("--dry-run", action="store_true", help="Skip GPU trials")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Measure real model VRAM usage instead of the lightweight surrogate",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional experiment YAML merged over configs/base.yaml "
        "(e.g. configs/cpt/cpt_8b_a1b_layerft.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Report output path (default: experiments/reports/phase0_memory.md)",
    )
    args = parser.parse_args()
    cfg = load_project_config("base.yaml")
    if args.config:
        from lfm25_ja.utils.config import load_config, merge_configs

        cfg = merge_configs(cfg, load_config(args.config))

    mode = "real" if args.real else "surrogate"
    model_name: str | None = None
    gpu_name: str | None = None

    if args.dry_run:
        results = [
            probe_result(p["seq_len"], p["batch_size"], p["n_trainable_layers"], True, 0)
            for p in iter_probe_grid(cfg)
        ]
    elif mode == "real":
        model_name = cfg.get("model_name")
        results = run_probe_grid(cfg, mode="real")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
    else:
        results = run_probe_grid(cfg)

    output_path = Path(args.output) if args.output else None
    out = write_probe_report(
        results, output_path=output_path, mode=mode, model_name=model_name, gpu_name=gpu_name
    )
    summary = summarize_probe_results(results)
    print(f"Probe report written to {out}")
    print(f"Successful configs: {summary['successful']}/{summary['total']}")


if __name__ == "__main__":
    main()
