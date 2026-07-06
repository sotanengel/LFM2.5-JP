"""OOM probing: seq_len x batch x LoRA rank grid exploration."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from lfm25_ja.utils.config import load_project_config, project_root
from lfm25_ja.utils.memory import (
    get_vram_usage,
    is_oom_error,
    probe_result,
    reset_peak_memory,
)


def iter_probe_grid(cfg: dict[str, Any]) -> Iterator[dict[str, int]]:
    """Yield all probe combinations from config."""
    probe = cfg.get("memory_probe", {})
    for seq_len in probe.get("seq_lengths", [1024]):
        for batch_size in probe.get("batch_sizes", [1]):
            for lora_rank in probe.get("lora_ranks", [16]):
                yield {
                    "seq_len": int(seq_len),
                    "batch_size": int(batch_size),
                    "lora_rank": int(lora_rank),
                }


def summarize_probe_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize probe grid outcomes."""
    successful = [r for r in results if r.get("success")]
    max_ok = max(
        successful,
        key=lambda r: (r["seq_len"], r["batch_size"], r["lora_rank"]),
        default=None,
    )
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "max_successful": max_ok,
    }


def render_probe_report(results: list[dict[str, Any]]) -> str:
    """Render probe results as markdown table."""
    summary = summarize_probe_results(results)
    lines = [
        "# Phase 0 Memory Probe Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Total trials: {summary['total']}",
        f"- Successful: {summary['successful']}",
        f"- Failed: {summary['failed']}",
        "",
    ]
    if summary["max_successful"]:
        m = summary["max_successful"]
        lines.append(
            f"- Max successful config: seq_len={m['seq_len']}, "
            f"batch={m['batch_size']}, r={m['lora_rank']}"
        )
    lines.extend(
        [
            "",
            "## Grid Results",
            "",
            "| seq_len | batch | r | success | peak_vram | error |",
            "|---:|---:|---:|:---:|---|---|",
        ]
    )
    for r in results:
        lines.append(
            f"| {r['seq_len']} | {r['batch_size']} | {r['lora_rank']} | "
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
            params["lora_rank"],
            success=True,
            peak_bytes=0,
            error="CPU-only dry probe",
        )

    reset_peak_memory()
    try:
        hidden = 256
        seq_len = params["seq_len"]
        batch = params["batch_size"]
        rank = params["lora_rank"]
        # Lightweight surrogate matmul to approximate memory pressure
        a = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(hidden, hidden + rank * 4, device="cuda", dtype=torch.bfloat16)
        out = a @ b
        loss = out.mean()
        loss.backward()
        torch.cuda.synchronize()
        peak = get_vram_usage()["max_allocated"]
        del a, b, out, loss
        torch.cuda.empty_cache()
        return probe_result(
            params["seq_len"],
            params["batch_size"],
            params["lora_rank"],
            success=True,
            peak_bytes=peak,
        )
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        return probe_result(
            params["seq_len"],
            params["batch_size"],
            params["lora_rank"],
            success=False,
            peak_bytes=get_vram_usage()["max_allocated"],
            error=str(exc) if is_oom_error(exc) else str(exc),
        )


def run_probe_grid(
    cfg: dict[str, Any],
    trial_fn: Callable[[dict[str, int], dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Execute full probe grid."""
    fn = trial_fn or _default_trial
    return [fn(params, cfg) for params in iter_probe_grid(cfg)]


def write_probe_report(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Write markdown report to experiments/reports/phase0_memory.md."""
    path = output_path or (project_root() / "experiments" / "reports" / "phase0_memory.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_probe_report(results), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="OOM memory probe grid")
    parser.add_argument("--dry-run", action="store_true", help="Skip GPU trials")
    args = parser.parse_args()
    cfg = load_project_config("base.yaml")

    if args.dry_run:
        results = [
            probe_result(p["seq_len"], p["batch_size"], p["lora_rank"], True, 0)
            for p in iter_probe_grid(cfg)
        ]
    else:
        results = run_probe_grid(cfg)

    out = write_probe_report(results)
    summary = summarize_probe_results(results)
    print(f"Probe report written to {out}")
    print(f"Successful configs: {summary['successful']}/{summary['total']}")


if __name__ == "__main__":
    main()
