"""GPU memory monitoring and OOM probing utilities."""

from __future__ import annotations

from typing import Any

import torch


def format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"


def get_vram_usage() -> dict[str, int]:
    """Return current CUDA VRAM usage in bytes."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
    }


def log_vram(prefix: str = "") -> dict[str, str]:
    """Log VRAM usage and return formatted values."""
    usage = get_vram_usage()
    formatted = {k: format_bytes(v) for k, v in usage.items()}
    if prefix:
        alloc = formatted["allocated"]
        reserved = formatted["reserved"]
        print(f"[VRAM:{prefix}] allocated={alloc} reserved={reserved}")
    return formatted


def reset_peak_memory() -> None:
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def is_oom_error(exc: BaseException) -> bool:
    """Return True if exception is likely CUDA OOM."""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg


def probe_result(
    seq_len: int,
    batch_size: int,
    lora_rank: int,
    success: bool,
    peak_bytes: int,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a single OOM probe result record."""
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "success": success,
        "peak_vram": peak_bytes,
        "peak_vram_human": format_bytes(peak_bytes),
        "error": error,
    }
