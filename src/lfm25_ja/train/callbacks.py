"""Training callbacks for VRAM monitoring and sample logging."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from lfm25_ja.utils.memory import get_vram_usage, log_vram, reset_peak_memory


class VramMonitorCallback(TrainerCallback):
    """Log VRAM usage at each logging step."""

    def __init__(self, prefix: str = "train") -> None:
        self.prefix = prefix
        self.peak_bytes = 0

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        reset_peak_memory()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        usage = get_vram_usage()
        self.peak_bytes = max(self.peak_bytes, usage["max_allocated"])
        if state.global_step % max(args.logging_steps, 1) == 0:
            log_vram(self.prefix)


class LossTrackerCallback(TrainerCallback):
    """Track training loss history."""

    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))
