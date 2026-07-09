"""OOM probe grid tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from lfm25_ja.utils.memory_probe import (
    iter_probe_grid,
    make_real_trial,
    render_probe_report,
    run_probe_grid,
    summarize_probe_results,
)


def test_iter_probe_grid() -> None:
    cfg = {
        "memory_probe": {
            "seq_lengths": [1024, 2048],
            "batch_sizes": [1, 2],
            "n_trainable_layers": [1],
        }
    }
    grid = list(iter_probe_grid(cfg))
    assert len(grid) == 4
    assert grid[0] == {"seq_len": 1024, "batch_size": 1, "n_trainable_layers": 1}


def test_render_probe_report() -> None:
    results = [
        {
            "seq_len": 1024,
            "batch_size": 1,
            "n_trainable_layers": 1,
            "success": True,
            "peak_vram_human": "4.5 GiB",
            "error": None,
        },
        {
            "seq_len": 4096,
            "batch_size": 4,
            "n_trainable_layers": 4,
            "success": False,
            "peak_vram_human": "N/A",
            "error": "CUDA OOM",
        },
    ]
    md = render_probe_report(results)
    assert "1024" in md
    assert "CUDA OOM" in md
    assert "| seq_len |" in md
    assert "Issue #57" in md


def test_summarize_probe_results() -> None:
    results = [
        {"success": True, "seq_len": 1024, "batch_size": 1, "n_trainable_layers": 1},
        {"success": False, "seq_len": 2048, "batch_size": 2, "n_trainable_layers": 4},
    ]
    summary = summarize_probe_results(results)
    assert summary["max_successful"]["seq_len"] == 1024
    assert summary["total"] == 2
    assert summary["successful"] == 1


def _real_mode_cfg() -> dict[str, Any]:
    return {
        "model_name": "LiquidAI/LFM2.5-1.2B-Instruct",
        "memory_probe": {
            "seq_lengths": [1024, 2048],
            "batch_sizes": [1],
            "n_trainable_layers": [1, 2],
            "start_layer": 8,
        },
    }


def test_run_probe_grid_real_mode_plumbing() -> None:
    """A fake trial injected via `trial=` should be called once per grid point,
    with the OOM point surfacing status="oom" and the rest status="ok"."""
    calls: list[dict[str, int]] = []

    def fake_trial(params: dict[str, int], cfg: dict[str, Any]) -> dict[str, Any]:
        calls.append(params)
        if params["seq_len"] == 2048 and params["n_trainable_layers"] == 2:
            return {
                "seq_len": params["seq_len"],
                "batch_size": params["batch_size"],
                "n_trainable_layers": params["n_trainable_layers"],
                "success": False,
                "status": "oom",
                "peak_vram": 0,
                "peak_vram_human": "N/A",
                "error": "CUDA out of memory",
            }
        return {
            "seq_len": params["seq_len"],
            "batch_size": params["batch_size"],
            "n_trainable_layers": params["n_trainable_layers"],
            "success": True,
            "status": "ok",
            "peak_vram": 123,
            "peak_vram_human": "123 B",
            "error": None,
        }

    cfg = _real_mode_cfg()
    results = run_probe_grid(cfg, trial=fake_trial, mode="real")

    grid = list(iter_probe_grid(cfg))
    assert len(calls) == len(grid)
    assert len(results) == len(grid)

    oom_results = [r for r in results if r["status"] == "oom"]
    assert len(oom_results) == 1
    assert oom_results[0]["seq_len"] == 2048
    assert oom_results[0]["n_trainable_layers"] == 2

    ok_results = [r for r in results if r["status"] == "ok"]
    assert len(ok_results) == len(grid) - 1


def test_render_probe_report_real_mode_header_and_wddm_note() -> None:
    results = [
        {
            "seq_len": 6144,
            "batch_size": 1,
            "n_trainable_layers": 1,
            "success": True,
            "status": "ok",
            "peak_vram_human": "7.16 GiB",
            "error": None,
        },
        {
            "seq_len": 24576,
            "batch_size": 1,
            "n_trainable_layers": 1,
            "success": False,
            "status": "oom",
            "peak_vram_human": "N/A",
            "error": "CUDA out of memory",
        },
    ]
    md = render_probe_report(
        results,
        mode="real",
        model_name="LiquidAI/LFM2.5-1.2B-Instruct",
        gpu_name="NVIDIA GeForce RTX 3060 Ti",
    )
    assert "実モデル実測" in md
    assert "LiquidAI/LFM2.5-1.2B-Instruct" in md
    assert "NVIDIA GeForce RTX 3060 Ti" in md
    assert "grad ckpt" in md
    assert "WDDM" in md
    assert "代理計測" not in md


def test_render_probe_report_surrogate_mode_unchanged() -> None:
    results = [
        {
            "seq_len": 1024,
            "batch_size": 1,
            "n_trainable_layers": 1,
            "success": True,
            "peak_vram_human": "4.5 GiB",
            "error": None,
        }
    ]
    md = render_probe_report(results)
    assert "代理計測" in md
    assert "WDDM" not in md
    assert "実モデル実測" not in md


def test_make_real_trial_requires_cuda() -> None:
    cfg = _real_mode_cfg()
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError):
            make_real_trial(cfg)
