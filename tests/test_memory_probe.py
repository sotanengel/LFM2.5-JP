"""OOM probe grid tests."""

from __future__ import annotations

from lfm25_ja.utils.memory_probe import (
    iter_probe_grid,
    render_probe_report,
    summarize_probe_results,
)


def test_iter_probe_grid() -> None:
    cfg = {
        "memory_probe": {
            "seq_lengths": [1024, 2048],
            "batch_sizes": [1, 2],
            "lora_ranks": [16],
        }
    }
    grid = list(iter_probe_grid(cfg))
    assert len(grid) == 4
    assert grid[0] == {"seq_len": 1024, "batch_size": 1, "lora_rank": 16}


def test_render_probe_report() -> None:
    results = [
        {
            "seq_len": 1024,
            "batch_size": 1,
            "lora_rank": 16,
            "success": True,
            "peak_vram_human": "4.5 GiB",
            "error": None,
        },
        {
            "seq_len": 4096,
            "batch_size": 4,
            "lora_rank": 64,
            "success": False,
            "peak_vram_human": "N/A",
            "error": "CUDA OOM",
        },
    ]
    md = render_probe_report(results)
    assert "1024" in md
    assert "CUDA OOM" in md
    assert "| seq_len |" in md


def test_summarize_probe_results() -> None:
    results = [
        {"success": True, "seq_len": 1024, "batch_size": 1, "lora_rank": 16},
        {"success": False, "seq_len": 2048, "batch_size": 2, "lora_rank": 64},
    ]
    summary = summarize_probe_results(results)
    assert summary["max_successful"]["seq_len"] == 1024
    assert summary["total"] == 2
    assert summary["successful"] == 1
