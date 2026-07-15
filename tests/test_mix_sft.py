"""SFT-mixture sampling tests (Issue #105).

Separate suite from ``tests/test_data_pipeline.py``'s CPT ``mix_corpora``
coverage -- ``mix_sft_datasets`` samples by row count, not language ratio.
"""

from __future__ import annotations

import pytest
from lfm25_ja.data.mix_sft import mix_sft_datasets, render_mix_sft_report


def _rows(n: int, prefix: str) -> list[dict]:
    return [{"id": f"{prefix}{i}"} for i in range(n)]


def test_mix_sft_datasets_samples_up_to_target_per_component() -> None:
    components = {"a": _rows(10, "a"), "b": _rows(10, "b")}
    targets = {"a": 3, "b": 5}
    mixed, stats = mix_sft_datasets(components, targets, seed=42)

    assert len(mixed) == 8
    assert stats["components"]["a"]["selected"] == 3
    assert stats["components"]["b"]["selected"] == 5
    assert stats["total"] == 8


def test_mix_sft_datasets_none_target_takes_all_rows() -> None:
    components = {"a": _rows(7, "a")}
    targets = {"a": None}
    mixed, stats = mix_sft_datasets(components, targets, seed=42)

    assert len(mixed) == 7
    assert stats["components"]["a"]["selected"] == 7
    assert stats["components"]["a"]["target"] is None


def test_mix_sft_datasets_missing_target_key_takes_all_rows() -> None:
    components = {"a": _rows(4, "a")}
    mixed, _stats = mix_sft_datasets(components, targets={}, seed=42)
    assert len(mixed) == 4


def test_mix_sft_datasets_target_exceeding_available_uses_all_and_warns(caplog) -> None:
    components = {"a": _rows(3, "a")}
    targets = {"a": 100}
    with caplog.at_level("WARNING"):
        mixed, stats = mix_sft_datasets(components, targets, seed=42)

    assert len(mixed) == 3
    assert stats["components"]["a"]["selected"] == 3
    assert any("a" in r.message for r in caplog.records)


def test_mix_sft_datasets_is_deterministic_given_seed() -> None:
    components = {"a": _rows(20, "a"), "b": _rows(20, "b")}
    targets = {"a": 5, "b": 5}
    mixed_a, _ = mix_sft_datasets(components, targets, seed=42)
    mixed_b, _ = mix_sft_datasets(components, targets, seed=42)
    assert mixed_a == mixed_b


def test_mix_sft_datasets_different_seeds_can_differ() -> None:
    components = {"a": _rows(50, "a")}
    targets = {"a": 10}
    mixed_1, _ = mix_sft_datasets(components, targets, seed=1)
    mixed_2, _ = mix_sft_datasets(components, targets, seed=2)
    assert mixed_1 != mixed_2


def test_mix_sft_datasets_empty_components_raises() -> None:
    with pytest.raises(ValueError):
        mix_sft_datasets({}, {}, seed=42)


def test_mix_sft_datasets_all_rows_present_regardless_of_order() -> None:
    components = {"a": _rows(5, "a"), "b": _rows(5, "b")}
    mixed, _stats = mix_sft_datasets(components, targets={}, seed=42)
    ids = {row["id"] for row in mixed}
    assert ids == {f"a{i}" for i in range(5)} | {f"b{i}" for i in range(5)}


def test_render_mix_sft_report_includes_component_rows() -> None:
    stats = {
        "seed": 42,
        "total": 8,
        "components": {
            "a": {"selected": 3, "available": 10, "target": 3},
            "b": {"selected": 5, "available": 10, "target": None},
        },
    }
    report = render_mix_sft_report(stats)
    assert "| a | 3 | 10 | 3 |" in report
    assert "| b | 5 | 10 | (all) |" in report
