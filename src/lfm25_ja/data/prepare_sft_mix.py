"""End-to-end sft-002 mix pipeline: loaders -> format synthesis -> mix -> report
(Issue #105).

Orchestrates ``lfm25_ja.data.ichikara``, ``lfm25_ja.data.llm_jp_instruct``,
``lfm25_ja.data.aya_ja``, ``lfm25_ja.data.format_constraints``, and
``lfm25_ja.data.mix_sft`` into a single entry point driven by
``configs/data/mix_002.yaml``, writing the final mixture JSONL plus a
markdown stats report (counts, format-type breakdown, origin breakdown,
average response length).
"""

from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path
from typing import Any

from lfm25_ja.data.aya_ja import build_aya_ja_dataset, download_aya_raw
from lfm25_ja.data.clean import _write_jsonl
from lfm25_ja.data.format_constraints import build_format_constrained_samples
from lfm25_ja.data.ichikara import build_ichikara_dataset, download_ichikara_raw
from lfm25_ja.data.llm_jp_instruct import (
    build_llm_jp_instruct_dataset,
    download_llm_jp_instruct_raw,
)
from lfm25_ja.data.mix_sft import mix_sft_datasets
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

# Known mix component sources. Deliberately NOT a dict of captured function
# references: those would be resolved once at import time, which would make
# `unittest.mock.patch("lfm25_ja.data.prepare_sft_mix.download_ichikara_raw", ...)`
# a no-op (the dict entry would still point at the original, real function).
# `_load_component` below calls each loader by its bare module-level name
# instead, so patches on this module's attributes take effect as expected.
_KNOWN_SOURCES: tuple[str, ...] = ("ichikara", "llm_jp_instruct", "aya_ja")


def _tag_origin(rows: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    return [{**row, "origin": origin} for row in rows]


def _load_component(name: str) -> list[dict[str, Any]]:
    if name == "ichikara":
        raw = download_ichikara_raw()
        rows = build_ichikara_dataset(raw)
    elif name == "llm_jp_instruct":
        raw = download_llm_jp_instruct_raw()
        rows = build_llm_jp_instruct_dataset(raw)
    elif name == "aya_ja":
        raw = download_aya_raw()
        rows = build_aya_ja_dataset(raw)
    else:
        raise ValueError(f"Unknown mix component source: {name!r} (known: {_KNOWN_SOURCES})")
    return _tag_origin(rows, name)


def _response_lengths(rows: list[dict[str, Any]]) -> list[int]:
    lengths = []
    for row in rows:
        messages = row.get("messages", [])
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if assistant_msgs:
            lengths.append(len(assistant_msgs[-1].get("content", "")))
    return lengths


def render_mix_stats_report(stats: dict[str, Any]) -> str:
    lines = ["# sft-002 mix stats report (Issue #105)", ""]
    lines.append(f"- Total rows: {stats['total']}")
    lines.append(f"- Seed: {stats['seed']}")
    lines.append("")

    lines.append("## Component breakdown")
    lines.append("")
    lines.append("| component | selected | available | target |")
    lines.append("|---|---|---|---|")
    for name, s in stats["mix"]["components"].items():
        target = s["target"] if s["target"] is not None else "(all)"
        lines.append(f"| {name} | {s['selected']} | {s['available']} | {target} |")
    lines.append("")

    lines.append("## Format-constraint breakdown")
    lines.append("")
    lines.append("| verifier | count |")
    lines.append("|---|---|")
    for verifier, count in stats["format_counts"].items():
        lines.append(f"| {verifier} | {count} |")
    lines.append(f"| **total format-constrained** | **{stats['format_total']}** |")
    lines.append("")

    lines.append("## Origin breakdown (final mixture)")
    lines.append("")
    lines.append("| origin | count |")
    lines.append("|---|---|")
    for origin, count in stats["origin_counts"].items():
        lines.append(f"| {origin} | {count} |")
    lines.append("")

    length_stats = stats["response_length"]
    lines.append("## Response length (characters)")
    lines.append("")
    lines.append(
        f"- min={length_stats['min']} max={length_stats['max']} "
        f"mean={length_stats['mean']:.1f} median={length_stats['median']:.1f}"
    )

    return "\n".join(lines) + "\n"


def prepare_sft_mix(config_path: str | Path) -> dict[str, Any]:
    """Run the end-to-end sft-002 mix pipeline described by ``config_path``
    (see ``configs/data/mix_002.yaml``): pull each upstream component, build
    format-constrained samples from their pool, mix everything by target row
    count, write the mixture JSONL, and write a markdown stats report.

    Returns a stats dict: ``{"total", "seed", "mix", "format_counts",
    "format_total", "origin_counts", "response_length", "output_path",
    "report_path"}``.
    """
    config = load_config(config_path)
    mix_cfg = config.get("mix", config)

    seed = mix_cfg.get("seed", 42)
    output_path = mix_cfg["output_path"]
    stats_report_path = mix_cfg.get("stats_report")

    components_cfg: dict[str, Any] = mix_cfg.get("components", {})
    if not components_cfg:
        raise ValueError("mix.components must not be empty")

    components: dict[str, list[dict[str, Any]]] = {}
    targets: dict[str, int | None] = {}
    for name, entry in components_cfg.items():
        source = entry.get("source", name)
        components[name] = _load_component(source)
        targets[name] = entry.get("n_samples")

    # Format-constraint synthesis draws from the full upstream pool (every
    # non-format component, pre-sampling), matching Issue #105's "上流混合
    # (ichikara+llm-jp+aya-ja) から抽出" design.
    upstream_pool: list[dict[str, Any]] = []
    for rows in components.values():
        upstream_pool.extend(rows)

    fc_cfg = mix_cfg.get("format_constraints", {})
    fc_seed = fc_cfg.get("seed", seed)
    fc_targets = fc_cfg.get("targets", {})
    polite_form_origins = set(fc_cfg.get("polite_form_sources", []))

    format_rows = build_format_constrained_samples(
        upstream_pool, fc_targets, fc_seed, polite_form_origins=polite_form_origins or None
    )
    format_counts: dict[str, int] = {}
    for row in format_rows:
        verifier = row.get("verifier", "unknown")
        format_counts[verifier] = format_counts.get(verifier, 0) + 1

    components["format"] = format_rows
    targets["format"] = None  # format rows are always taken in full (already exact-sized)

    mixed, mix_stats = mix_sft_datasets(components, targets, seed=seed)

    _write_jsonl(output_path, mixed)

    origin_counts: dict[str, int] = {}
    for row in mixed:
        origin = row.get("origin", "unknown")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1

    lengths = _response_lengths(mixed)
    if lengths:
        length_stats = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
        }
    else:
        length_stats = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    stats = {
        "total": mix_stats["total"],
        "seed": seed,
        "mix": mix_stats,
        "format_counts": format_counts,
        "format_total": len(format_rows),
        "origin_counts": origin_counts,
        "response_length": length_stats,
        "output_path": str(output_path),
    }

    if stats_report_path:
        report = render_mix_stats_report(stats)
        Path(stats_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_report_path).write_text(report, encoding="utf-8")
        stats["report_path"] = str(stats_report_path)
        logger.info("Report written to %s", stats_report_path)

    logger.info("sft-002 mix written: %d rows -> %s", len(mixed), output_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build the sft-002 diversified SFT mix (Issue #105)"
    )
    parser.add_argument(
        "--config", default="configs/data/mix_002.yaml", help="Path to configs/data/mix_002.yaml"
    )
    args = parser.parse_args()

    result = prepare_sft_mix(args.config)
    logger.info(
        "Done: %d examples -> %s (report: %s)",
        result["total"],
        result["output_path"],
        result.get("report_path"),
    )


if __name__ == "__main__":
    main()
