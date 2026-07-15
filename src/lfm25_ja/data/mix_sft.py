"""SFT-mixture sampling for the sft-002 mix (Issue #105).

Separate from ``lfm25_ja.data.mix`` (CPT language-ratio mixing, untouched by
this issue): SFT mixing here combines already-chat-formatted components
(ichikara / llm-jp-instruct / aya-ja / format-constrained) by row count
rather than by language-ratio/token volume, so the two concerns don't share
a function despite both being "mixing" in the abstract.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def mix_sft_datasets(
    components: dict[str, list[dict[str, Any]]],
    targets: dict[str, int | None],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample an SFT mixture from ``components`` (name -> list of chat-format
    rows).

    For each component, ``targets.get(name)`` rows are randomly sampled
    (seeded, deterministic); a missing key or a ``None`` value means "take
    every row" (used for the ichikara "全数" component and for the
    already-exact-sized format-constrained component). If a target exceeds
    the component's available row count, every available row is used and a
    WARN is logged.

    The combined selection is shuffled (same ``seed``) before being returned.
    Returns ``(mixed_rows, stats)`` where ``stats`` reports, per component,
    ``selected``/``available``/``target``, plus the overall ``total``.
    """
    if not components:
        raise ValueError("components must not be empty")

    rng = random.Random(seed)
    mixed: list[dict[str, Any]] = []
    component_stats: dict[str, Any] = {}

    for name, rows in components.items():
        available = len(rows)
        target = targets.get(name)

        if target is None:
            selected = list(rows)
        else:
            if target > available:
                logger.warning(
                    "mix_sft_datasets: component '%s' target=%d exceeds available=%d; "
                    "using all available rows",
                    name,
                    target,
                    available,
                )
            n = min(target, available)
            idxs = list(range(available))
            rng.shuffle(idxs)
            selected = [rows[i] for i in idxs[:n]]

        mixed.extend(selected)
        component_stats[name] = {
            "selected": len(selected),
            "available": available,
            "target": target,
        }

    rng.shuffle(mixed)

    stats = {"seed": seed, "total": len(mixed), "components": component_stats}
    return mixed, stats


def render_mix_sft_report(stats: dict[str, Any]) -> str:
    """Render a markdown summary table of SFT mixing stats."""
    lines = [
        "# SFT mixing report",
        "",
        f"- Seed: {stats.get('seed')}",
        f"- Total: {stats.get('total', 0)}",
        "",
        "| component | selected | available | target |",
        "|---|---|---|---|",
    ]
    for name, s in stats.get("components", {}).items():
        target = s["target"] if s["target"] is not None else "(all)"
        lines.append(f"| {name} | {s['selected']} | {s['available']} | {target} |")
    return "\n".join(lines)


def main() -> None:
    """Not used directly as a CLI entry point (see ``prepare_sft_mix.py``, which
    orchestrates loaders + format-constraint synthesis + this mixing step
    together); kept for ad-hoc invocation/debugging against pre-built
    per-component JSONL files."""
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Mix pre-built SFT component JSONL files by row count (Issue #105)"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="name=path.jsonl pairs, e.g. --inputs ichikara=a.jsonl aya_ja=b.jsonl",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=[],
        help="name=n pairs capping how many rows to sample per component (omit for 'all')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--output", required=True, help="Output JSONL path for the mixed dataset")
    parser.add_argument("--report", default=None, help="Optional markdown report output path")
    args = parser.parse_args()

    def _read_jsonl(path: str) -> list[dict[str, Any]]:
        rows = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    components: dict[str, list[dict[str, Any]]] = {}
    for pair in args.inputs:
        name, path = pair.split("=", 1)
        components[name] = _read_jsonl(path)

    targets: dict[str, int | None] = {}
    for pair in args.targets:
        name, n = pair.split("=", 1)
        targets[name] = int(n)

    mixed, stats = mix_sft_datasets(components, targets, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    logger.info("Mixed SFT dataset written: %d rows -> %s", len(mixed), args.output)

    if args.report:
        Path(args.report).write_text(render_mix_sft_report(stats), encoding="utf-8")
        logger.info("Report written to %s", args.report)


if __name__ == "__main__":
    main()
