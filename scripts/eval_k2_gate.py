#!/usr/bin/env python3
"""Apply the frozen K2 cpt-D decision gate to measured metrics (Issue #132 / #138).

Reads JKB aggregate JSON (overall + by_domain accuracies), IFEval aggregate
prompt_strict_acc, and llm-jp-eval AVG, then writes a gate verdict JSON/Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lfm25_ja.eval.k2_decision import apply_k2_decision_gate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _jkb_overall_and_domains(agg: dict[str, Any]) -> tuple[float, dict[str, float]]:
    overall = agg["overall"]
    if isinstance(overall, dict):
        overall_acc = float(overall["accuracy"])
    else:
        overall_acc = float(overall)
    by_domain_raw = agg.get("by_domain") or {}
    by_domain: dict[str, float] = {}
    for name, stat in by_domain_raw.items():
        if isinstance(stat, dict):
            by_domain[name] = float(stat["accuracy"])
        else:
            by_domain[name] = float(stat)
    return overall_acc, by_domain


def _ifeval_prompt_strict(agg: dict[str, Any]) -> float:
    if "prompt_strict_acc" in agg:
        return float(agg["prompt_strict_acc"])
    raise SystemExit("IFEval aggregate missing prompt_strict_acc")


def _llmjp_avg(result: dict[str, Any]) -> float:
    for key in ("AVG", "avg", "score"):
        if key in result:
            return float(result[key])
    # Nested forms used by some llm-jp-eval dumps
    if "scores" in result and isinstance(result["scores"], dict):
        scores = result["scores"]
        if "AVG" in scores:
            return float(scores["AVG"])
    raise SystemExit("llm-jp-eval result missing AVG")


def _render_md(verdict: dict[str, Any]) -> str:
    lines = [
        f"# K2 decision gate: **{verdict['verdict']}**",
        "",
        "| condition | pass | value | threshold |",
        "|---|---|---|---|",
    ]
    conds = verdict["conditions"]
    o = conds["jkb_overall_ge_60"]
    lines.append(
        f"| JKB overall ≥ 60% | {o['pass']} | {o['value']:.4f} | {o['threshold']:.4f} |"
    )
    d = conds["jkb_domains_within_3pt_of_base"]
    n_fail = len(d["failures"])
    lines.append(
        f"| JKB domains within base−3pt | {d['pass']} | failures={n_fail} | "
        f"drop≤{d['threshold_drop_max']:.4f} |"
    )
    i = conds["ifeval_prompt_strict_ge_0_920"]
    lines.append(
        f"| IFEval prompt_strict ≥ 0.920 | {i['pass']} | {i['value']:.4f} | {i['threshold']:.4f} |"
    )
    l = conds["llmjp_avg_ge_0_459"]
    lines.append(
        f"| llm-jp-eval AVG ≥ 0.459 | {l['pass']} | {l['value']:.4f} | {l['threshold']:.4f} |"
    )
    if d["failures"]:
        lines.extend(["", "## Domain failures", ""])
        for f in d["failures"]:
            lines.append(
                f"- {f.get('domain')}: reason={f.get('reason')} "
                f"cand={f.get('candidate')} base={f.get('base')} delta={f.get('delta')}"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jkb-candidate", type=Path, required=True)
    parser.add_argument("--jkb-base", type=Path, required=True)
    parser.add_argument("--ifeval", type=Path, required=True)
    parser.add_argument("--llmjp", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    cand_overall, cand_domains = _jkb_overall_and_domains(_load_json(args.jkb_candidate))
    _base_overall, base_domains = _jkb_overall_and_domains(_load_json(args.jkb_base))
    ifeval = _ifeval_prompt_strict(_load_json(args.ifeval))
    llmjp = _llmjp_avg(_load_json(args.llmjp))

    verdict = apply_k2_decision_gate(
        jkb_overall=cand_overall,
        jkb_by_domain=cand_domains,
        base_by_domain=base_domains,
        ifeval_prompt_strict=ifeval,
        llmjp_avg=llmjp,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = _render_md(verdict)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
