#!/usr/bin/env python3
"""Apply the frozen K3 factual DPO decision gate (Issue #124 / #145)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lfm25_ja.eval.k3_decision import apply_k3_decision_gate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _jkb_overall(agg: dict[str, Any]) -> float:
    overall = agg["overall"]
    if isinstance(overall, dict):
        return float(overall["accuracy"])
    return float(overall)


def _ifeval_prompt_strict(agg: dict[str, Any]) -> float:
    if "prompt_strict_acc" in agg:
        return float(agg["prompt_strict_acc"])
    raise SystemExit("IFEval aggregate missing prompt_strict_acc")


def _llmjp_avg(result: dict[str, Any]) -> float:
    for key in ("AVG", "avg", "score"):
        if key in result:
            return float(result[key])
    if "scores" in result and isinstance(result["scores"], dict) and "AVG" in result["scores"]:
        return float(result["scores"]["AVG"])
    evaluation = result.get("evaluation")
    if isinstance(evaluation, dict):
        scores = evaluation.get("scores")
        if isinstance(scores, dict) and "AVG" in scores:
            return float(scores["AVG"])
    raise SystemExit("llm-jp-eval result missing AVG")


def _render_md(verdict: dict[str, Any]) -> str:
    lines = [
        f"# K3 factual DPO decision gate: **{verdict['verdict']}**",
        "",
        "| condition | pass | value | threshold |",
        "|---|---|---|---|",
    ]
    conds = verdict["conditions"]
    j = conds["jkb_overall_ge_52_9"]
    lines.append(
        f"| JKB overall ≥ 52.9% | {j['pass']} | {j['value']:.4f} | {j['threshold']:.4f} |"
    )
    i = conds["ifeval_prompt_strict_ge_0_920"]
    lines.append(
        f"| IFEval prompt_strict ≥ 0.920 | {i['pass']} | {i['value']:.4f} | {i['threshold']:.4f} |"
    )
    l = conds["llmjp_avg_ge_0_459"]
    lines.append(
        f"| llm-jp-eval AVG ≥ 0.459 | {l['pass']} | {l['value']:.4f} | {l['threshold']:.4f} |"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jkb", type=Path, required=True)
    parser.add_argument("--ifeval", type=Path, required=True)
    parser.add_argument("--llmjp", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    jkb_overall = _jkb_overall(_load_json(args.jkb))
    ifeval = _ifeval_prompt_strict(_load_json(args.ifeval))
    llmjp = _llmjp_avg(_load_json(args.llmjp))

    verdict = apply_k3_decision_gate(
        jkb_overall=jkb_overall,
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
