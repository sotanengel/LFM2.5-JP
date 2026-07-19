"""K1 資産再評価: 全モデル一括評価&統計処理ドライバ (Issue #122).

このスクリプトは、既に個別に生成された JKB v1 の scored.jsonl / IFEval の
aggregate.json / llm-jp-eval の result_baseline-*.json を読み込み、
以下を出力する:

- 全モデル×指標のクロス表 (JKB overall / by_difficulty / by_domain,
  IFEval prompt_strict, llm-jp-eval AVG + niilc)
- ワースト分野×難度セル一覧 (K2 CPT ターゲット選定入力)
- base 比 McNemar p 値 + bootstrap 95% CI
- 目標水準の推定 (base +10pt/+15pt/+9pt を初期値とし、実測分布を反映して固定)

前提: JKB は scripts/run_jkb.py で既に生成済み、IFEval は
scripts/60_eval_ifeval_ja.sh、llm-jp-eval は scripts/50_eval_all.sh で済み。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from lfm25_ja.eval import jkb

logger = logging.getLogger(__name__)


K1_MODELS: tuple[str, ...] = ("base", "cptB-final", "dpo-001-b005", "sft005-distill")


def _load_scored(scored_path: Path) -> dict[str, dict[str, Any]]:
    """Load one model's JKB scored.jsonl into {id -> per_row_dict}."""
    rows: dict[str, dict[str, Any]] = {}
    with scored_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _mcnemar_p(b: int, c: int) -> float:
    """Two-tailed exact McNemar p-value for a 2x2 paired table with discordant counts b, c.

    Uses the binomial exact test on min(b, c) with n = b + c (mid-p not applied).
    Returns 1.0 when b + c == 0 (no evidence of difference).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k | n, 0.5) * 2
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    p_one_sided = total / (2**n)
    return min(1.0, 2 * p_one_sided)


def _bootstrap_diff_ci(
    a_correct: list[bool],
    b_correct: list[bool],
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI for (mean(a) - mean(b))."""
    n = len(a_correct)
    assert n == len(b_correct)
    rng = random.Random(seed)
    diffs: list[float] = []
    idxs = list(range(n))
    for _ in range(n_boot):
        sample = [rng.choice(idxs) for _ in range(n)]
        a_mean = sum(a_correct[i] for i in sample) / n
        b_mean = sum(b_correct[i] for i in sample) / n
        diffs.append(a_mean - b_mean)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    return (lo, hi)


def compare_to_base(
    model_rows: dict[str, dict[str, dict[str, Any]]],
    ids_in_order: list[str],
) -> dict[str, dict[str, Any]]:
    """For each non-base model, compute McNemar + paired bootstrap CI vs base."""
    base_correct = [bool(model_rows["base"][i]["correct"]) for i in ids_in_order]
    result: dict[str, dict[str, Any]] = {}
    for label, rows in model_rows.items():
        if label == "base":
            continue
        m_correct = [bool(rows[i]["correct"]) for i in ids_in_order]
        # b = model right, base wrong; c = base right, model wrong
        b = sum(1 for x, y in zip(m_correct, base_correct) if x and not y)
        c = sum(1 for x, y in zip(m_correct, base_correct) if y and not x)
        p = _mcnemar_p(b, c)
        lo, hi = _bootstrap_diff_ci(m_correct, base_correct)
        result[label] = {
            "delta_pct": (sum(m_correct) - sum(base_correct)) / len(base_correct) * 100,
            "mcnemar_b_win": b,
            "mcnemar_c_lose": c,
            "mcnemar_p": p,
            "bootstrap_ci_pt": [lo * 100, hi * 100],
        }
    return result


def _read_llm_jp_eval_score(path: Path) -> dict[str, Any]:
    """Load AVG + niilc scores from a WSL llm-jp-eval result_baseline-*.json."""
    d = json.loads(path.read_text(encoding="utf-8"))
    ev = d["evaluation"]
    scores = ev["scores"]
    return {
        "avg": scores.get("AVG"),
        "niilc_exact_match": scores.get("niilc_exact_match"),
        "niilc_char_f1": scores.get("niilc_char_f1"),
        "niilc_ool": scores.get("niilc_ool"),
        "jsem_ool": scores.get("jsem_ool"),  # >0.5 = fix未適用の可能性
        "path": str(path),
    }


def _read_ifeval_aggregate(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    verifier_polite = d.get("by_verifier", {}).get("polite_form", {})
    verifier_char = d.get("by_verifier", {}).get("char_count", {})
    return {
        "prompt_strict": d["prompt_strict_acc"],
        "prompt_loose": d["prompt_loose_acc"],
        "instruction_strict": d["instruction_strict_acc"],
        "polite_form": verifier_polite.get("instruction_strict_acc"),
        "char_count": verifier_char.get("instruction_strict_acc"),
        "path": str(path),
    }


def _load_all(
    jkb_root: Path,
    ifeval_root: Path,
    llmjp_paths: dict[str, Path],
) -> dict[str, Any]:
    """Cross-load JKB + IFEval + llm-jp-eval results for every K1 model."""
    dataset = jkb.load_jkb_jsonl("datasets/eval/jkb/eval.jsonl")
    ids_in_order = [row["id"] for row in dataset]

    model_rows: dict[str, dict[str, dict[str, Any]]] = {}
    aggs: dict[str, dict[str, Any]] = {}
    for label in K1_MODELS:
        scored_path = jkb_root / label / "scored.jsonl"
        rows_map = _load_scored(scored_path)
        model_rows[label] = rows_map
        # Re-aggregate directly over per-row correct/incorrect (scored.jsonl carries
        # the flag, so we don't need to rescore via jkb.aggregate here).
        agg_by_domain: dict[str, list[bool]] = {}
        agg_by_diff: dict[str, list[bool]] = {}
        agg_by_cell: dict[tuple[str, str], list[bool]] = {}
        all_correct: list[bool] = []
        for row in dataset:
            rid = row["id"]
            correct = bool(rid in rows_map and rows_map[rid]["correct"])
            all_correct.append(correct)
            agg_by_domain.setdefault(row["domain"], []).append(correct)
            agg_by_diff.setdefault(row["difficulty"], []).append(correct)
            agg_by_cell.setdefault((row["domain"], row["difficulty"]), []).append(correct)
        aggs[label] = {
            "overall": {
                "n": len(all_correct),
                "correct": sum(all_correct),
                "accuracy": sum(all_correct) / len(all_correct),
            },
            "by_domain": {
                d: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                for d, v in agg_by_domain.items()
            },
            "by_difficulty": {
                d: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                for d, v in agg_by_diff.items()
            },
            "by_cell": {
                f"{d}::{diff}": {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                for (d, diff), v in agg_by_cell.items()
            },
        }

    stats_vs_base = compare_to_base(model_rows, ids_in_order)

    ifeval: dict[str, Any] = {}
    ifeval_label_map = {
        "base": "base-jp202606",
        "cptB-final": "cptB-final",
        "dpo-001-b005": "dpo-001-b005",
        "sft005-distill": "sft005-distill",
    }
    for label, ife_label in ifeval_label_map.items():
        path = ifeval_root / ife_label / "aggregate.json"
        if path.exists():
            ifeval[label] = _read_ifeval_aggregate(path)
        else:
            ifeval[label] = {"missing": True, "path": str(path)}

    llmjp: dict[str, Any] = {}
    for label, path in llmjp_paths.items():
        if path is None or not path.exists():
            llmjp[label] = {"missing": True, "path": str(path) if path else None}
        else:
            llmjp[label] = _read_llm_jp_eval_score(path)

    return {
        "jkb": aggs,
        "jkb_stats_vs_base": stats_vs_base,
        "ifeval": ifeval,
        "llm_jp_eval": llmjp,
    }


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "-"


def render_summary_markdown(summary: dict[str, Any]) -> str:
    """Emit the consolidated K1 report table."""
    out: list[str] = ["# K1 統合サマリ", ""]

    # Overall
    out += [
        "## JKB v1 全体 (base 比較)",
        "",
        "| model | overall | vs base (pt) | McNemar p | 95% bootstrap CI (pt) |",
        "|---|---|---|---|---|",
    ]
    for label in K1_MODELS:
        acc = summary["jkb"][label]["overall"]["accuracy"]
        if label == "base":
            out.append(f"| {label} | {acc * 100:.1f}% | (基準) | - | - |")
            continue
        s = summary["jkb_stats_vs_base"][label]
        out.append(
            f"| {label} | {acc * 100:.1f}% | {s['delta_pct']:+.1f} | {s['mcnemar_p']:.3f} | "
            f"[{s['bootstrap_ci_pt'][0]:+.1f}, {s['bootstrap_ci_pt'][1]:+.1f}] |"
        )
    out.append("")

    # By difficulty
    out += ["## JKB v1 難度別", "", "| model | core | standard | advanced |", "|---|---|---|---|"]
    for label in K1_MODELS:
        row = [label]
        for diff in ("core", "standard", "advanced"):
            stat = summary["jkb"][label]["by_difficulty"].get(diff)
            row.append(
                f"{stat['accuracy'] * 100:.1f}% ({stat['correct']}/{stat['n']})" if stat else "-"
            )
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # By domain
    out += [
        "## JKB v1 分野別",
        "",
        "| model | " + " | ".join(jkb.JKB_DOMAINS) + " |",
        "|---|" + "---|" * len(jkb.JKB_DOMAINS),
    ]
    for label in K1_MODELS:
        row = [label]
        for domain in jkb.JKB_DOMAINS:
            stat = summary["jkb"][label]["by_domain"].get(domain)
            row.append(f"{stat['accuracy'] * 100:.1f}%" if stat else "-")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # Guards
    out += [
        "## ガード指標",
        "",
        "| model | IFEval prompt_strict | IFEval polite | llm-jp-eval AVG "
        "| niilc EM | niilc char_f1 |",
        "|---|---|---|---|---|---|",
    ]
    for label in K1_MODELS:
        ife = summary["ifeval"].get(label, {})
        lje = summary["llm_jp_eval"].get(label, {})
        niilc_em = lje.get("niilc_exact_match")
        row = [
            label,
            _fmt_pct(ife.get("prompt_strict")) if "prompt_strict" in ife else "-",
            _fmt_pct(ife.get("polite_form")) if "polite_form" in ife else "-",
            f"{lje.get('avg'):.3f}" if lje.get("avg") is not None else "-",
            _fmt_pct(niilc_em) if niilc_em is not None else "-",
            f"{lje.get('niilc_char_f1'):.3f}" if lje.get("niilc_char_f1") is not None else "-",
        ]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    return "\n".join(out) + "\n"


def _emit_worst_cells(
    summary: dict[str, Any], model_label: str, k: int = 10
) -> list[dict[str, Any]]:
    """Sort (domain, difficulty) cells by accuracy asc and return the worst k."""
    cells = summary["jkb"][model_label]["by_cell"]
    ranked = sorted(cells.items(), key=lambda kv: (kv[1]["accuracy"], -kv[1]["n"]))
    out: list[dict[str, Any]] = []
    for key, stat in ranked[:k]:
        domain, difficulty = key.split("::", 1)
        out.append(
            {
                "domain": domain,
                "difficulty": difficulty,
                "n": stat["n"],
                "correct": stat["correct"],
                "accuracy": stat["accuracy"],
            }
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="K1 資産再評価集計 (Issue #122)")
    parser.add_argument(
        "--jkb-root",
        default="outputs/eval/jkb/k1-full",
        help="JKB per-model 結果のルート(<label>/scored.jsonl)",
    )
    parser.add_argument(
        "--ifeval-root", default="outputs/eval/ifeval_ja", help="IFEval per-model 結果のルート"
    )
    parser.add_argument(
        "--llmjp-base", type=Path, required=True, help="llm-jp-eval base 結果 JSON パス"
    )
    parser.add_argument("--llmjp-cptb", type=Path, required=True)
    parser.add_argument("--llmjp-dpo", type=Path, required=True)
    parser.add_argument("--llmjp-sft005", type=Path, required=True)
    parser.add_argument("--out-json", default="outputs/eval/jkb/k1-full/k1_summary.json")
    parser.add_argument("--out-md", default="outputs/eval/jkb/k1-full/k1_summary.md")
    args = parser.parse_args()

    llmjp_paths = {
        "base": args.llmjp_base,
        "cptB-final": args.llmjp_cptb,
        "dpo-001-b005": args.llmjp_dpo,
        "sft005-distill": args.llmjp_sft005,
    }
    summary = _load_all(Path(args.jkb_root), Path(args.ifeval_root), llmjp_paths)

    # ワースト 10 セル (base)
    summary["worst_cells_base"] = _emit_worst_cells(summary, "base", 10)

    Path(args.out_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.out_md).write_text(render_summary_markdown(summary), encoding="utf-8")
    print(f"K1 summary written: {args.out_json} / {args.out_md}")


if __name__ == "__main__":
    main()
