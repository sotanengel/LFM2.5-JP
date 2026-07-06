"""llm-jp-eval baseline evaluation wrapper."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from lfm25_ja.utils.config import load_config, merge_configs, project_root


def load_eval_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load evaluation config merged with base.yaml."""
    root = project_root()
    base = load_config(root / "configs" / "base.yaml")
    eval_path = Path(path) if path else root / "configs" / "eval" / "llm_jp_eval.yaml"
    eval_cfg = load_config(eval_path)
    return merge_configs(base, eval_cfg)


def build_eval_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-model evaluation plan from config."""
    eval_cfg = cfg["eval"]
    tasks = eval_cfg.get("tasks", [])
    plan: list[dict[str, Any]] = []
    for model in eval_cfg.get("models", []):
        plan.append(
            {
                "name": model["name"],
                "hf_path": model["hf_path"],
                "tasks": list(tasks),
                "output_dir": str(Path(eval_cfg.get("output_dir", "outputs/eval")) / model["name"]),
                "num_few_shots": int(eval_cfg.get("num_few_shots", 3)),
            }
        )
    return plan


def build_llm_jp_eval_command(plan_item: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Build llm-jp-eval CLI command (if installed)."""
    gen = cfg["eval"].get("generation", {})
    return [
        "llm-jp-eval",
        "--model",
        plan_item["hf_path"],
        "--tasks",
        ",".join(plan_item["tasks"]),
        "--output-dir",
        plan_item["output_dir"],
        "--max-new-tokens",
        str(gen.get("max_new_tokens", 256)),
    ]


def run_baseline_eval(
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run baseline evaluation or return plan when dry_run=True."""
    cfg = load_eval_config(config_path)
    plan = build_eval_plan(cfg)
    results: dict[str, Any] = {"plan": plan, "runs": []}

    if dry_run or shutil.which("llm-jp-eval") is None:
        results["status"] = "dry_run"
        results["commands"] = [build_llm_jp_eval_command(item, cfg) for item in plan]
        return results

    for item in plan:
        cmd = build_llm_jp_eval_command(item, cfg)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        results["runs"].append(
            {
                "model": item["name"],
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        )
    results["status"] = "executed"
    return results


def write_eval_summary(results: dict[str, Any], output_path: Path | None = None) -> Path:
    """Write evaluation summary JSON for EXPERIMENT_LOG reference."""
    path = output_path or (project_root() / "outputs" / "eval" / "baseline_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run llm-jp-eval baseline")
    parser.add_argument("--config", default=None, help="Eval config YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()
    results = run_baseline_eval(config_path=args.config, dry_run=args.dry_run)
    out = write_eval_summary(results)
    print(f"Eval summary written to {out}")
    if results.get("status") == "dry_run":
        print("llm-jp-eval not found or dry-run requested. Commands:")
        for cmd in results.get("commands", []):
            print(" ", " ".join(cmd))


if __name__ == "__main__":
    main()
