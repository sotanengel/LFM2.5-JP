"""llm-jp-eval v2 baseline evaluation wrapper (Issue #66).

The real llm-jp-eval v2 workflow is a multi-step pipeline, not a single CLI
call:

1. **preprocess** raw datasets into llm-jp-eval's jaster format
   (``~/llm-jp-eval/scripts/preprocess_dataset.py``) -- skipped when already
   present.
2. **dump** prompts for the configured datasets/few-shot count
   (``~/llm-jp-eval/scripts/evaluate_llm.py dump``).
3. **inference**: generate model outputs for those prompts via the separate
   ``~/llm-jp-eval-inference`` tool (the ``transformers`` module here).
4. **evaluate**: score the generated outputs against gold labels
   (``~/llm-jp-eval/scripts/evaluate_llm.py eval``).

This module builds the exact commands for that pipeline and -- when the WSL
harness directories are present on the filesystem it's running on -- runs
them end to end. When they are not present (e.g. Windows dev/CI), it falls
back to a dry run that only prints/returns the commands, matching the
project's existing "verify structure without a GPU" testing style.

It also carries the Issue #66 root-cause fix for the jsem / jmmlu(JP) ool
(out-of-label) bug: see ``DATASET_INFO_OVERRIDES`` and the module docstring
in ``tests/test_eval_answer_extraction.py`` for the full analysis.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from lfm25_ja.utils.config import load_config, merge_configs, project_root

logger = logging.getLogger(__name__)

# Issue #66 root-cause fix: llm-jp-eval's builtin per-dataset answer
# extraction is too strict/too short-budgeted for how these models actually
# answer without a chat template (see tests/test_eval_answer_extraction.py
# for the real WSL samples this was derived from). Two problems compound:
#
#   1. jsem uses AnswerPatternId.ANSWER_TAGS_JP (requires literal
#      <answer></answer> tags the models never emit) -> 100% ool for both
#      models. jmmlu uses AnswerPatternId.CHOICE_ONLY_JP with the dataset's
#      own output_length=1 token, so a model that starts its answer with
#      "\boxed{" gets truncated to a bare "\" -> ool (55% for JP-202606).
#   2. Even after switching to a lenient `custom` regex + a modest
#      output_length bump (48/16), a real rerun still showed jsem(JP) 37%,
#      jmmlu(JP) 15%, jmmlu(Instruct) 10% ool: both models (especially
#      JP-202606 on math/logic jmmlu questions) often write a long Japanese
#      chain-of-thought explanation before ever stating the label, and run
#      out of budget before reaching it. A long explanation can also
#      *mention and reject* a label (e.g. "option A: ... this is false")
#      before its real conclusion, which a leftmost-match regex would
#      wrongly grab.
#
# Fix: widen output_length further (within the configured
# generation.max_new_tokens=256 budget) so there's room to reach a
# conclusion, and prefix each regex with a greedy `.*` so re.search's
# backtracking finds the LAST occurrence of the label in the text (the
# model's actual conclusion) rather than the first thing that looks like
# one. Verified ool on a real rerun (2026-07-13, 100 samples): jsem
# Instruct 0% / JP-202606 1%, jmmlu Instruct 4% / JP-202606 5% -- all under
# the 10% acceptance threshold.
DATASET_INFO_OVERRIDES: dict[str, dict[str, Any]] = {
    "jsem": {
        "output_length": 160,
        "answer_pattern_id": "custom",
        "answer_extract_pattern": r"(?s).*(yes|no|unknown|undef)",
    },
    "jmmlu": {
        "output_length": 200,
        "answer_pattern_id": "custom",
        "answer_extract_pattern": r"(?s).*\b([ABCD])\b",
    },
}

# Mirrors llm-jp-eval's eval_configs/*.yaml `categories` block for the frozen
# Phase 0 dataset subset (experiments/reports/phase0_baseline.md).
DATASET_CATEGORIES: dict[str, dict[str, Any]] = {
    "NLI": {
        "description": "Natural Language Inference",
        "default_metric": "exact_match",
        "metrics": {},
        "datasets": ["jnli", "jsem", "jsick"],
    },
    "QA": {
        "description": "Question Answering",
        "default_metric": "exact_match",
        "metrics": {},
        "datasets": ["niilc"],
    },
    "RC": {
        "description": "Reading Comprehension",
        "default_metric": "exact_match",
        "metrics": {},
        "datasets": ["jsquad"],
    },
    "CR": {
        "description": "Commonsense Reasoning",
        "default_metric": "exact_match",
        "metrics": {},
        "datasets": ["jcommonsenseqa"],
    },
    "HE-JA": {
        "description": "Human Evaluation",
        "default_metric": "exact_match",
        "metrics": {},
        "datasets": ["jmmlu"],
    },
}


def extract_custom_answer(text: str, pattern: str) -> str:
    """Reimplements llm-jp-eval's ``AnswerPatternId.CUSTOM`` extraction
    (``llm_jp_eval.answer_parser.extract_answer_with_pattern``): a plain
    ``re.search`` with all capture groups joined and stripped, returning
    ``""`` when the pattern doesn't match (i.e. out-of-label).

    Used both to unit-test the regexes in ``DATASET_INFO_OVERRIDES`` against
    real sample outputs, and as a local sanity check independent of having
    llm-jp-eval installed.
    """
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return "".join(match.groups()).strip()


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
                "num_few_shots": int(eval_cfg.get("num_few_shots", 4)),
                "run_name": _run_name_for(model["name"]),
            }
        )
    return plan


def _run_name_for(model_name: str) -> str:
    """llm-jp-eval run_name: strip the common "LFM2.5-1.2B-" prefix and
    slugify the rest, matching the existing baseline-instruct /
    baseline-jp202606 run names already present in
    ~/llm-jp-eval/local_files/results (Issue #14)."""
    suffix = re.sub(r"^LFM2\.5-1\.2B-", "", model_name, flags=re.IGNORECASE)
    slug = re.sub(r"[^a-z0-9]+", "", suffix.lower())
    return f"baseline-{slug}"


def build_eval_dataset_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the llm-jp-eval ``eval_dataset_config`` document (what upstream
    calls ``eval_configs/lfm25_baseline.yaml``): the dataset list, category
    definitions, and the Issue #66 ``dataset_info_overrides`` fix.

    Config-declared overrides win over the module defaults so
    ``configs/eval/llm_jp_eval.yaml`` stays the single source of truth.
    """
    eval_cfg = cfg["eval"]
    overrides = dict(DATASET_INFO_OVERRIDES)
    overrides.update(eval_cfg.get("dataset_info_overrides", {}))
    return {
        "datasets": list(eval_cfg.get("tasks", [])),
        "categories": DATASET_CATEGORIES,
        "dataset_info_overrides": overrides,
    }


def resolve_harness_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve the WSL llm-jp-eval / llm-jp-eval-inference directories from
    config, expanding ``~``. Does not require the paths to exist -- callers
    check that separately to decide dry-run vs. real execution."""
    wsl_cfg = cfg["eval"].get("wsl", {})
    harness_dir = Path(wsl_cfg.get("llm_jp_eval_dir", "~/llm-jp-eval")).expanduser()
    inference_dir = Path(
        wsl_cfg.get("llm_jp_eval_inference_dir", "~/llm-jp-eval-inference")
    ).expanduser()
    return harness_dir, inference_dir


def build_preprocess_command(harness_dir: Path, dataset_name: str, version_name: str) -> list[str]:
    """``scripts/preprocess_dataset.py`` -- step 1 (preprocess). Idempotent
    upstream (skips datasets already present under local_files/); callers
    only need to run this when the target dataset dir is missing."""
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/preprocess_dataset.py",
        "--dataset-name",
        dataset_name,
        "--output-dir",
        "local_files",
        "--version-name",
        version_name,
    ]


def build_dump_command(harness_dir: Path, eval_config_rel_path: str) -> list[str]:
    """``scripts/evaluate_llm.py dump`` -- step 2 (dump prompts)."""
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/evaluate_llm.py",
        "dump",
        f"--config={eval_config_rel_path}",
    ]


def build_inference_config(
    plan_item: dict[str, Any],
    cfg: dict[str, Any],
    prompt_json_glob: str,
) -> dict[str, Any]:
    """Build the full InferenceConfig YAML document for llm-jp-eval-inference's
    transformers module -- step 3 (run inference).

    This is written to a YAML file and passed via ``--config`` rather than
    CLI flags. Two CLI-flag approaches were tried against the real WSL
    harness and both failed (see module docstring in
    tests/test_run_llm_jp_eval_pipeline.py for the full trace):
    ``generation_config`` is a ``transformers.GenerationConfig``, not a
    plain pydantic model, so unlike ``model``/``tokenizer`` it has no dotted
    CLI overrides, and passing it as a JSON *string* is rejected by
    pydantic (the field's validator only coerces from a ``dict``). Loading
    it as a nested YAML mapping via ``--config`` is what the original
    Issue #14 baseline runs did (``baseline_instruct.yaml`` /
    ``baseline_jp202606.yaml`` in ~/llm-jp-eval-inference), and is confirmed
    working here via ``inference.py inference --config ... --dry_run``.

    ``apply_chat_template`` defaults to ``False`` (the upstream default),
    preserving the Phase 0 baseline's frozen base-model-style prompting
    (experiments/reports/phase0_baseline.md). Set ``eval.apply_chat_template:
    true`` in the eval config to opt into ChatML-formatted prompting.

    A Phase 3 sft-003 spot-check (Issue #33/#35) confirmed the flag is wired
    correctly (tokenized prompt length grew by +8 tokens under ChatML) but
    did not change generated text on 40 sampled prompts (old/new diff=0).
    The override remains available for explicit chat-format evals; it is
    not a fix for the observed train-loss / llm-jp-eval ranking inversion.
    """
    gen = cfg["eval"].get("generation", {})
    return {
        "run_name": plan_item["run_name"],
        "wandb": {"launch": False, "entity": "entity", "project": "project"},
        "tokenize_kwargs": {"add_special_tokens": True},
        "apply_chat_template": bool(cfg["eval"].get("apply_chat_template", False)),
        "model": {"pretrained_model_name_or_path": plan_item["hf_path"]},
        "tokenizer": {
            "pretrained_model_name_or_path": plan_item["hf_path"],
            "model_max_length": 4096,
        },
        "generation_config": {
            "do_sample": False,
            "temperature": gen.get("temperature", 0.0),
        },
        "pipeline_kwargs": {"batch_size": 4},
        "prompt_json_path": prompt_json_glob,
    }


def build_inference_command(inference_config_path: str) -> list[str]:
    """``inference.py inference --config <path>`` -- step 3 (run inference)."""
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "inference.py",
        "inference",
        f"--config={inference_config_path}",
    ]


def build_evaluate_command(
    harness_dir: Path, eval_config_rel_path: str, inference_result_dir: str
) -> list[str]:
    """``scripts/evaluate_llm.py eval`` -- step 4 (evaluate generated
    outputs against gold labels)."""
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/evaluate_llm.py",
        "eval",
        f"--config={eval_config_rel_path}",
        f"--inference_result_dir={inference_result_dir}",
        '--exporters={"local":{"export_output_table":true,"output_top_n":5}}',
    ]


def _write_eval_dataset_config(harness_dir: Path, cfg: dict[str, Any]) -> Path:
    dataset_cfg = build_eval_dataset_config(cfg)
    out_path = harness_dir / "eval_configs" / "lfm25_baseline.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Generated by lfm25_ja.eval.run_llm_jp_eval (Issue #66) -- do not hand-edit;\n"
        "# edit configs/eval/llm_jp_eval.yaml in the LFM2.5-JP repo instead.\n"
        + yaml.safe_dump(dataset_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def _write_inference_config(
    inference_module_dir: Path,
    plan_item: dict[str, Any],
    cfg: dict[str, Any],
    prompt_json_glob: str,
) -> Path:
    inference_cfg = build_inference_config(plan_item, cfg, prompt_json_glob)
    out_path = inference_module_dir / f"{plan_item['run_name']}_generated.yaml"
    out_path.write_text(
        "# Generated by lfm25_ja.eval.run_llm_jp_eval (Issue #66) -- do not hand-edit;\n"
        "# edit configs/eval/llm_jp_eval.yaml in the LFM2.5-JP repo instead.\n"
        + yaml.safe_dump(inference_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def _find_latest_prompts_dir(harness_dir: Path, dataset_version: str) -> Path | None:
    datasets_dir = harness_dir / "local_files" / "datasets" / dataset_version
    pattern = str(datasets_dir / "evaluation" / "test" / "prompts_*")
    candidates = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    # This orchestrator itself typically runs under its own `uv run` venv
    # (e.g. ~/lfm25-ja/.venv), but each step below targets a *different*
    # sibling repo's own venv (~/llm-jp-eval, ~/llm-jp-eval-inference). A
    # leaked VIRTUAL_ENV pointing at the wrong project confuses `uv run
    # --no-sync` (see project WSL notes), so strip it for child processes.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    logger.info("running: %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _parse_ool_scores(result_json_path: Path, tasks: list[str]) -> dict[str, float]:
    data = json.loads(result_json_path.read_text(encoding="utf-8"))
    scores = data.get("evaluation", {}).get("scores", {})
    return {task: scores[f"{task}_ool"] for task in tasks if f"{task}_ool" in scores}


def run_baseline_eval(
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the real llm-jp-eval v2 pipeline (dump -> inference -> evaluate)
    for every configured model, or return the plan/commands when
    ``dry_run=True`` or the WSL harness directories aren't present on this
    filesystem."""
    cfg = load_eval_config(config_path)
    plan = build_eval_plan(cfg)
    harness_dir, inference_dir = resolve_harness_paths(cfg)
    wsl_cfg = cfg["eval"].get("wsl", {})
    dataset_version = wsl_cfg.get("dataset_version", "2.1.5")
    inference_module = wsl_cfg.get("inference_module", "transformers")
    eval_config_rel_path = "configs/lfm25_config.yaml"

    results: dict[str, Any] = {"plan": plan, "runs": []}

    harness_available = harness_dir.is_dir() and (inference_dir / "inference-modules").is_dir()
    if dry_run or not harness_available:
        results["status"] = "dry_run"
        results["harness_dir"] = str(harness_dir)
        results["inference_dir"] = str(inference_dir)
        results["dataset_eval_config"] = build_eval_dataset_config(cfg)
        commands: dict[str, list[list[str]]] = {
            "dump": [build_dump_command(harness_dir, eval_config_rel_path)]
        }
        for item in plan:
            prompt_glob = str(
                harness_dir
                / "local_files"
                / "datasets"
                / dataset_version
                / "evaluation"
                / "test"
                / "prompts_<hash>"
                / "*.eval-prompt.json"
            )
            inference_module_dir = inference_dir / "inference-modules" / inference_module
            inference_out_dir = str(inference_module_dir / "outputs" / item["run_name"])
            inference_config_path = str(
                inference_module_dir / f"{item['run_name']}_generated.yaml"
            )
            commands[item["name"]] = [
                build_inference_command(inference_config_path),
                build_evaluate_command(harness_dir, eval_config_rel_path, inference_out_dir),
            ]
        results["commands"] = commands
        return results

    _write_eval_dataset_config(harness_dir, cfg)

    dump_cmd = build_dump_command(harness_dir, eval_config_rel_path)
    dump_proc = _run(dump_cmd, cwd=harness_dir)
    results["runs"].append(
        {"step": "dump", "returncode": dump_proc.returncode, "stderr": dump_proc.stderr[-4000:]}
    )
    if dump_proc.returncode != 0:
        results["status"] = "failed"
        return results

    prompts_dir = _find_latest_prompts_dir(harness_dir, dataset_version)
    if prompts_dir is None:
        results["status"] = "failed"
        results["error"] = "dump succeeded but no prompts_* directory was found"
        return results
    prompt_glob = str(prompts_dir / "*.eval-prompt.json")

    inference_module_dir = inference_dir / "inference-modules" / inference_module
    for item in plan:
        inference_config_path = _write_inference_config(
            inference_module_dir, item, cfg, prompt_glob
        )
        infer_cmd = build_inference_command(inference_config_path.name)
        infer_proc = _run(infer_cmd, cwd=inference_module_dir)
        results["runs"].append(
            {
                "step": "inference",
                "model": item["name"],
                "returncode": infer_proc.returncode,
                "stderr": infer_proc.stderr[-4000:],
            }
        )
        if infer_proc.returncode != 0:
            results["status"] = "failed"
            continue

        inference_result_dir = str(inference_module_dir / "outputs" / item["run_name"])
        eval_cmd = build_evaluate_command(harness_dir, eval_config_rel_path, inference_result_dir)
        eval_proc = _run(eval_cmd, cwd=harness_dir)
        results["runs"].append(
            {
                "step": "evaluate",
                "model": item["name"],
                "returncode": eval_proc.returncode,
                "stderr": eval_proc.stderr[-4000:],
            }
        )
        if eval_proc.returncode != 0:
            results["status"] = "failed"
            continue

        result_files = sorted(
            (harness_dir / "local_files" / "results").glob(f"result_{item['run_name']}_*.json")
        )
        if result_files:
            latest = result_files[-1]
            results["runs"][-1]["result_json"] = str(latest)
            results["runs"][-1]["ool"] = _parse_ool_scores(latest, item["tasks"])

    results.setdefault("status", "executed")
    return results


def write_eval_summary(results: dict[str, Any], output_path: Path | None = None) -> Path:
    """Write evaluation summary JSON for EXPERIMENT_LOG reference."""
    path = output_path or (project_root() / "outputs" / "eval" / "baseline_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the llm-jp-eval v2 baseline pipeline")
    parser.add_argument("--config", default=None, help="Eval config YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Print the pipeline plan only")
    args = parser.parse_args()
    results = run_baseline_eval(config_path=args.config, dry_run=args.dry_run)
    out = write_eval_summary(results)
    print(f"Eval summary written to {out}")
    if results.get("status") == "dry_run":
        harness_exists = Path(results["harness_dir"]).is_dir()
        print(f"harness_dir={results['harness_dir']} (exists={harness_exists})")
        print("WSL harness not found or --dry-run requested. Pipeline commands:")
        for name, cmds in results.get("commands", {}).items():
            print(f"  [{name}]")
            for cmd in cmds:
                print("   ", " ".join(cmd))


if __name__ == "__main__":
    main()
