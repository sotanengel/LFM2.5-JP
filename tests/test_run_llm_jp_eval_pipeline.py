"""Real llm-jp-eval v2 pipeline wiring for run_llm_jp_eval (Issue #66).

The previous implementation assumed a nonexistent single-call ``llm-jp-eval``
CLI. The actual workflow (as run in WSL against ``~/llm-jp-eval`` +
``~/llm-jp-eval-inference``) is: preprocess -> dump prompts -> run inference
via the transformers module in llm-jp-eval-inference -> evaluate the
generated outputs. These tests check the commands/configs built for that
real pipeline, independent of whether the WSL harness directories exist on
the machine running the tests (they won't on the Windows CI/dev box; the
constructed values must still be correct).

``build_inference_config``/``build_inference_command`` went through two
iterations against the real WSL harness (see git history / PR description):

1. First attempt used dotted CLI overrides for everything
   (``--generation_config.do_sample=false``), which failed instantly with
   "unrecognized arguments" -- ``generation_config`` is a
   ``transformers.GenerationConfig``, not a plain pydantic model, so
   (unlike ``model``/``tokenizer``) the CLI parser doesn't expose dotted
   sub-field overrides for it.
2. Passing it as a single ``--generation_config='{"do_sample": false}'`` JSON
   string also failed: pydantic validated the raw string against
   ``GenerationConfig`` and rejected it (the field's ``field_validator``
   only coerces from a ``dict``, not a JSON string, over the CLI).
3. The working approach -- confirmed against the real harness with
   ``inference.py inference --config <path> --dry_run`` (exit 0) -- is what
   the original Issue #14 baseline runs already did: write a full YAML
   config (nested ``generation_config:`` mapping, parsed as a native dict)
   and pass only ``--config <path>``.
"""

from __future__ import annotations

from pathlib import Path

from lfm25_ja.eval.run_llm_jp_eval import (
    build_dump_command,
    build_eval_plan,
    build_evaluate_command,
    build_inference_command,
    build_inference_config,
    load_eval_config,
    resolve_harness_paths,
)


def _cfg():
    root = Path(__file__).resolve().parents[1]
    return load_eval_config(root / "configs" / "eval" / "llm_jp_eval.yaml")


def test_build_eval_plan_run_names_match_existing_wsl_results():
    # ~/llm-jp-eval/local_files/results already has result_baseline-instruct_*.json
    # and result_baseline-jp202606_*.json from the Issue #14 baseline; the
    # rerun must reuse those same run_name slugs.
    cfg = _cfg()
    plan = build_eval_plan(cfg)
    run_names = {item["name"]: item["run_name"] for item in plan}
    assert run_names["LFM2.5-1.2B-Instruct"] == "baseline-instruct"
    assert run_names["LFM2.5-1.2B-JP-202606"] == "baseline-jp202606"


def test_resolve_harness_paths_expands_user_and_names():
    cfg = _cfg()
    harness_dir, inference_dir = resolve_harness_paths(cfg)
    assert not str(harness_dir).startswith("~")
    assert not str(inference_dir).startswith("~")
    assert harness_dir.name == "llm-jp-eval"
    assert inference_dir.name == "llm-jp-eval-inference"


def test_build_dump_command_calls_evaluate_llm_dump_with_no_sync():
    cfg = _cfg()
    harness_dir, _ = resolve_harness_paths(cfg)
    cmd = build_dump_command(harness_dir, "configs/lfm25_config.yaml")
    assert cmd[:3] == ["uv", "run", "--no-sync"]
    assert "python" in cmd
    assert "scripts/evaluate_llm.py" in cmd
    assert "dump" in cmd
    assert "--config=configs/lfm25_config.yaml" in cmd or "configs/lfm25_config.yaml" in cmd


def test_build_inference_config_is_a_full_yaml_serializable_document():
    cfg = _cfg()
    plan = build_eval_plan(cfg)
    item = plan[0]
    prompt_glob = "local_files/datasets/2.1.5/evaluation/test/prompts_abc/*.eval-prompt.json"
    inference_cfg = build_inference_config(item, cfg, prompt_glob)

    assert inference_cfg["run_name"] == item["run_name"]
    assert inference_cfg["model"]["pretrained_model_name_or_path"] == item["hf_path"]
    assert inference_cfg["tokenizer"]["pretrained_model_name_or_path"] == item["hf_path"]
    assert inference_cfg["tokenizer"]["model_max_length"] == 4096
    assert inference_cfg["prompt_json_path"] == prompt_glob
    assert inference_cfg["pipeline_kwargs"]["batch_size"] == 4
    # generation_config must be a nested mapping (parsed as dict -> GenerationConfig
    # by the field's pydantic validator), never a JSON string or dotted flags.
    assert inference_cfg["generation_config"]["do_sample"] is False


def test_build_inference_config_keeps_base_style_no_chat_template():
    # Phase 0 baseline froze on base-model-style prompting (no chat
    # template) for a controlled comparison; the fix must not silently
    # switch that.
    cfg = _cfg()
    plan = build_eval_plan(cfg)
    inference_cfg = build_inference_config(plan[0], cfg, "prompts/*.eval-prompt.json")
    assert inference_cfg["apply_chat_template"] is False


def test_build_inference_command_only_passes_config_path():
    # Regression: no dotted --generation_config.* or JSON-string
    # --generation_config=... flags -- both fail against the real CLI parser
    # (see module docstring). The only way to set generation_config is via
    # the YAML file passed through --config.
    cmd = build_inference_command("baseline-instruct_generated.yaml")
    assert cmd[:3] == ["uv", "run", "--no-sync"]
    assert "inference.py" in cmd
    assert "inference" in cmd
    assert "--config=baseline-instruct_generated.yaml" in cmd
    assert not any(arg.startswith("--generation_config") for arg in cmd)
    assert not any(arg.startswith("--model.") for arg in cmd)


def test_build_evaluate_command_points_at_inference_result_dir():
    cfg = _cfg()
    harness_dir, _ = resolve_harness_paths(cfg)
    result_dir = (
        "/home/usr/llm-jp-eval-inference/inference-modules/transformers/outputs/baseline-instruct"
    )
    cmd = build_evaluate_command(harness_dir, "configs/lfm25_config.yaml", result_dir)
    joined = " ".join(cmd)
    assert "scripts/evaluate_llm.py" in joined
    assert "eval" in cmd
    assert f"--inference_result_dir={result_dir}" in joined
