"""Baseline evaluation config tests."""

from pathlib import Path

from lfm25_ja.eval.run_llm_jp_eval import build_eval_plan, load_eval_config

_ROOT = Path(__file__).resolve().parents[1]
_BASELINE = _ROOT / "configs" / "eval" / "llm_jp_eval.yaml"
_SFT004 = _ROOT / "configs" / "eval" / "llm_jp_eval_sft004.yaml"

_SFT004_MODEL_NAMES = [
    "sft004-L9-lr1e-5",
    "sft004-L9-lr3e-5",
    "sft004-L9-lr1e-4",
    "sft004-L6-9-lr1e-5",
    "sft004-L6-9-lr3e-5",
    "sft004-L6-9-lr1e-4",
    "base-jp202606",
]


def test_load_eval_config() -> None:
    cfg = load_eval_config(_BASELINE)
    assert cfg["eval"]["harness"] == "llm-jp-eval"
    assert len(cfg["eval"]["models"]) == 2


def test_build_eval_plan() -> None:
    cfg = load_eval_config(_BASELINE)
    plan = build_eval_plan(cfg)
    assert len(plan) == 2
    assert plan[0]["hf_path"] == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert "jmmlu" in plan[0]["tasks"]


def test_load_sft004_eval_config() -> None:
    """Issue #36: sft-004 評価はモデル一覧以外を baseline 凍結条件と揃える。"""
    cfg = load_eval_config(_SFT004)
    ev = cfg["eval"]
    assert ev["harness"] == "llm-jp-eval"
    assert ev["apply_chat_template"] is False
    assert ev["num_few_shots"] == 4
    assert ev["max_num_samples"] == 100
    assert ev["generation"]["temperature"] == 0.0
    assert ev["generation"]["max_new_tokens"] == 256
    assert [m["name"] for m in ev["models"]] == _SFT004_MODEL_NAMES
    assert ev["models"][-1]["hf_path"] == "LiquidAI/LFM2.5-1.2B-JP-202606"


def test_sft004_eval_frozen_fields_match_baseline() -> None:
    base = load_eval_config(_BASELINE)["eval"]
    sft = load_eval_config(_SFT004)["eval"]
    for key in (
        "tasks",
        "few_shot",
        "num_few_shots",
        "max_num_samples",
        "generation",
        "dataset_info_overrides",
    ):
        assert sft[key] == base[key], key
    # baseline YAML にキーが無くても実行時デフォルトは false(PR #88)
    assert sft.get("apply_chat_template", False) == base.get("apply_chat_template", False)


def test_build_sft004_eval_plan() -> None:
    cfg = load_eval_config(_SFT004)
    plan = build_eval_plan(cfg)
    assert len(plan) == 7
    assert [p["name"] for p in plan] == _SFT004_MODEL_NAMES
    assert plan[0]["num_few_shots"] == 4
    assert "jmmlu" in plan[0]["tasks"]
