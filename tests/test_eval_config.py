"""Baseline evaluation config tests."""

from pathlib import Path

from lfm25_ja.eval.run_llm_jp_eval import build_eval_plan, load_eval_config


def test_load_eval_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_eval_config(root / "configs" / "eval" / "llm_jp_eval.yaml")
    assert cfg["eval"]["harness"] == "llm-jp-eval"
    assert len(cfg["eval"]["models"]) == 2


def test_build_eval_plan() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_eval_config(root / "configs" / "eval" / "llm_jp_eval.yaml")
    plan = build_eval_plan(cfg)
    assert len(plan) == 2
    assert plan[0]["hf_path"] == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert "jmmlu" in plan[0]["tasks"]
