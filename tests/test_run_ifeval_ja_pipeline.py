"""ifeval_ja pipeline tests: config load, dry-run plan, scoring end-to-end
(Issue #104). CPU-only -- generate_ifeval_ja never imports torch/transformers
at module scope, so --dry-run and plan-building work without a GPU stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lfm25_ja.eval import run_ifeval_ja
from lfm25_ja.eval.generate_ifeval_ja import (
    build_generation_plan,
    load_ifeval_config,
    run_generation,
)
from lfm25_ja.eval.score_ifeval_ja import score_model

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "ifeval_ja.yaml"
EXPECTED_MODEL_NAMES = [
    "base-jp202606",
    "sft004-L9-lr1e-5",
    "sft004-L9-lr3e-5",
    "sft004-L6-9-lr1e-5",
    "sft003-L9",
    "sft002-mix",  # Issue #105: sft-002 mix arm appended to the frozen Issue #104 list
    "sft005-distill",  # Issue #109: sft-005 distill arm appended, list otherwise untouched
    "dpo-001-b005",  # Issue #115/#117: dpo-001 beta sweep arms appended
    "dpo-001-b01",
    "dpo-001-b03",
]


def _cfg():
    return load_ifeval_config(CONFIG_PATH)


def test_config_loads_and_validates():
    cfg = _cfg()
    eval_cfg = cfg["eval"]
    assert eval_cfg["harness"] == "ifeval_ja"
    assert eval_cfg["apply_chat_template"] is True
    assert eval_cfg["dataset_path"] == "datasets/eval/ifeval_ja/prompts.jsonl"
    assert [m["name"] for m in eval_cfg["models"]] == EXPECTED_MODEL_NAMES
    for m in eval_cfg["models"]:
        assert m["hf_path"]
    gen = eval_cfg["generation"]
    assert gen["max_new_tokens"] == 512
    assert gen["temperature"] == 0.0
    assert gen["do_sample"] is False


FIXTURE_DATASET_PATH = REPO_ROOT / "tests" / "fixtures" / "ifeval_ja_sample.jsonl"


def _cfg_with_fixture_dataset():
    # configs/eval/ifeval_ja.yaml's real dataset_path
    # (datasets/eval/ifeval_ja/prompts.jsonl) is intentionally not yet added
    # to the repo (Issue #104 scope split); point at the 5-row sample fixture
    # instead so prompt-count assertions have something real to load.
    cfg = _cfg()
    cfg["eval"]["dataset_path"] = str(FIXTURE_DATASET_PATH)
    return cfg


def test_build_generation_plan_covers_all_five_models():
    cfg = _cfg_with_fixture_dataset()
    plan = build_generation_plan(cfg)
    names = [item["name"] for item in plan]
    assert names == EXPECTED_MODEL_NAMES
    for item in plan:
        assert item["num_prompts"] == 5  # tests/fixtures/ifeval_ja_sample.jsonl has 5 rows
        assert item["output_path"].endswith("generations.jsonl")


def test_build_generation_plan_respects_models_filter_and_resume_from():
    cfg = _cfg_with_fixture_dataset()
    plan = build_generation_plan(cfg, models=["sft004-L9-lr1e-5", "sft003-L9"])
    assert [item["name"] for item in plan] == ["sft004-L9-lr1e-5", "sft003-L9"]

    plan = build_generation_plan(cfg, resume_from="sft004-L9-lr3e-5")
    assert [item["name"] for item in plan] == [
        "sft004-L9-lr3e-5",
        "sft004-L6-9-lr1e-5",
        "sft003-L9",
        "sft002-mix",
        "sft005-distill",
        "dpo-001-b005",
        "dpo-001-b01",
        "dpo-001-b03",
    ]


def test_build_generation_plan_respects_limit():
    cfg = _cfg_with_fixture_dataset()
    plan = build_generation_plan(cfg, limit=2)
    assert all(item["num_prompts"] == 2 for item in plan)


def test_build_generation_plan_uses_real_dataset_when_present():
    # datasets/eval/ifeval_ja/prompts.jsonl (Fable5's 100-prompt dataset,
    # Issue #104) is present in this checkout; the plan should report the
    # real prompt count.
    cfg = _cfg()
    dataset_path = Path(cfg["eval"]["dataset_path"])
    if not (REPO_ROOT / dataset_path).exists():
        pytest.skip("real dataset not present in this checkout")
    plan = build_generation_plan(cfg)
    assert [item["name"] for item in plan] == EXPECTED_MODEL_NAMES
    assert all(item["num_prompts"] == 100 for item in plan)


def test_build_generation_plan_tolerates_missing_dataset():
    # --dry-run must not hard-fail when the dataset file is absent (e.g. a
    # checkout that predates the dataset PR, or a bogus path); the plan
    # should still build with num_prompts=None rather than raising.
    cfg = _cfg()
    cfg["eval"]["dataset_path"] = "datasets/eval/ifeval_ja/does_not_exist.jsonl"
    plan = build_generation_plan(cfg)
    assert [item["name"] for item in plan] == EXPECTED_MODEL_NAMES
    assert all(item["num_prompts"] is None for item in plan)


def test_run_generation_dry_run_does_not_import_torch():
    torch_was_loaded_before = "torch" in sys.modules

    results = run_generation(config_path=CONFIG_PATH, dry_run=True)

    assert results["status"] == "dry_run"
    names = [item["name"] for item in results["plan"]]
    assert names == EXPECTED_MODEL_NAMES
    # dry-run must never trigger the lazy `import torch` inside generate_for_model.
    if not torch_was_loaded_before:
        assert "torch" not in sys.modules


def test_generate_dry_run_cli_prints_all_five_model_names(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["run_ifeval_ja", "generate", "--config", str(CONFIG_PATH), "--dry-run"]
    )
    run_ifeval_ja.main()
    out = capsys.readouterr().out
    assert "status=dry_run" in out
    for name in EXPECTED_MODEL_NAMES:
        assert name in out


# --- scoring end-to-end against a crafted fixture ---


def _write_fake_generations(path: Path) -> None:
    rows = [
        {
            "id": "ifja-101",
            "prompt": "短く挨拶してください。",
            "response": "こんにちは",
            "instruction_id_list": ["char_count"],
            "kwargs": {"char_count": {"max": 10}},
            "category": "依頼",
        },
        {
            "id": "ifja-102",
            "prompt": "5文字以内で挨拶してください。",
            "response": "これは10文字をゆうに超える長い文章です",
            "instruction_id_list": ["char_count"],
            "kwargs": {"char_count": {"max": 5}},
            "category": "依頼",
        },
        {
            "id": "ifja-103",
            "prompt": "敬体で一言述べてください。",
            "response": "これはペンだ。",
            "instruction_id_list": ["polite_form"],
            "kwargs": {"polite_form": {"style": "polite"}},
            "category": "敬語",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_score_model_end_to_end(tmp_path):
    output_dir = tmp_path / "outputs"
    _write_fake_generations(output_dir / "testmodel" / "generations.jsonl")
    cfg = {"eval": {"output_dir": str(output_dir)}}

    aggregate = score_model("testmodel", cfg)

    assert aggregate["num_prompts"] == 3
    assert aggregate["num_instructions"] == 3
    assert aggregate["prompt_strict_acc"] == pytest.approx(1 / 3)
    assert aggregate["instruction_strict_acc"] == pytest.approx(1 / 3)
    assert aggregate["by_verifier"]["char_count"]["count"] == 2
    assert aggregate["by_verifier"]["polite_form"]["count"] == 1
    assert aggregate["by_category"]["依頼"]["count"] == 2
    assert aggregate["by_category"]["敬語"]["count"] == 1

    scores_path = output_dir / "testmodel" / "scores.jsonl"
    aggregate_path = output_dir / "testmodel" / "aggregate.json"
    assert scores_path.exists()
    assert aggregate_path.exists()

    scores = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    by_id = {s["id"]: s for s in scores}
    assert by_id["ifja-101"]["strict_pass"] is True
    assert by_id["ifja-102"]["strict_pass"] is False
    assert by_id["ifja-103"]["strict_pass"] is False


def test_score_model_raises_when_generations_missing(tmp_path):
    cfg = {"eval": {"output_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError):
        score_model("nonexistent-model", cfg)
