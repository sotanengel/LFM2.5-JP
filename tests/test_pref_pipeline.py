"""dpo-001 preference-pair pipeline tests (Issue #115).

Covers the five CPU/GPU-split phases described in the issue:
  P0 (CPU) lfm25_ja.data.pref_prompts   -- prompt pool construction
  G  (GPU) lfm25_ja.data.pref_generate  -- base on-policy K-sample generation
  V  (CPU) lfm25_ja.data.pref_verify    -- rule-based verdicts
  J  (GPU) lfm25_ja.eval.judge_swallow  -- LLM pointwise quality judge
  P  (CPU) lfm25_ja.data.pref_pairs     -- chosen/rejected pairing + hard gates

Only the CPU phases are exercised end-to-end; the GPU phases (pref_generate,
judge_swallow) are tested at the pure-function / dry-run level only (prompt
building, JSON parsing, idempotent-target selection) -- no model is loaded,
matching the project convention established by
tests/test_distill_select.py and the ifeval_ja generation harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lfm25_ja.data.pref_generate import _existing_generation_keys, build_generation_plan
from lfm25_ja.data.pref_pairs import (
    build_pair,
    build_preference_pairs,
    merge_samples,
    render_pairs_stats_report,
    select_chosen,
    select_rejected,
)
from lfm25_ja.data.pref_prompts import (
    build_prompt_pool,
    build_source_a_prompts,
    build_source_b_prompts,
    check_prompt_pool_non_duplication,
    render_pool_stats_report,
)
from lfm25_ja.data.pref_verify import run_verification, verify_generations, verify_sample
from lfm25_ja.eval.judge_swallow import (
    build_judge_prompt,
    parse_judge_output,
    select_judge_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV_PATH = REPO_ROOT / "datasets" / "sft" / "sft005_distill_candidateB_prompts.csv"
REAL_EVAL_PROMPTS_PATH = REPO_ROOT / "datasets" / "eval" / "ifeval_ja" / "prompts.jsonl"

EVAL_CHAR_VALUES = {50, 60, 70, 80, 90, 100, 120, 150}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _csv_row(
    category: str,
    detail: dict,
    prompt: str = "質問",
    topic: str = "topic",
    row_id: str = "distill-00001",
    instruction_id_list: str | None = None,
) -> dict:
    """A row shaped like lfm25_ja.data.distill_select.read_distill_csv's output
    (no 'response' -- pref_prompts never reads it)."""
    return {
        "id": row_id,
        "category": category,
        "instruction_id_list": instruction_id_list or category,
        "constraint_detail": json.dumps(detail, ensure_ascii=False),
        "detail": detail,
        "topic": topic,
        "prompt": prompt,
    }


def _eval_prompts(char_count_values: list[int], prompt_texts: list[str]) -> list[dict]:
    prompts = []
    for i, mx in enumerate(char_count_values):
        prompts.append(
            {
                "id": f"ifja-{i}",
                "prompt": f"prompt {i}",
                "instruction_id_list": ["char_count"],
                "kwargs": {"char_count": {"max": mx}},
                "category": "依頼",
            }
        )
    for i, text in enumerate(prompt_texts):
        prompts.append(
            {
                "id": f"ifja-text-{i}",
                "prompt": text,
                "instruction_id_list": [],
                "kwargs": {},
                "category": "依頼",
            }
        )
    return prompts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _pool_row(
    row_id: str,
    category: str,
    detail: dict,
    prompt: str = "質問",
    topic: str = "topic",
) -> dict:
    """A row shaped like pref_prompts' pool output (constraint_detail is a
    nested JSON object, not a string -- JSONL, unlike the CSV, supports this
    directly)."""
    return {
        "id": row_id,
        "category": category,
        "instruction_id_list": category,
        "constraint_detail": detail,
        "topic": topic,
        "prompt": prompt,
    }


def _filler(n: int, offset: int = 0) -> str:
    start = 0x4E00 + offset * 503
    return "".join(chr(start + (i % 4000)) for i in range(n))


# ===========================================================================
# Phase P0: pref_prompts.py -- prompt pool construction
# ===========================================================================


def test_build_source_a_prompts_drops_eval_colliding_char_values() -> None:
    rows = [
        _csv_row("char_count", {"max": 100}, row_id="a"),
        _csv_row("compound", {"max": 90, "include": ["x"]}, row_id="b"),
        _csv_row("char_count", {"max": 45}, row_id="c"),
    ]
    kept, dropped = build_source_a_prompts(rows, eval_char_values=EVAL_CHAR_VALUES)
    assert dropped == 2
    assert len(kept) == 1
    assert kept[0]["constraint_detail"] == {"max": 45}


def test_build_source_a_prompts_keeps_non_char_categories_untouched() -> None:
    rows = [_csv_row("no_constraint", {}, row_id="a")]
    kept, dropped = build_source_a_prompts(rows, eval_char_values=EVAL_CHAR_VALUES)
    assert dropped == 0
    assert len(kept) == 1


def test_build_source_a_prompts_schema_and_sequential_ids() -> None:
    rows = [
        _csv_row("no_constraint", {}, prompt="P1", topic="T1", row_id="a"),
        _csv_row("polite_form", {"style": "polite"}, prompt="P2", topic="T2", row_id="b"),
    ]
    kept, _ = build_source_a_prompts(rows, eval_char_values=set(), start_index=1)
    assert [r["id"] for r in kept] == ["pref-00001", "pref-00002"]
    assert kept[0]["prompt"] == "P1"
    assert kept[0]["topic"] == "T1"
    assert set(kept[0].keys()) == {
        "id",
        "category",
        "instruction_id_list",
        "constraint_detail",
        "topic",
        "prompt",
    }
    # response column must never leak through
    assert "response" not in kept[0]


def test_build_source_a_prompts_start_index_continues_numbering() -> None:
    rows = [_csv_row("no_constraint", {}, row_id="a")]
    kept, _ = build_source_a_prompts(rows, eval_char_values=set(), start_index=50)
    assert kept[0]["id"] == "pref-00050"


_TOPICS_80 = [f"topic{i:03d}" for i in range(80)]


def test_build_source_b_prompts_bucket_counts_match_spec_targets() -> None:
    rows = build_source_b_prompts(_TOPICS_80, start_index=1, seed=42)

    lte = [
        r
        for r in rows
        if r["category"] == "char_count"
        and "max" in r["constraint_detail"]
        and "min" not in r["constraint_detail"]
    ]
    gte = [
        r
        for r in rows
        if r["category"] == "char_count"
        and "min" in r["constraint_detail"]
        and "max" not in r["constraint_detail"]
    ]
    both = [
        r
        for r in rows
        if r["category"] == "char_count"
        and "min" in r["constraint_detail"]
        and "max" in r["constraint_detail"]
    ]
    plain = [r for r in rows if r["category"] == "polite_form"]

    # ~300 / ~200 / ~150 / ~200 per Issue #115 spec (exact counts are this
    # module's own deterministic bucket sizing, documented in pref_prompts.py).
    assert 280 <= len(lte) <= 320
    assert 180 <= len(gte) <= 220
    assert 130 <= len(both) <= 170
    assert 180 <= len(plain) <= 220

    assert {r["constraint_detail"]["max"] for r in lte} == {40, 45, 55, 65, 75}
    assert {r["constraint_detail"]["min"] for r in gte} == {30, 40, 55, 65, 75}
    assert {(r["constraint_detail"]["min"], r["constraint_detail"]["max"]) for r in both} == {
        (40, 110),
        (55, 130),
        (65, 140),
        (75, 160),
    }
    for r in plain:
        assert r["constraint_detail"] == {"style": "plain"}


def test_build_source_b_prompts_char_values_avoid_eval_set() -> None:
    rows = build_source_b_prompts(_TOPICS_80, start_index=1, seed=42)
    all_values = set()
    for r in rows:
        detail = r["constraint_detail"]
        for key in ("min", "max"):
            if key in detail:
                all_values.add(detail[key])
    assert all_values & EVAL_CHAR_VALUES == set()


def test_build_source_b_prompts_deterministic_for_same_seed() -> None:
    a = build_source_b_prompts(_TOPICS_80, start_index=1, seed=42)
    b = build_source_b_prompts(_TOPICS_80, start_index=1, seed=42)
    assert a == b


def test_build_source_b_prompts_differs_for_different_seed() -> None:
    a = build_source_b_prompts(_TOPICS_80, start_index=1, seed=42)
    b = build_source_b_prompts(_TOPICS_80, start_index=1, seed=1)
    assert [r["prompt"] for r in a] != [r["prompt"] for r in b]


def test_build_source_b_prompts_ids_are_sequential_from_start_index() -> None:
    rows = build_source_b_prompts(_TOPICS_80, start_index=100, seed=42)
    ids = [r["id"] for r in rows]
    assert ids[0] == "pref-00100"
    assert ids == sorted(ids)


def test_check_prompt_pool_non_duplication_passes_when_clean() -> None:
    rows = [_pool_row("pref-00001", "char_count", {"max": 45}, topic="キャンプ")]
    eval_prompts = _eval_prompts([50, 60], ["別のプロンプトです。"])
    result = check_prompt_pool_non_duplication(rows, eval_prompts)
    assert result["value_overlap"] == []
    assert result["topic_hits"] == []


def test_check_prompt_pool_non_duplication_raises_on_char_value_collision() -> None:
    rows = [_pool_row("pref-00001", "char_count", {"max": 100})]
    eval_prompts = _eval_prompts([100], [])
    with pytest.raises(ValueError):
        check_prompt_pool_non_duplication(rows, eval_prompts)


def test_check_prompt_pool_non_duplication_raises_on_min_value_collision() -> None:
    # The 以上-type prompts carry a 'min' only -- the gate must check min too,
    # not just max (distill_select's original gate only ever needed to check
    # max, since the CSV never has a bare min-only char_count row).
    rows = [_pool_row("pref-00001", "char_count", {"min": 50})]
    eval_prompts = _eval_prompts([50], [])
    with pytest.raises(ValueError):
        check_prompt_pool_non_duplication(rows, eval_prompts)


def test_check_prompt_pool_non_duplication_raises_on_topic_substring() -> None:
    rows = [_pool_row("pref-00001", "no_constraint", {}, topic="家庭菜園")]
    eval_prompts = _eval_prompts([], ["家庭菜園についての質問です。"])
    with pytest.raises(ValueError):
        check_prompt_pool_non_duplication(rows, eval_prompts)


def test_render_pool_stats_report_contains_key_sections() -> None:
    stats = {
        "seed": 42,
        "total": 4841,
        "source_a_count": 3988,
        "source_a_dropped_eval_collision": 12,
        "source_b_count": 853,
        "category_counts": {"char_count": 1200, "polite_form": 1132},
        "non_duplication": {
            "eval_char_values": [50, 60, 70, 80, 90, 100, 120, 150],
            "pool_char_values": [40, 45],
            "value_overlap": [],
            "topic_hits": [],
        },
        "output_path": "data/processed/dpo/pref_prompts.jsonl",
    }
    report = render_pool_stats_report(stats)
    assert "dpo-001" in report
    assert "4841" in report
    assert "12" in report
    assert "char_count" in report


def test_build_prompt_pool_end_to_end_smoke(tmp_path: Path) -> None:
    import csv

    csv_rows = []
    for i in range(20):
        csv_rows.append(
            {
                "id": f"distill-{i:05d}",
                "category": "no_constraint",
                "instruction_id_list": "no_constraint",
                "constraint_detail": json.dumps({}),
                "topic": f"topicA{i}",
                "prompt": f"質問{i}",
                "response": "あ" * 100,
                "response_char_count": "100",
            }
        )
    # one row that must be dropped by the eval-collision proactive gate
    csv_rows.append(
        {
            "id": "distill-00099",
            "category": "char_count",
            "instruction_id_list": "char_count",
            "constraint_detail": json.dumps({"max": 100}),
            "topic": "topicCollide",
            "prompt": "衝突する質問",
            "response": "あ" * 70,
            "response_char_count": "70",
        }
    )
    csv_path = tmp_path / "distill.csv"
    fieldnames = [
        "id",
        "category",
        "instruction_id_list",
        "constraint_detail",
        "topic",
        "prompt",
        "response",
        "response_char_count",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    eval_prompts_path = tmp_path / "eval_prompts.jsonl"
    _write_jsonl(eval_prompts_path, _eval_prompts([100], ["まったく別のプロンプトです。"]))

    output_path = tmp_path / "pref_prompts.jsonl"
    stats_report_path = tmp_path / "phase4_dpo001_pool_stats.md"
    config = {
        "pref_prompts": {
            "seed": 42,
            "source_csv": str(csv_path),
            "eval_prompts_path": str(eval_prompts_path),
            "output_path": str(output_path),
            "stats_report": str(stats_report_path),
        }
    }
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    stats = build_prompt_pool(config_path)

    assert stats["source_a_dropped_eval_collision"] == 1
    assert stats["source_a_count"] == 20
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == stats["total"]
    for line in lines:
        row = json.loads(line)
        assert row["id"].startswith("pref-")
        assert "prompt" in row
        assert "constraint_detail" in row
    assert Path(stats["report_path"]).exists()


def test_real_distill_csv_and_eval_prompts_produce_a_clean_pool(tmp_path: Path) -> None:
    """Integration smoke test against the real data files: confirms the
    Source A eval-collision drop count matches what was found by manual
    inspection (12 'compound' rows with max=90) and that the full pool
    (Source A + Source B) passes the non-duplication hard gate cleanly."""
    if not REAL_CSV_PATH.exists() or not REAL_EVAL_PROMPTS_PATH.exists():
        pytest.skip("real data files not present in this checkout")

    output_path = tmp_path / "pref_prompts.jsonl"
    stats_report_path = tmp_path / "phase4_dpo001_pool_stats.md"
    config = {
        "pref_prompts": {
            "seed": 42,
            "source_csv": str(REAL_CSV_PATH),
            "eval_prompts_path": str(REAL_EVAL_PROMPTS_PATH),
            "output_path": str(output_path),
            "stats_report": str(stats_report_path),
        }
    }
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    stats = build_prompt_pool(config_path)

    assert stats["source_a_dropped_eval_collision"] == 12
    assert stats["source_a_count"] == 4000 - 12
    assert stats["non_duplication"]["value_overlap"] == []
    assert stats["non_duplication"]["topic_hits"] == []
    ids = [json.loads(line)["id"] for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(set(ids))


# ===========================================================================
# Phase G: pref_generate.py -- plan building + idempotency (CPU parts only)
# ===========================================================================


def test_build_generation_plan_expands_k_samples_per_prompt() -> None:
    prompts = [
        _pool_row("pref-00001", "no_constraint", {}),
        _pool_row("pref-00002", "no_constraint", {}),
    ]
    plan = build_generation_plan(prompts, num_samples=3, existing_keys=set())
    assert len(plan) == 6
    assert {(p["prompt_id"], p["k"]) for p in plan} == {
        ("pref-00001", 0),
        ("pref-00001", 1),
        ("pref-00001", 2),
        ("pref-00002", 0),
        ("pref-00002", 1),
        ("pref-00002", 2),
    }


def test_build_generation_plan_skips_existing_keys() -> None:
    prompts = [_pool_row("pref-00001", "no_constraint", {})]
    existing = {("pref-00001", 0), ("pref-00001", 2)}
    plan = build_generation_plan(prompts, num_samples=3, existing_keys=existing)
    assert [(p["prompt_id"], p["k"]) for p in plan] == [("pref-00001", 1)]


def test_build_generation_plan_limit_truncates_prompts_not_samples() -> None:
    prompts = [
        _pool_row("pref-00001", "no_constraint", {}),
        _pool_row("pref-00002", "no_constraint", {}),
    ]
    plan = build_generation_plan(prompts, num_samples=4, existing_keys=set(), limit=1)
    assert len(plan) == 4
    assert all(p["prompt_id"] == "pref-00001" for p in plan)


def test_existing_generation_keys_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt_id": "pref-00001", "k": 0, "prompt": "q", "response": "a"},
            {"prompt_id": "pref-00001", "k": 1, "prompt": "q", "response": "b"},
        ],
    )
    assert _existing_generation_keys(path) == {("pref-00001", 0), ("pref-00001", 1)}
    assert _existing_generation_keys(tmp_path / "missing.jsonl") == set()


# ===========================================================================
# Phase V: pref_verify.py -- rule-based verdicts
# ===========================================================================


def test_verify_sample_char_count_pass() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 100})
    verdict = verify_sample(prompt_row, _filler(70))
    assert verdict["rule_pass"] is True
    assert verdict["degenerate"] is False


def test_verify_sample_char_count_fail_over_max() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 100})
    verdict = verify_sample(prompt_row, "あ" * 150)
    assert verdict["rule_pass"] is False
    assert verdict["rule_reason"]


def test_verify_sample_char_count_min_only() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"min": 50})
    assert verify_sample(prompt_row, _filler(60))["rule_pass"] is True
    assert verify_sample(prompt_row, _filler(30))["rule_pass"] is False


def test_verify_sample_polite_form_plain_pass_and_fail() -> None:
    prompt_row = _pool_row("pref-00001", "polite_form", {"style": "plain"})
    assert verify_sample(prompt_row, "東京は日本の首都だ。")["rule_pass"] is True
    assert verify_sample(prompt_row, "東京は日本の首都です。")["rule_pass"] is False


def test_verify_sample_no_constraint() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    assert verify_sample(prompt_row, _filler(100))["rule_pass"] is True
    assert verify_sample(prompt_row, "短い")["rule_pass"] is False


def test_verify_sample_degenerate_empty_response() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    verdict = verify_sample(prompt_row, "")
    assert verdict["degenerate"] is True


def test_verify_sample_degenerate_too_short() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 300})
    verdict = verify_sample(prompt_row, "短い")
    assert verdict["degenerate"] is True


def test_verify_sample_degenerate_repetition() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    response = "\n".join(["同じ行です。"] * 3 + [_filler(30)])
    verdict = verify_sample(prompt_row, response)
    assert verdict["degenerate"] is True


def test_verify_sample_format_json_exempt_from_repetition_check() -> None:
    payload = json.dumps([{"名称": f"項目{i}", "分類": "X"} for i in range(5)], ensure_ascii=False)
    prompt_row = _pool_row("pref-00001", "format_json", {"keys": ["名称"], "count": 5})
    verdict = verify_sample(prompt_row, payload)
    assert verdict["degenerate"] is False
    assert verdict["rule_pass"] is True


def test_verify_sample_unknown_category_fails() -> None:
    prompt_row = _pool_row("pref-00001", "mystery", {})
    verdict = verify_sample(prompt_row, _filler(50))
    assert verdict["rule_pass"] is False
    assert verdict["rule_reason"]


def test_verify_generations_maps_by_prompt_id_and_k() -> None:
    prompts_by_id = {
        "pref-00001": _pool_row("pref-00001", "char_count", {"max": 100}),
    }
    generations = [
        {"prompt_id": "pref-00001", "k": 0, "prompt": "質問", "response": "あ" * 70},
        {"prompt_id": "pref-00001", "k": 1, "prompt": "質問", "response": "あ" * 150},
    ]
    verdicts = verify_generations(prompts_by_id, generations)
    assert len(verdicts) == 2
    assert verdicts[0]["k"] == 0
    assert verdicts[0]["rule_pass"] is True
    assert verdicts[1]["rule_pass"] is False


def test_run_verification_end_to_end_writes_jsonl(tmp_path: Path) -> None:
    prompts_path = tmp_path / "pref_prompts.jsonl"
    _write_jsonl(
        prompts_path,
        [
            _pool_row("pref-00001", "char_count", {"max": 100}),
            _pool_row("pref-00002", "no_constraint", {}),
        ],
    )
    generations_path = tmp_path / "generations.jsonl"
    _write_jsonl(
        generations_path,
        [
            {"prompt_id": "pref-00001", "k": 0, "prompt": "質問", "response": "あ" * 70},
            {"prompt_id": "pref-00001", "k": 1, "prompt": "質問", "response": "あ" * 150},
            {"prompt_id": "pref-00002", "k": 0, "prompt": "質問2", "response": _filler(100)},
        ],
    )
    output_path = tmp_path / "verdicts.jsonl"
    config = {
        "pref_verify": {
            "prompts_path": str(prompts_path),
            "generations_path": str(generations_path),
            "output_path": str(output_path),
        }
    }
    config_path = tmp_path / "verify.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    result = run_verification(config_path)

    assert result["total"] == 3
    assert result["rule_pass"] == 2
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert "prompt_id" in row
        assert "k" in row
        assert "rule_pass" in row
        assert "degenerate" in row


# ===========================================================================
# Phase J: judge_swallow.py -- LLM judge (GPU-free unit tests only)
# ===========================================================================


def test_build_judge_prompt_contains_prompt_and_response() -> None:
    prompt = build_judge_prompt("これは何ですか？", "これは回答です。")
    assert "これは何ですか？" in prompt
    assert "これは回答です。" in prompt


def test_build_judge_prompt_mentions_truncation_caveat() -> None:
    prompt = build_judge_prompt("Q", "R")
    assert "切れている" in prompt


def test_build_judge_prompt_requests_strict_json() -> None:
    prompt = build_judge_prompt("Q", "R")
    assert "score" in prompt
    assert "reason" in prompt


def test_parse_judge_output_valid_json() -> None:
    result = parse_judge_output('{"score": 4, "reason": "自然な日本語です"}')
    assert result == {"score": 4, "reason": "自然な日本語です"}


def test_parse_judge_output_handles_surrounding_text() -> None:
    text = 'はい、採点します。\n{"score": 3, "reason": "普通です"}\nありがとうございました。'
    result = parse_judge_output(text)
    assert result["score"] == 3


def test_parse_judge_output_handles_code_fence() -> None:
    text = '```json\n{"score": 5, "reason": "とても良い"}\n```'
    result = parse_judge_output(text)
    assert result["score"] == 5


def test_parse_judge_output_invalid_json_returns_none_score() -> None:
    result = parse_judge_output("これはJSONではありません")
    assert result["score"] is None


def test_parse_judge_output_out_of_range_score_returns_none() -> None:
    result = parse_judge_output('{"score": 9, "reason": "x"}')
    assert result["score"] is None


def test_parse_judge_output_non_integer_score_returns_none() -> None:
    result = parse_judge_output('{"score": "high", "reason": "x"}')
    assert result["score"] is None


def test_parse_judge_output_empty_text_returns_none() -> None:
    result = parse_judge_output("")
    assert result["score"] is None


_PROMPTS_FOR_JUDGE = {
    "pref-00001": _pool_row("pref-00001", "char_count", {"max": 100}),
    "pref-00002": _pool_row("pref-00002", "polite_form", {"style": "polite"}),
    "pref-00003": _pool_row("pref-00003", "no_constraint", {}),
    "pref-00004": _pool_row("pref-00004", "keyword_include", {"include": ["x"]}),
}


def test_select_judge_targets_includes_pair_forming_prompts() -> None:
    generations = [
        {"prompt_id": "pref-00001", "k": 0, "response": "あ" * 70},
        {"prompt_id": "pref-00001", "k": 1, "response": "あ" * 150},
    ]
    verdicts = [
        {"prompt_id": "pref-00001", "k": 0, "rule_pass": True, "degenerate": False},
        {"prompt_id": "pref-00001", "k": 1, "rule_pass": False, "degenerate": False},
    ]
    targets = select_judge_targets(_PROMPTS_FOR_JUDGE, generations, verdicts)
    assert {(t["prompt_id"], t["k"]) for t in targets} == {("pref-00001", 0), ("pref-00001", 1)}


def test_select_judge_targets_includes_polite_form_even_when_all_pass() -> None:
    generations = [
        {"prompt_id": "pref-00002", "k": 0, "response": "東京は首都です。"},
        {"prompt_id": "pref-00002", "k": 1, "response": "東京は日本の首都です。"},
    ]
    verdicts = [
        {"prompt_id": "pref-00002", "k": 0, "rule_pass": True, "degenerate": False},
        {"prompt_id": "pref-00002", "k": 1, "rule_pass": True, "degenerate": False},
    ]
    targets = select_judge_targets(_PROMPTS_FOR_JUDGE, generations, verdicts)
    assert len(targets) == 2


def test_select_judge_targets_excludes_all_pass_non_special_category() -> None:
    # keyword_include is not polite_form/no_constraint, and all samples pass
    # -> no pair is possible and no quality-selection override applies.
    generations = [
        {"prompt_id": "pref-00004", "k": 0, "response": "xを含む文章です。"},
        {"prompt_id": "pref-00004", "k": 1, "response": "xを含む別の文章です。"},
    ]
    verdicts = [
        {"prompt_id": "pref-00004", "k": 0, "rule_pass": True, "degenerate": False},
        {"prompt_id": "pref-00004", "k": 1, "rule_pass": True, "degenerate": False},
    ]
    targets = select_judge_targets(_PROMPTS_FOR_JUDGE, generations, verdicts)
    assert targets == []


def test_select_judge_targets_skips_empty_response() -> None:
    generations = [
        {"prompt_id": "pref-00001", "k": 0, "response": "あ" * 70},
        {"prompt_id": "pref-00001", "k": 1, "response": "   "},
    ]
    verdicts = [
        {"prompt_id": "pref-00001", "k": 0, "rule_pass": True, "degenerate": False},
        {"prompt_id": "pref-00001", "k": 1, "rule_pass": False, "degenerate": True},
    ]
    targets = select_judge_targets(_PROMPTS_FOR_JUDGE, generations, verdicts)
    assert ("pref-00001", 1) not in {(t["prompt_id"], t["k"]) for t in targets}


# ===========================================================================
# Phase P: pref_pairs.py -- chosen/rejected pairing + hard gates
# ===========================================================================


def _sample(k: int, response: str, rule_pass: bool, degenerate: bool = False, score=None) -> dict:
    return {
        "k": k,
        "response": response,
        "rule_pass": rule_pass,
        "rule_reason": "" if rule_pass else "違反",
        "degenerate": degenerate,
        "score": score,
        "judge_reason": "",
    }


def test_select_chosen_picks_max_score_pass_non_degenerate() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [
        _sample(0, _filler(80), rule_pass=True, score=3),
        _sample(1, _filler(90), rule_pass=True, score=5),
        _sample(2, _filler(60), rule_pass=False, score=5),
    ]
    chosen = select_chosen(prompt_row, samples)
    assert chosen["k"] == 1


def test_select_chosen_tie_break_prefers_shorter_response() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [
        _sample(0, _filler(120), rule_pass=True, score=4),
        _sample(1, _filler(80), rule_pass=True, score=4),
    ]
    chosen = select_chosen(prompt_row, samples)
    assert chosen["k"] == 1


def test_select_chosen_polite_form_requires_min_score() -> None:
    prompt_row = _pool_row("pref-00001", "polite_form", {"style": "polite"})
    samples = [
        _sample(0, "です。", rule_pass=True, score=2),
        _sample(1, "です。", rule_pass=True, score=3),
    ]
    chosen = select_chosen(prompt_row, samples, polite_min_score=3)
    assert chosen["k"] == 1

    samples_all_low = [_sample(0, "です。", rule_pass=True, score=2)]
    assert select_chosen(prompt_row, samples_all_low, polite_min_score=3) is None


def test_select_chosen_excludes_null_score_candidates() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [
        _sample(0, _filler(80), rule_pass=True, score=None),
        _sample(1, _filler(80), rule_pass=True, score=2),
    ]
    chosen = select_chosen(prompt_row, samples)
    assert chosen["k"] == 1


def test_select_chosen_excludes_degenerate_candidates() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [_sample(0, _filler(80), rule_pass=True, degenerate=True, score=5)]
    assert select_chosen(prompt_row, samples) is None


def test_select_chosen_returns_none_when_no_candidates() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [_sample(0, _filler(80), rule_pass=False, score=5)]
    assert select_chosen(prompt_row, samples) is None


def test_select_rejected_picks_max_score_fail_non_degenerate() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 100})
    samples = [
        _sample(0, _filler(150), rule_pass=False, score=2),
        _sample(1, _filler(120), rule_pass=False, score=4),
        _sample(2, _filler(80), rule_pass=True, score=5),
    ]
    rejected = select_rejected(prompt_row, samples)
    assert rejected["k"] == 1


def test_select_rejected_tie_break_prefers_smaller_violation_margin() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 100})
    samples = [
        _sample(0, _filler(180), rule_pass=False, score=3),  # overage 80
        _sample(1, _filler(110), rule_pass=False, score=3),  # overage 10
    ]
    rejected = select_rejected(prompt_row, samples)
    assert rejected["k"] == 1


def test_select_rejected_allows_null_score_candidates_as_fallback() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 100})
    samples = [
        _sample(0, _filler(180), rule_pass=False, score=None),
        _sample(1, _filler(110), rule_pass=False, score=None),
    ]
    rejected = select_rejected(prompt_row, samples)
    assert rejected is not None
    assert rejected["k"] == 1  # still tie-broken by smallest violation margin


def test_select_rejected_prefers_scored_over_null_candidates() -> None:
    prompt_row = _pool_row("pref-00001", "char_count", {"max": 100})
    samples = [
        _sample(0, _filler(110), rule_pass=False, score=None),  # closer miss but unscored
        _sample(1, _filler(180), rule_pass=False, score=3),  # farther miss but scored
    ]
    rejected = select_rejected(prompt_row, samples)
    assert rejected["k"] == 1


def test_select_rejected_returns_none_when_no_candidates() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [_sample(0, _filler(80), rule_pass=True, score=5)]
    assert select_rejected(prompt_row, samples) is None


def test_build_pair_returns_reason_when_no_chosen() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [_sample(0, _filler(80), rule_pass=False, score=5)]
    pair, reason = build_pair(prompt_row, samples)
    assert pair is None
    assert reason == "no_chosen_candidate"


def test_build_pair_returns_reason_when_no_rejected() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {})
    samples = [_sample(0, _filler(80), rule_pass=True, score=5)]
    pair, reason = build_pair(prompt_row, samples)
    assert pair is None
    assert reason == "no_rejected_candidate"


def test_build_pair_produces_trl_dpo_schema() -> None:
    prompt_row = _pool_row("pref-00001", "no_constraint", {}, prompt="質問文")
    samples = [
        _sample(0, _filler(176), rule_pass=True, score=4),
        _sample(1, _filler(90), rule_pass=False, score=3),
    ]
    pair, reason = build_pair(prompt_row, samples)
    assert reason == "ok"
    assert pair["prompt"] == "質問文"
    assert pair["chosen"] == _filler(176)
    assert pair["rejected"] == _filler(90)
    assert pair["meta"]["prompt_id"] == "pref-00001"
    assert pair["meta"]["category"] == "no_constraint"


def test_merge_samples_joins_generations_verdicts_and_judgments() -> None:
    generations_by_prompt = {
        "pref-00001": [
            {"prompt_id": "pref-00001", "k": 0, "response": "あ" * 70},
        ]
    }
    verdicts_by_key = {
        ("pref-00001", 0): {
            "prompt_id": "pref-00001",
            "k": 0,
            "rule_pass": True,
            "degenerate": False,
            "rule_reason": "",
        }
    }
    judgments_by_key = {
        ("pref-00001", 0): {"prompt_id": "pref-00001", "k": 0, "score": 4, "reason": "良い"}
    }

    samples = merge_samples("pref-00001", generations_by_prompt, verdicts_by_key, judgments_by_key)
    assert len(samples) == 1
    assert samples[0]["score"] == 4
    assert samples[0]["rule_pass"] is True


def test_render_pairs_stats_report_contains_key_sections() -> None:
    stats = {
        "total_pairs": 500,
        "total_prompts": 800,
        "category_pair_counts": {"char_count": 200, "polite_form": 100},
        "skip_reasons": {
            "no_chosen_candidate": 100,
            "no_rejected_candidate": 150,
            "no_generations": 50,
        },
        "chosen_length": {"mean": 180.0, "median": 176, "min": 100, "max": 250},
        "rejected_length": {"mean": 210.0, "median": 200, "min": 90, "max": 320},
        "score_distribution": {"chosen_mean": 4.2, "rejected_mean": 2.1},
        "length_guard": {
            "mean": 180.0,
            "median": 176,
            "min": 100,
            "max": 250,
            "lower_bound": 140.8,
            "upper_bound": 211.2,
            "base_mean": 176,
            "tolerance": 0.20,
            "within_band": True,
        },
        "non_duplication": {
            "eval_char_values": [50, 60],
            "distill_max_values": [45],
            "value_overlap": [],
            "topic_hits": [],
        },
    }
    report = render_pairs_stats_report(stats)
    assert "dpo-001" in report
    assert "500" in report
    assert "char_count" in report
    assert "no_chosen_candidate" in report


def test_build_preference_pairs_end_to_end_smoke(tmp_path: Path) -> None:
    prompts = [
        _pool_row("pref-00001", "no_constraint", {}, prompt="質問1"),
        _pool_row("pref-00002", "char_count", {"max": 100}, prompt="質問2"),
        _pool_row("pref-00003", "no_constraint", {}, prompt="質問3"),  # no pair possible
    ]
    prompts_path = tmp_path / "pref_prompts.jsonl"
    _write_jsonl(prompts_path, prompts)

    generations = [
        {"prompt_id": "pref-00001", "k": 0, "response": _filler(176)},
        {"prompt_id": "pref-00001", "k": 1, "response": _filler(60)},
        {"prompt_id": "pref-00002", "k": 0, "response": _filler(70)},
        {"prompt_id": "pref-00002", "k": 1, "response": _filler(150)},
        {"prompt_id": "pref-00003", "k": 0, "response": _filler(176)},
    ]
    generations_path = tmp_path / "generations.jsonl"
    _write_jsonl(generations_path, generations)

    verdicts = [
        {
            "prompt_id": "pref-00001",
            "k": 0,
            "rule_pass": True,
            "degenerate": False,
            "rule_reason": "",
        },
        {
            "prompt_id": "pref-00001",
            "k": 1,
            "rule_pass": False,
            "degenerate": False,
            "rule_reason": "短すぎ",
        },
        {
            "prompt_id": "pref-00002",
            "k": 0,
            "rule_pass": True,
            "degenerate": False,
            "rule_reason": "",
        },
        {
            "prompt_id": "pref-00002",
            "k": 1,
            "rule_pass": False,
            "degenerate": False,
            "rule_reason": "文字数超過",
        },
        {
            "prompt_id": "pref-00003",
            "k": 0,
            "rule_pass": True,
            "degenerate": False,
            "rule_reason": "",
        },
    ]
    verdicts_path = tmp_path / "verdicts.jsonl"
    _write_jsonl(verdicts_path, verdicts)

    judgments = [
        {"prompt_id": "pref-00001", "k": 0, "score": 4, "reason": "良い"},
        {"prompt_id": "pref-00001", "k": 1, "score": 2, "reason": "短い"},
        {"prompt_id": "pref-00002", "k": 0, "score": 4, "reason": "良い"},
        {"prompt_id": "pref-00002", "k": 1, "score": 3, "reason": "普通"},
        # pref-00003 k0 judged (no_constraint is a quality-select category
        # upstream) but the prompt has no failing sample, so pairing skips it
        # with no_rejected_candidate.
        {"prompt_id": "pref-00003", "k": 0, "score": 4, "reason": "良い"},
    ]
    judgments_path = tmp_path / "judgments.jsonl"
    _write_jsonl(judgments_path, judgments)

    eval_prompts_path = tmp_path / "eval_prompts.jsonl"
    _write_jsonl(eval_prompts_path, _eval_prompts([50, 60], ["まったく別のプロンプトです。"]))

    output_path = tmp_path / "dpo_pairs.jsonl"
    stats_report_path = tmp_path / "phase4_dpo001_pairs_stats.md"
    config = {
        "pref_pairs": {
            "prompts_path": str(prompts_path),
            "generations_path": str(generations_path),
            "verdicts_path": str(verdicts_path),
            "judgments_path": str(judgments_path),
            "eval_prompts_path": str(eval_prompts_path),
            "output_path": str(output_path),
            "stats_report": str(stats_report_path),
            "polite_min_score": 3,
            # Wide tolerance: this test exercises pairing mechanics, not the
            # length guard (which has its own violation test below); the tiny
            # fixture's chosen mean (~123) shouldn't trip it.
            "length_guard": {"base_mean_chars": 176, "tolerance": 0.45},
        }
    }
    config_path = tmp_path / "pairs.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    stats = build_preference_pairs(config_path)

    assert stats["total_pairs"] == 2
    assert stats["skip_reasons"].get("no_rejected_candidate", 0) == 1

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert set(row.keys()) == {"prompt", "chosen", "rejected", "meta"}
    assert Path(stats["report_path"]).exists()


def test_build_preference_pairs_raises_on_length_guard_violation(tmp_path: Path) -> None:
    prompts = [_pool_row("pref-00001", "no_constraint", {}, prompt="質問1")]
    prompts_path = tmp_path / "pref_prompts.jsonl"
    _write_jsonl(prompts_path, prompts)

    generations = [
        {"prompt_id": "pref-00001", "k": 0, "response": _filler(30)},
        {"prompt_id": "pref-00001", "k": 1, "response": _filler(25)},
    ]
    generations_path = tmp_path / "generations.jsonl"
    _write_jsonl(generations_path, generations)

    verdicts = [
        {
            "prompt_id": "pref-00001",
            "k": 0,
            "rule_pass": True,
            "degenerate": False,
            "rule_reason": "",
        },
        {
            "prompt_id": "pref-00001",
            "k": 1,
            "rule_pass": False,
            "degenerate": False,
            "rule_reason": "x",
        },
    ]
    verdicts_path = tmp_path / "verdicts.jsonl"
    _write_jsonl(verdicts_path, verdicts)

    judgments = [
        {"prompt_id": "pref-00001", "k": 0, "score": 4, "reason": "良い"},
        {"prompt_id": "pref-00001", "k": 1, "score": 2, "reason": "短い"},
    ]
    judgments_path = tmp_path / "judgments.jsonl"
    _write_jsonl(judgments_path, judgments)

    eval_prompts_path = tmp_path / "eval_prompts.jsonl"
    _write_jsonl(eval_prompts_path, [])

    output_path = tmp_path / "dpo_pairs.jsonl"
    config = {
        "pref_pairs": {
            "prompts_path": str(prompts_path),
            "generations_path": str(generations_path),
            "verdicts_path": str(verdicts_path),
            "judgments_path": str(judgments_path),
            "eval_prompts_path": str(eval_prompts_path),
            "output_path": str(output_path),
            "length_guard": {"base_mean_chars": 176, "tolerance": 0.20},
        }
    }
    config_path = tmp_path / "pairs.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        build_preference_pairs(config_path)
