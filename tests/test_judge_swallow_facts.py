"""Tests for Swallow factual judge helpers (Issue #124 / #145)."""

from __future__ import annotations

import json

from lfm25_ja.eval.judge_swallow import (
    build_factual_judge_prompt,
    parse_judge_output,
    select_factual_judge_targets,
)


def test_build_factual_judge_prompt_includes_reference_and_response() -> None:
    text = build_factual_judge_prompt(
        question="日本で一番高い山は?",
        reference_answers=["富士山", "富士"],
        source_quote="富士山は日本で最も高い山である。",
        response="富士山です。",
    )
    assert "富士山" in text
    assert "参照解" in text or "reference" in text.lower() or "正解" in text
    assert "富士山です。" in text


def test_parse_judge_output_accepts_valid_json() -> None:
    parsed = parse_judge_output('{"score": 4, "reason": "正しい"}')
    assert parsed["score"] == 4


def test_select_factual_judge_targets_includes_all_non_empty() -> None:
    prompts_by_id = {"p1": {"id": "p1", "category": "jkb_fact", "prompt": "Q?"}}
    generations = [
        {"prompt_id": "p1", "k": 0, "response": "A"},
        {"prompt_id": "p1", "k": 1, "response": "   "},
    ]
    targets = select_factual_judge_targets(prompts_by_id, generations)
    assert len(targets) == 1
    assert targets[0]["k"] == 0
