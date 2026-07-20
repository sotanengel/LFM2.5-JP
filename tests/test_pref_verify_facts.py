"""Tests for K3 factual verification (Issue #124 / #145)."""

from __future__ import annotations

from lfm25_ja.data.pref_verify_facts import is_escape_response, verify_sample


def _prompt_row() -> dict:
    return {
        "id": "k3pref-00001",
        "category": "jkb_fact",
        "constraint_detail": {
            "jkb_id": "jkb-geo-core-011",
            "format": "short_answer",
            "answers": ["日本海"],
            "choices": None,
            "correct_choice": None,
        },
    }


def test_verify_sample_marks_correct_answer_as_rule_pass() -> None:
    v = verify_sample(_prompt_row(), "日本海です。")
    assert v["rule_pass"] is True
    assert v["fact_correct"] is True


def test_verify_sample_marks_wrong_answer_as_fail() -> None:
    v = verify_sample(_prompt_row(), "太平洋です。")
    assert v["rule_pass"] is False
    assert v["fact_correct"] is False


def test_verify_sample_marks_escape_as_fail_even_if_short() -> None:
    v = verify_sample(_prompt_row(), "申し訳ありませんが、わかりません。")
    assert v["escape"] is True
    assert v["rule_pass"] is False


def test_is_escape_response() -> None:
    assert is_escape_response("詳しくは分かりません。") is True
    assert is_escape_response("日本海") is False
