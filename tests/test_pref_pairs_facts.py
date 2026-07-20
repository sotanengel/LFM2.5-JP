"""Tests for K3 factual preference pairing (Issue #124 / #145)."""

from __future__ import annotations

from lfm25_ja.data.pref_pairs_facts import (
    build_factual_pair,
    select_chosen_factual,
    select_gap_pair,
    select_rejected_factual,
)


def _sample(k: int, response: str, score: int, degenerate: bool = False) -> dict:
    return {
        "k": k,
        "response": response,
        "rule_pass": score >= 4,
        "rule_reason": "",
        "degenerate": degenerate,
        "score": score,
        "judge_reason": "",
    }


def test_select_chosen_picks_highest_score() -> None:
    samples = [_sample(0, "日本海", 3), _sample(1, "日本海です", 5)]
    chosen = select_chosen_factual(samples)
    assert chosen is not None
    assert chosen["k"] == 1


def test_select_rejected_picks_low_score_or_escape() -> None:
    samples = [_sample(0, "わかりません", 1), _sample(1, "日本海", 5)]
    rejected = select_rejected_factual(samples)
    assert rejected is not None
    assert rejected["k"] == 0


def test_select_gap_pair_when_no_pass_fail_contrast() -> None:
    samples = [_sample(0, "日本海", 4), _sample(1, "日本の海", 5)]
    gap = select_gap_pair(samples)
    assert gap is not None
    hi, lo = gap
    assert hi["score"] == 5
    assert lo["score"] == 4


def test_build_factual_pair_pass_fail() -> None:
    prompt_row = {
        "id": "k3pref-00001",
        "category": "jkb_fact",
        "prompt": "Q",
        "constraint_detail": {"jkb_id": "jkb-x"},
    }
    samples = [
        _sample(0, "誤答", 1),
        _sample(1, "日本海", 5),
    ]
    pair, status = build_factual_pair(prompt_row, samples)
    assert status == "ok"
    assert pair is not None
    assert pair["meta"]["pair_type"] == "pass_fail"
