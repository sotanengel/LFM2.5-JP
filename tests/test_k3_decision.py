"""Unit tests for frozen K3 factual DPO decision gate (Issue #124 / #145)."""

from __future__ import annotations

import pytest

from lfm25_ja.eval.k3_decision import (
    K3_IFEVAL_PROMPT_STRICT_MIN,
    K3_JKB_OVERALL_MIN,
    K3_LLMJP_AVG_MIN,
    apply_k3_decision_gate,
)


def test_k3_gate_pass_when_all_conditions_met() -> None:
    result = apply_k3_decision_gate(
        jkb_overall=0.529,
        ifeval_prompt_strict=0.920,
        llmjp_avg=0.459,
    )
    assert result["pass"] is True
    assert result["verdict"] == "PASS"


def test_k3_gate_fail_when_jkb_below_plus_3pt() -> None:
    result = apply_k3_decision_gate(
        jkb_overall=0.528,
        ifeval_prompt_strict=0.950,
        llmjp_avg=0.469,
    )
    assert result["pass"] is False
    assert result["conditions"]["jkb_overall_ge_52_9"]["pass"] is False


def test_k3_gate_fail_when_ifeval_guard_violated() -> None:
    result = apply_k3_decision_gate(
        jkb_overall=0.55,
        ifeval_prompt_strict=0.919,
        llmjp_avg=0.469,
    )
    assert result["pass"] is False
    assert result["conditions"]["ifeval_prompt_strict_ge_0_920"]["pass"] is False


def test_k3_frozen_thresholds() -> None:
    assert K3_JKB_OVERALL_MIN == pytest.approx(0.529)
    assert K3_IFEVAL_PROMPT_STRICT_MIN == pytest.approx(0.920)
    assert K3_LLMJP_AVG_MIN == pytest.approx(0.459)
