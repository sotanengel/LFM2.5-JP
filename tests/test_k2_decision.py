"""Unit tests for the frozen K2 cpt-D decision gate (Issue #132 / #138)."""

from __future__ import annotations

from lfm25_ja.eval.k2_decision import (
    K2_DOMAIN_DROP_MAX,
    K2_IFEVAL_PROMPT_STRICT_MIN,
    K2_JKB_OVERALL_MIN,
    K2_LLMJP_AVG_MIN,
    apply_k2_decision_gate,
)


def _base_domains() -> dict[str, float]:
    # 12 domains at a flat base accuracy (values unused except for delta checks)
    names = [
        "地理",
        "歴史",
        "政治・制度",
        "経済",
        "生活・慣習",
        "食文化",
        "伝統文化",
        "言語",
        "文学",
        "地域・観光",
        "スポーツ",
        "科学技術・産業",
    ]
    return {n: 0.50 for n in names}


def test_gate_pass_when_all_four_conditions_met() -> None:
    base = _base_domains()
    cand = {k: 0.50 for k in base}  # exactly base → within -3pt
    cand["生活・慣習"] = 0.55
    result = apply_k2_decision_gate(
        jkb_overall=0.60,
        jkb_by_domain=cand,
        base_by_domain=base,
        ifeval_prompt_strict=0.920,
        llmjp_avg=0.459,
    )
    assert result["pass"] is True
    assert result["verdict"] == "PASS"
    assert all(c["pass"] for c in result["conditions"].values())


def test_gate_fail_when_jkb_overall_just_below_threshold() -> None:
    base = _base_domains()
    result = apply_k2_decision_gate(
        jkb_overall=K2_JKB_OVERALL_MIN - 1e-9,
        jkb_by_domain=dict(base),
        base_by_domain=base,
        ifeval_prompt_strict=K2_IFEVAL_PROMPT_STRICT_MIN,
        llmjp_avg=K2_LLMJP_AVG_MIN,
    )
    assert result["pass"] is False
    assert result["conditions"]["jkb_overall_ge_60"]["pass"] is False


def test_gate_fail_when_one_domain_drops_more_than_3pt() -> None:
    base = _base_domains()
    cand = dict(base)
    cand["言語"] = base["言語"] - (K2_DOMAIN_DROP_MAX + 0.001)
    result = apply_k2_decision_gate(
        jkb_overall=0.70,
        jkb_by_domain=cand,
        base_by_domain=base,
        ifeval_prompt_strict=0.95,
        llmjp_avg=0.47,
    )
    assert result["pass"] is False
    assert result["conditions"]["jkb_domains_within_3pt_of_base"]["pass"] is False
    failures = result["conditions"]["jkb_domains_within_3pt_of_base"]["failures"]
    assert any(f["domain"] == "言語" for f in failures)


def test_gate_pass_when_domain_drop_is_exactly_3pt() -> None:
    base = _base_domains()
    cand = dict(base)
    cand["歴史"] = base["歴史"] - K2_DOMAIN_DROP_MAX
    result = apply_k2_decision_gate(
        jkb_overall=0.60,
        jkb_by_domain=cand,
        base_by_domain=base,
        ifeval_prompt_strict=0.920,
        llmjp_avg=0.459,
    )
    assert result["pass"] is True


def test_gate_fail_ifeval_or_llmjp_guard() -> None:
    base = _base_domains()
    common = dict(
        jkb_overall=0.65,
        jkb_by_domain=dict(base),
        base_by_domain=base,
    )
    bad_ifeval = apply_k2_decision_gate(
        **common,
        ifeval_prompt_strict=K2_IFEVAL_PROMPT_STRICT_MIN - 0.001,
        llmjp_avg=K2_LLMJP_AVG_MIN,
    )
    assert bad_ifeval["pass"] is False
    assert bad_ifeval["conditions"]["ifeval_prompt_strict_ge_0_920"]["pass"] is False

    bad_llmjp = apply_k2_decision_gate(
        **common,
        ifeval_prompt_strict=K2_IFEVAL_PROMPT_STRICT_MIN,
        llmjp_avg=K2_LLMJP_AVG_MIN - 0.001,
    )
    assert bad_llmjp["pass"] is False
    assert bad_llmjp["conditions"]["llmjp_avg_ge_0_459"]["pass"] is False


def test_gate_fail_on_missing_domain_side() -> None:
    base = _base_domains()
    cand = dict(base)
    del cand["経済"]
    result = apply_k2_decision_gate(
        jkb_overall=0.70,
        jkb_by_domain=cand,
        base_by_domain=base,
        ifeval_prompt_strict=0.95,
        llmjp_avg=0.47,
    )
    assert result["pass"] is False
    failures = result["conditions"]["jkb_domains_within_3pt_of_base"]["failures"]
    assert any(f["domain"] == "経済" and f["reason"] == "missing_side" for f in failures)
