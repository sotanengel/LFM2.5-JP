"""K3 factual DPO decision gate (frozen rules from Issue #145).

Issue #124: machine-apply the three AND conditions for K3 factual DPO adoption.
Unlike K2, there is no per-domain -3pt guard — only JKB overall +Ypt and
style/NLU guard rails inherited from dpo-001.
"""

from __future__ import annotations

from typing import Any

# base JKB eval 49.9% + 3pt = 52.9% (Issue #145 session decision)
K3_JKB_OVERALL_MIN = 0.529
K3_IFEVAL_PROMPT_STRICT_MIN = 0.920
K3_LLMJP_AVG_MIN = 0.459


def apply_k3_decision_gate(
    *,
    jkb_overall: float,
    ifeval_prompt_strict: float,
    llmjp_avg: float,
) -> dict[str, Any]:
    """Apply the frozen K3 AND gate; return a structured PASS/FAIL verdict."""
    cond_jkb = jkb_overall >= K3_JKB_OVERALL_MIN
    cond_ifeval = ifeval_prompt_strict >= K3_IFEVAL_PROMPT_STRICT_MIN
    cond_llmjp = llmjp_avg >= K3_LLMJP_AVG_MIN

    conditions = {
        "jkb_overall_ge_52_9": {
            "pass": cond_jkb,
            "value": jkb_overall,
            "threshold": K3_JKB_OVERALL_MIN,
        },
        "ifeval_prompt_strict_ge_0_920": {
            "pass": cond_ifeval,
            "value": ifeval_prompt_strict,
            "threshold": K3_IFEVAL_PROMPT_STRICT_MIN,
        },
        "llmjp_avg_ge_0_459": {
            "pass": cond_llmjp,
            "value": llmjp_avg,
            "threshold": K3_LLMJP_AVG_MIN,
        },
    }
    passed = all(c["pass"] for c in conditions.values())
    return {
        "pass": passed,
        "conditions": conditions,
        "verdict": "PASS" if passed else "FAIL",
    }
