"""K2 cpt-D decision gate (frozen rules from k1_asset_reassessment.md).

Issue #132 / #138: machine-apply the four AND conditions that decide whether
cpt-D is adopted (and whether deci may proceed to final).
"""

from __future__ import annotations

from typing import Any

# Frozen thresholds (experiments/reports/k1_asset_reassessment.md §K2 決定規則)
K2_JKB_OVERALL_MIN = 0.60
K2_DOMAIN_DROP_MAX = 0.03  # candidate >= base - 3pt
K2_IFEVAL_PROMPT_STRICT_MIN = 0.920
K2_LLMJP_AVG_MIN = 0.459


def apply_k2_decision_gate(
    *,
    jkb_overall: float,
    jkb_by_domain: dict[str, float],
    base_by_domain: dict[str, float],
    ifeval_prompt_strict: float,
    llmjp_avg: float,
) -> dict[str, Any]:
    """Apply the frozen K2 AND gate; return a structured PASS/FAIL verdict.

    Parameters are accuracies / scores in ``[0, 1]`` (not percentage points).
    ``jkb_by_domain`` / ``base_by_domain`` map domain name -> accuracy.
    Domains present only on one side are treated as failures for condition 2.
    """
    cond_overall = jkb_overall >= K2_JKB_OVERALL_MIN
    domain_failures: list[dict[str, Any]] = []
    domains = sorted(set(jkb_by_domain) | set(base_by_domain))
    if not domains:
        cond_domains = False
        domain_failures.append(
            {
                "domain": None,
                "reason": "no_domains",
                "candidate": None,
                "base": None,
                "delta": None,
            }
        )
    else:
        for domain in domains:
            if domain not in jkb_by_domain or domain not in base_by_domain:
                domain_failures.append(
                    {
                        "domain": domain,
                        "reason": "missing_side",
                        "candidate": jkb_by_domain.get(domain),
                        "base": base_by_domain.get(domain),
                        "delta": None,
                    }
                )
                continue
            cand = jkb_by_domain[domain]
            base = base_by_domain[domain]
            delta = cand - base
            # Allow exactly -3.0pt (float-safe): fail only when drop exceeds threshold.
            if delta < -K2_DOMAIN_DROP_MAX - 1e-12:
                domain_failures.append(
                    {
                        "domain": domain,
                        "reason": "drop_exceeds_3pt",
                        "candidate": cand,
                        "base": base,
                        "delta": delta,
                    }
                )
        cond_domains = len(domain_failures) == 0

    cond_ifeval = ifeval_prompt_strict >= K2_IFEVAL_PROMPT_STRICT_MIN
    cond_llmjp = llmjp_avg >= K2_LLMJP_AVG_MIN

    conditions = {
        "jkb_overall_ge_60": {
            "pass": cond_overall,
            "value": jkb_overall,
            "threshold": K2_JKB_OVERALL_MIN,
        },
        "jkb_domains_within_3pt_of_base": {
            "pass": cond_domains,
            "failures": domain_failures,
            "threshold_drop_max": K2_DOMAIN_DROP_MAX,
        },
        "ifeval_prompt_strict_ge_0_920": {
            "pass": cond_ifeval,
            "value": ifeval_prompt_strict,
            "threshold": K2_IFEVAL_PROMPT_STRICT_MIN,
        },
        "llmjp_avg_ge_0_459": {
            "pass": cond_llmjp,
            "value": llmjp_avg,
            "threshold": K2_LLMJP_AVG_MIN,
        },
    }
    passed = all(c["pass"] for c in conditions.values())
    return {
        "pass": passed,
        "conditions": conditions,
        "verdict": "PASS" if passed else "FAIL",
    }
