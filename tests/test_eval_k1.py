"""Tests for scripts/eval_k1.py (K1 資産再評価集計、Issue #122).

McNemar exact p-value と paired bootstrap CI の正しさ・再現性を検証する。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_k1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_k1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcnemar_no_discordant_returns_one():
    """b=c=0 (すべて一致) は McNemar 差検出できないため p=1.0 を返すこと。"""
    mod = _load_module()
    assert mod._mcnemar_p(0, 0) == 1.0


def test_mcnemar_symmetric_returns_one():
    """b=c で対称なら差なし = p=1.0 が返ること (両側二項検定)。"""
    mod = _load_module()
    assert mod._mcnemar_p(5, 5) == 1.0


def test_mcnemar_extreme_asymmetric_significant():
    """b=10, c=0 (10 問全て model が勝つ) は p が有意水準 0.01 未満になること。"""
    mod = _load_module()
    p = mod._mcnemar_p(10, 0)
    # binom(10, 10, 0.5) * 2 = 2 * (1/1024) ≈ 0.00195
    assert p < 0.01


def test_mcnemar_moderate_asymmetric():
    """b=8, c=2 の two-tailed exact p = 2 * (binom(10,0)+binom(10,1)+binom(10,2))/2^10 ≈ 0.109."""
    mod = _load_module()
    p = mod._mcnemar_p(8, 2)
    assert 0.10 < p < 0.12


def test_bootstrap_ci_zero_diff():
    """完全に同じベクトルなら差の CI が [0, 0] を含み幅がゼロに近い。"""
    mod = _load_module()
    a = [True, False, True, False, True] * 20
    lo, hi = mod._bootstrap_diff_ci(a, a, n_boot=200, seed=0)
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_seeded_reproducible():
    """同 seed なら決定的、seed 変更で違う値になることの sanity check。"""
    mod = _load_module()
    a = [True] * 50 + [False] * 50
    b = [False] * 30 + [True] * 20 + [False] * 50
    lo1, hi1 = mod._bootstrap_diff_ci(a, b, n_boot=200, seed=1)
    lo2, hi2 = mod._bootstrap_diff_ci(a, b, n_boot=200, seed=1)
    assert (lo1, hi1) == (lo2, hi2)
    lo3, hi3 = mod._bootstrap_diff_ci(a, b, n_boot=200, seed=2)
    assert (lo1, hi1) != (lo3, hi3)  # seed が違えば結果もずれる


def test_bootstrap_ci_covers_true_diff():
    """既知の paired 差 20% が bootstrap 95% CI に含まれることを 3σ 以内で確認。"""
    mod = _load_module()
    # a: 60 正解 / 100 問, b: 40 正解 / 100 問, 完全 paired
    a = [True] * 60 + [False] * 40
    b = [True] * 40 + [False] * 60
    lo, hi = mod._bootstrap_diff_ci(a, b, n_boot=2000, seed=42)
    assert lo <= 0.20 <= hi


def test_compare_to_base_identity_case():
    """base と bit-identical なモデルは delta=0, p=1.0 になる。"""
    mod = _load_module()
    ids = [f"q{i}" for i in range(50)]
    correct = [i % 3 == 0 for i in range(50)]
    model_rows = {
        "base": {qid: {"correct": c} for qid, c in zip(ids, correct)},
        "same": {qid: {"correct": c} for qid, c in zip(ids, correct)},
    }
    stats = mod.compare_to_base(model_rows, ids)
    assert stats["same"]["mcnemar_b_win"] == 0
    assert stats["same"]["mcnemar_c_lose"] == 0
    assert stats["same"]["mcnemar_p"] == 1.0
    assert stats["same"]["delta_pct"] == 0.0
