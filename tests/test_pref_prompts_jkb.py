"""Tests for K3 JKB train-split preference prompt pool (Issue #124 / #145)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lfm25_ja.data.pref_prompts_jkb import (
    build_jkb_prompt_pool,
    check_jkb_pool_non_duplication,
    jkb_row_to_pool_row,
)
from lfm25_ja.eval.jkb import load_jkb_jsonl

_REPO = Path(__file__).resolve().parents[1]
_TRAIN = _REPO / "datasets/eval/jkb/train.jsonl"
_EVAL = _REPO / "datasets/eval/jkb/eval.jsonl"


def test_jkb_row_to_pool_row_has_jkb_fact_category() -> None:
    rows = load_jkb_jsonl(_TRAIN)
    pool_row = jkb_row_to_pool_row(rows[0], index=1)
    assert pool_row["category"] == "jkb_fact"
    assert pool_row["id"] == "k3pref-00001"
    assert "質問:" in pool_row["prompt"]
    assert pool_row["constraint_detail"]["jkb_id"] == rows[0]["id"]


def test_check_jkb_pool_non_duplication_rejects_eval_id_overlap() -> None:
    train_rows = load_jkb_jsonl(_TRAIN)
    eval_rows = load_jkb_jsonl(_EVAL)
    pool = [jkb_row_to_pool_row(train_rows[0], index=1)]
    pool[0]["constraint_detail"]["jkb_id"] = eval_rows[0]["id"]
    with pytest.raises(ValueError, match="eval id overlap"):
        check_jkb_pool_non_duplication(pool, eval_rows)


def test_build_jkb_prompt_pool_writes_all_train_rows(tmp_path: Path) -> None:
    out = tmp_path / "pool.jsonl"
    stats = build_jkb_prompt_pool(
        train_path=_TRAIN,
        eval_path=_EVAL,
        output_path=out,
    )
    assert stats["total"] == len(load_jkb_jsonl(_TRAIN))
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written) == stats["total"]
    train_ids = {r["id"] for r in load_jkb_jsonl(_TRAIN)}
    eval_ids = {r["id"] for r in load_jkb_jsonl(_EVAL)}
    pool_jkb_ids = {r["constraint_detail"]["jkb_id"] for r in written}
    assert pool_jkb_ids == train_ids
    assert pool_jkb_ids.isdisjoint(eval_ids)
