"""Schema check for the ifeval_ja prompt dataset (Issue #104)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lfm25_ja.eval.instruction_verifiers import VERIFIERS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "ifeval_ja" / "prompts.jsonl"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "ifeval_ja_sample.jsonl"

REQUIRED_FIELDS = {"id", "prompt", "instruction_id_list", "kwargs", "category"}
VALID_CATEGORIES = {"依頼", "質問", "要約", "敬語"}
ID_RE = re.compile(r"^ifja-\d{3}$")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _check_rows(rows: list[dict]) -> None:
    assert rows, "dataset must not be empty"
    for row in rows:
        missing = REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"row {row.get('id')} missing fields: {missing}"

        assert ID_RE.match(row["id"]), f"id {row['id']!r} does not match ifja-\\d{{3}}"

        assert row["category"] in VALID_CATEGORIES, (
            f"row {row['id']} has invalid category: {row['category']!r}"
        )

        instruction_ids = row["instruction_id_list"]
        assert isinstance(instruction_ids, list) and instruction_ids, (
            f"row {row['id']} instruction_id_list must be a non-empty list"
        )
        for instruction_id in instruction_ids:
            assert instruction_id in VERIFIERS, (
                f"row {row['id']} references unregistered verifier {instruction_id!r}"
            )

        kwargs = row["kwargs"]
        assert isinstance(kwargs, dict), f"row {row['id']} kwargs must be a dict"
        for instruction_id in instruction_ids:
            # format_json takes no params, so an empty dict is acceptable, but
            # the key itself must still be present for every listed instruction.
            assert instruction_id in kwargs, (
                f"row {row['id']} kwargs missing entry for {instruction_id!r}"
            )
            assert isinstance(kwargs[instruction_id], dict), (
                f"row {row['id']} kwargs[{instruction_id!r}] must be a dict"
            )


def test_sample_fixture_schema():
    rows = _load_jsonl(FIXTURE_PATH)
    _check_rows(rows)


def test_dataset_schema():
    if not DATASET_PATH.exists():
        pytest.skip("dataset not yet added")
    rows = _load_jsonl(DATASET_PATH)
    _check_rows(rows)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate ids in dataset"
