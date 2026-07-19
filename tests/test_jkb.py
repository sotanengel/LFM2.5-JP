"""JKB v1 scorer + I/O tests: load_jkb_jsonl, scoring, aggregate, report (Issue #121)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lfm25_ja.eval.jkb import (
    _MCQ_ANSWER_RE,
    JKB_DIFFICULTIES,
    JKB_DOMAINS,
    aggregate,
    load_jkb_jsonl,
    render_report_markdown,
    score_mcq,
    score_row,
    score_short_answer,
)

# ---------------------------------------------------------------------------
# fixture rows: 2 short_answer (core geo, advanced hist) + 2 mcq (standard
# pol, core lit) -- covers all 3 difficulties and both formats across tests.
# ---------------------------------------------------------------------------


def _valid_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "jkb-geo-core-001",
            "domain": "地理",
            "difficulty": "core",
            "format": "short_answer",
            "prompt": "日本で一番高い山は何ですか?",
            "answers": ["富士山", "富士"],
            "choices": None,
            "correct_choice": None,
            "source_url": "https://ja.wikipedia.org/wiki/富士山",
            "source_quote": "富士山は日本国内で最も高い山である。",
        },
        {
            "id": "jkb-hist-advanced-001",
            "domain": "歴史",
            "difficulty": "advanced",
            "format": "short_answer",
            "prompt": "旧国名「近江」は現在の何県にあたりますか?",
            "answers": ["滋賀"],
            "choices": None,
            "correct_choice": None,
            "source_url": "https://ja.wikipedia.org/wiki/近江国",
            "source_quote": "近江国は現在の滋賀県にあたる。",
        },
        {
            "id": "jkb-pol-standard-001",
            "domain": "政治・制度",
            "difficulty": "standard",
            "format": "mcq",
            "prompt": "日本国憲法が施行された西暦は次のうちどれですか?",
            "answers": None,
            "choices": [
                {"label": "A", "text": "1945"},
                {"label": "B", "text": "1947"},
            ],
            "correct_choice": "B",
            "source_url": "https://ja.wikipedia.org/wiki/日本国憲法",
            "source_quote": "日本国憲法は1947年5月3日に施行された。",
        },
        {
            "id": "jkb-lit-core-001",
            "domain": "文学",
            "difficulty": "core",
            "format": "mcq",
            "prompt": "『源氏物語』の作者は誰ですか?",
            "answers": None,
            "choices": [
                {"label": "A", "text": "紫式部"},
                {"label": "B", "text": "清少納言"},
            ],
            "correct_choice": "A",
            "source_url": "https://ja.wikipedia.org/wiki/源氏物語",
            "source_quote": "『源氏物語』は紫式部によって書かれた。",
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# load_jkb_jsonl
# ---------------------------------------------------------------------------


def test_load_jkb_jsonl_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    _write_jsonl(path, _valid_rows())

    rows = load_jkb_jsonl(path)

    assert len(rows) == 4
    assert {r["id"] for r in rows} == {
        "jkb-geo-core-001",
        "jkb-hist-advanced-001",
        "jkb-pol-standard-001",
        "jkb-lit-core-001",
    }


def test_load_jkb_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    rows = _valid_rows()[:1]
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
        f.write("\n")
        f.write("   \n")

    assert len(load_jkb_jsonl(path)) == 1


def test_load_jkb_jsonl_missing_field_raises_with_id(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    del bad["source_url"]
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="jkb-geo-core-001") as exc_info:
        load_jkb_jsonl(path)
    assert "source_url" in str(exc_info.value)


def test_load_jkb_jsonl_missing_id_falls_back_to_line_number(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    del bad["id"]
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="line 1"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_unknown_domain_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    bad["domain"] = "架空分野"
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="unknown domain"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_unknown_difficulty_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    bad["difficulty"] = "expert"
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="unknown difficulty"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_unknown_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    bad["format"] = "essay"
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="format"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_short_answer_empty_answers_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    bad["answers"] = []
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="short_answer"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_short_answer_missing_answers_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[0]
    bad["answers"] = None
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="short_answer"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_mcq_missing_choices_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[2]
    bad["choices"] = None
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="mcq"):
        load_jkb_jsonl(path)


def test_load_jkb_jsonl_mcq_correct_choice_not_in_labels_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _valid_rows()[2]
    bad["correct_choice"] = "Z"
    _write_jsonl(path, [bad])

    with pytest.raises(ValueError, match="correct_choice"):
        load_jkb_jsonl(path)


# ---------------------------------------------------------------------------
# score_short_answer
# ---------------------------------------------------------------------------


def test_score_short_answer_substring_hit() -> None:
    assert score_short_answer("富士山です。", ["富士山", "富士"]) is True


def test_score_short_answer_no_hit() -> None:
    assert score_short_answer("わかりません。", ["富士山"]) is False


def test_score_short_answer_boundary_choice_list_leak_is_false() -> None:
    # The correct answer word appears only inside a self-generated choice
    # list after the model's real (wrong) first answer -- extract_answer_segment
    # must cut it off, so this must NOT count as a hit.
    text = "岐阜県 A:高尾山 B:立山 C:滋賀県"
    assert score_short_answer(text, ["滋賀"]) is False


# ---------------------------------------------------------------------------
# _MCQ_ANSWER_RE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["A", "A. ", "（A）", " A: ", "A、", "A．"],
)
def test_mcq_answer_re_matches_various_label_formats(text: str) -> None:
    match = _MCQ_ANSWER_RE.match(text)
    assert match is not None
    assert match.group(1) == "A"


def test_mcq_answer_re_does_not_match_fullwidth_letter() -> None:
    assert _MCQ_ANSWER_RE.match("Ａ") is None


# ---------------------------------------------------------------------------
# score_mcq
# ---------------------------------------------------------------------------


def test_score_mcq_leading_label_hit() -> None:
    assert score_mcq(" B. 平成です。", "B") is True


def test_score_mcq_answer_tag_fallback_hit() -> None:
    # No leading label; extract_answer_segment also cuts the "。" sentence
    # boundary before "答え: B" -- fallback must still find it.
    text = "わかりません。答え: B"
    assert score_mcq(text, "B") is True


def test_score_mcq_no_match_is_false() -> None:
    assert score_mcq("さっぱりわかりません", "A") is False


def test_score_mcq_choice_list_leak_picks_answer_tag_not_leading_label() -> None:
    # Pathological continuation that leaked the choice list the runner
    # prepends to the prompt. The leading "A:" is just choice A's label, not
    # an assertion -- only the trailing "答え: B" tag names the real answer.
    text = "A: 昭和\nB: 平成\n答え: B"
    assert score_mcq(text, "B") is True
    assert score_mcq(text, "A") is False


def test_score_mcq_choice_list_leak_correct_choice_a() -> None:
    text = "A: 昭和\nB: 平成\n答え: A"
    assert score_mcq(text, "A") is True
    assert score_mcq(text, "B") is False


# ---------------------------------------------------------------------------
# score_row
# ---------------------------------------------------------------------------


def test_score_row_dispatches_short_answer() -> None:
    row = _valid_rows()[0]
    assert score_row(row, "富士山です。") is True


def test_score_row_dispatches_mcq() -> None:
    row = _valid_rows()[2]
    assert score_row(row, "B") is True


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _aggregate_fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "r1",
            "domain": "地理",
            "difficulty": "core",
            "format": "short_answer",
            "answers": ["富士山"],
        },
        {
            "id": "r2",
            "domain": "地理",
            "difficulty": "advanced",
            "format": "short_answer",
            "answers": ["琵琶湖"],
        },
        {
            "id": "r3",
            "domain": "歴史",
            "difficulty": "core",
            "format": "mcq",
            "choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
            "correct_choice": "B",
        },
    ]


def test_aggregate_counts_and_accuracies_with_missing_id() -> None:
    rows = _aggregate_fixture_rows()
    raw_texts = {"r1": "富士山です。", "r3": "B"}  # r2 missing -> unanswered

    agg = aggregate(rows, raw_texts)

    assert agg["overall"] == {"n": 3, "correct": 2, "accuracy": pytest.approx(2 / 3)}
    assert agg["by_domain"]["地理"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert agg["by_domain"]["歴史"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert agg["by_difficulty"]["core"] == {"n": 2, "correct": 2, "accuracy": 1.0}
    assert agg["by_difficulty"]["advanced"] == {"n": 1, "correct": 0, "accuracy": 0.0}
    assert agg["by_cell"][("地理", "core")] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert agg["by_cell"][("地理", "advanced")] == {"n": 1, "correct": 0, "accuracy": 0.0}
    assert agg["by_cell"][("歴史", "core")] == {"n": 1, "correct": 1, "accuracy": 1.0}

    per_row_by_id = {r["id"]: r for r in agg["per_row"]}
    assert per_row_by_id["r1"]["correct"] is True
    assert per_row_by_id["r2"]["correct"] is False
    assert per_row_by_id["r3"]["correct"] is True


def test_aggregate_empty_rows() -> None:
    agg = aggregate([], {})
    assert agg["overall"] == {"n": 0, "correct": 0, "accuracy": 0.0}
    assert agg["by_domain"] == {}
    assert agg["by_difficulty"] == {}
    assert agg["by_cell"] == {}
    assert agg["per_row"] == []


def test_aggregate_orders_by_domain_in_schema_order() -> None:
    rows = list(reversed(_aggregate_fixture_rows()))
    agg = aggregate(rows, {})
    assert list(agg["by_domain"].keys()) == [
        d for d in JKB_DOMAINS if d in ("地理", "歴史")
    ]


# ---------------------------------------------------------------------------
# render_report_markdown
# ---------------------------------------------------------------------------


def test_render_report_markdown_smoke() -> None:
    rows = _aggregate_fixture_rows()
    raw_texts = {"r1": "富士山です。", "r3": "B"}
    agg = aggregate(rows, raw_texts)

    report = render_report_markdown(agg, "base")

    assert "base" in report
    assert "正答率" in report
    assert "地理" in report
    assert "歴史" in report
    assert "±" in report


def test_render_report_markdown_empty_by_cell_does_not_crash() -> None:
    agg = aggregate([], {})
    report = render_report_markdown(agg, "empty-model")
    assert "empty-model" in report
    assert "±" in report


def test_render_report_markdown_includes_all_difficulties_present() -> None:
    rows = _aggregate_fixture_rows()
    agg = aggregate(rows, {"r1": "富士山です。", "r3": "B"})
    report = render_report_markdown(agg, "base")
    for difficulty in JKB_DIFFICULTIES:
        if difficulty in agg["by_difficulty"]:
            assert difficulty in report
