"""Format-constraint synthesis tests (Issue #105).

Each verifier gets a positive (should synthesize a row) and a negative
(should return ``None``) fixture. ``instruction_verifiers.py`` is imported
as-is (frozen, Issue #104/#105) -- these tests exercise the real verifier
functions, not mocks, so a synthesized row is only accepted if it genuinely
passes.
"""

from __future__ import annotations

import random

from lfm25_ja.data.format_constraints import (
    _count_bullets,
    _extract_keyword_noun,
    _extract_length,
    _extract_qa,
    build_format_constrained_samples,
    synthesize_bullet_count,
    synthesize_char_count,
    synthesize_format_json,
    synthesize_keyword,
    synthesize_polite_form,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_extract_length_counts_nfkc_normalized_chars() -> None:
    assert _extract_length("ａｂｃ") == 3  # fullwidth -> halfwidth, still 3 chars


def test_count_bullets_counts_dash_and_numbered_lines() -> None:
    text = "- 項目1\n・項目2\n3. 項目3\n本文行"
    assert _count_bullets(text) == 3


def test_count_bullets_zero_for_plain_text() -> None:
    assert _count_bullets("ただの文章です。") == 0


def test_extract_keyword_noun_finds_katakana_run() -> None:
    noun = _extract_keyword_noun("東京タワーはランドマークです。")
    assert noun is not None
    assert noun in "東京タワーはランドマークです。"


def test_extract_keyword_noun_returns_none_when_no_candidate() -> None:
    assert _extract_keyword_noun("あいうえお") is None


def test_extract_qa_returns_last_user_and_assistant_turn() -> None:
    messages = [
        {"role": "user", "content": "質問1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "質問2"},
        {"role": "assistant", "content": "回答2"},
    ]
    result = _extract_qa(messages)
    assert result == ("質問2", "回答2")


def test_extract_qa_returns_none_when_last_message_not_assistant() -> None:
    messages = [{"role": "user", "content": "質問"}]
    assert _extract_qa(messages) is None


def test_extract_qa_returns_none_when_no_preceding_user_turn() -> None:
    messages = [{"role": "assistant", "content": "回答"}]
    assert _extract_qa(messages) is None


# ---------------------------------------------------------------------------
# synthesize_char_count
# ---------------------------------------------------------------------------


def _row(question: str, answer: str, origin: str = "ichikara") -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "origin": origin,
    }


def test_synthesize_char_count_accepts_answer_in_range() -> None:
    row = _row("質問", "あ" * 50)
    result = synthesize_char_count(row, random.Random(42))
    assert result is not None
    assert result["verifier"] == "char_count"
    assert "文字以内で答えてください" in result["messages"][0]["content"]
    assert result["messages"][1]["content"] == "あ" * 50


def test_synthesize_char_count_rejects_too_short_answer() -> None:
    row = _row("質問", "短い")
    assert synthesize_char_count(row, random.Random(42)) is None


def test_synthesize_char_count_rejects_too_long_answer() -> None:
    row = _row("質問", "あ" * 400)
    assert synthesize_char_count(row, random.Random(42)) is None


def test_synthesize_char_count_returns_none_for_no_qa() -> None:
    row = {"messages": [{"role": "assistant", "content": "x" * 50}], "origin": "ichikara"}
    assert synthesize_char_count(row, random.Random(42)) is None


# ---------------------------------------------------------------------------
# synthesize_bullet_count
# ---------------------------------------------------------------------------


def test_synthesize_bullet_count_accepts_2_to_5_bullets() -> None:
    row = _row("質問", "- 項目1\n- 項目2\n- 項目3")
    result = synthesize_bullet_count(row, random.Random(42))
    assert result is not None
    assert result["verifier"] == "bullet_count"
    assert "箇条書き3項目で答えてください" in result["messages"][0]["content"]


def test_synthesize_bullet_count_rejects_single_bullet() -> None:
    row = _row("質問", "- 項目1のみ")
    assert synthesize_bullet_count(row, random.Random(42)) is None


def test_synthesize_bullet_count_rejects_too_many_bullets() -> None:
    bullets = "\n".join(f"- 項目{i}" for i in range(6))
    row = _row("質問", bullets)
    assert synthesize_bullet_count(row, random.Random(42)) is None


# ---------------------------------------------------------------------------
# synthesize_format_json
# ---------------------------------------------------------------------------


def test_synthesize_format_json_accepts_mid_length_answer() -> None:
    row = _row("質問", "あ" * 100)
    result = synthesize_format_json(row, random.Random(42))
    assert result is not None
    assert result["verifier"] == "format_json"
    assert '"answer"' in result["messages"][1]["content"]


def test_synthesize_format_json_rejects_too_short_answer() -> None:
    row = _row("質問", "短い回答")
    assert synthesize_format_json(row, random.Random(42)) is None


def test_synthesize_format_json_rejects_too_long_answer() -> None:
    row = _row("質問", "あ" * 300)
    assert synthesize_format_json(row, random.Random(42)) is None


# ---------------------------------------------------------------------------
# synthesize_polite_form
# ---------------------------------------------------------------------------


def test_synthesize_polite_form_accepts_polite_answer() -> None:
    row = _row("質問", "東京は日本の首都です。")
    result = synthesize_polite_form(row, random.Random(42))
    assert result is not None
    assert result["verifier"] == "polite_form"
    assert "敬体" in result["messages"][0]["content"]
    assert result["messages"][1]["content"] == "東京は日本の首都です。"


def test_synthesize_polite_form_rejects_plain_answer() -> None:
    row = _row("質問", "東京は日本の首都だ。")
    assert synthesize_polite_form(row, random.Random(42)) is None


# ---------------------------------------------------------------------------
# synthesize_keyword
# ---------------------------------------------------------------------------


def test_synthesize_keyword_accepts_answer_with_noun_phrase() -> None:
    row = _row("質問", "東京タワーは有名な観光地です。")
    result = synthesize_keyword(row, random.Random(42))
    assert result is not None
    assert result["verifier"] == "keyword"
    assert "を必ず含めてください" in result["messages"][0]["content"]


def test_synthesize_keyword_rejects_answer_without_noun_phrase() -> None:
    row = _row("質問", "あいうえおかきくけこ")
    assert synthesize_keyword(row, random.Random(42)) is None


# ---------------------------------------------------------------------------
# build_format_constrained_samples
# ---------------------------------------------------------------------------


def _diverse_source_rows() -> list[dict]:
    rows = []
    for i in range(5):
        rows.append(_row(f"質問{i}", "あ" * (30 + i), origin="ichikara"))
    for i in range(5):
        rows.append(_row(f"質問b{i}", "- 項目1\n- 項目2\n- 項目3", origin="llm_jp_instruct"))
    for i in range(5):
        rows.append(_row(f"質問c{i}", "東京は日本の首都です。", origin="aya_ja"))
    for i in range(5):
        rows.append(_row(f"質問d{i}", "東京タワーは有名な観光地です。", origin="llm_jp_instruct"))
    return rows


def test_build_format_constrained_samples_respects_targets() -> None:
    rows = _diverse_source_rows()
    targets = {"char_count": 2, "bullet_count": 2, "polite_form": 1, "keyword": 2}
    result = build_format_constrained_samples(rows, targets, seed=42)

    counts: dict[str, int] = {}
    for row in result:
        counts[row["verifier"]] = counts.get(row["verifier"], 0) + 1

    assert counts.get("char_count") == 2
    assert counts.get("bullet_count") == 2
    assert counts.get("polite_form") == 1
    assert counts.get("keyword") == 2


def test_build_format_constrained_samples_polite_form_respects_origin_filter() -> None:
    rows = _diverse_source_rows()
    # Only 5 "aya_ja"-origin rows pass polite_form; llm_jp_instruct/ichikara
    # origins should never surface even though ichikara rows exist too.
    targets = {"polite_form": 10}
    result = build_format_constrained_samples(
        rows, targets, seed=42, polite_form_origins={"aya_ja", "ichikara"}
    )
    assert len(result) == 5
    for row in result:
        assert row["origin"] == "aya_ja"


def test_build_format_constrained_samples_logs_warning_when_target_unreachable(
    caplog,
) -> None:
    rows = _diverse_source_rows()
    # Only the "aya_ja" (です。) and "d"-group llm_jp_instruct (タワー...です。)
    # rows have a genuine sentence-final polite ending; the "ichikara" rows
    # (bare あああ...) and "b"-group llm_jp_instruct rows (bullet lines, no
    # sentences) don't -- so 10/20 pass, well short of an unreachable target.
    targets = {"polite_form": 100}
    with caplog.at_level("WARNING"):
        result = build_format_constrained_samples(rows, targets, seed=42)
    assert len(result) == 10
    assert any("polite_form" in r.message for r in caplog.records)


def test_build_format_constrained_samples_is_deterministic() -> None:
    rows = _diverse_source_rows()
    targets = {"char_count": 2, "keyword": 2}
    a = build_format_constrained_samples(rows, targets, seed=42)
    b = build_format_constrained_samples(rows, targets, seed=42)
    assert a == b


def test_build_format_constrained_samples_empty_targets_returns_empty_list() -> None:
    rows = _diverse_source_rows()
    assert build_format_constrained_samples(rows, {}, seed=42) == []
