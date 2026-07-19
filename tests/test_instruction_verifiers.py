"""Table-driven tests for the ifeval_ja rule verifiers (Issue #104)."""

from __future__ import annotations

import pytest

from lfm25_ja.eval.instruction_verifiers import (
    VERIFIERS,
    strip_preamble,
    verify_bullet_count,
    verify_char_count,
    verify_format_json,
    verify_format_markdown_table,
    verify_keyword,
    verify_numeric_only,
    verify_polite_form,
)

# Each case: (verifier_id, response, params, expected_pass)
CASES: list[tuple[str, str, dict, bool]] = [
    # --- char_count ---
    ("char_count", "こんにちは", {"max": 10}, True),
    ("char_count", "こんにちは世界これは長い文章です", {"max": 10}, False),
    ("char_count", "12345", {"min": 5}, True),
    ("char_count", "1234", {"min": 5}, False),
    ("char_count", "12345", {"min": 3, "max": 5}, True),
    ("char_count", "123456", {"min": 3, "max": 5}, False),
    # NFKC normalization: full-width ASCII collapses to half-width before counting.
    ("char_count", "ＡＢＣ", {"max": 3}, True),
    # --- bullet_count ---
    ("bullet_count", "- a\n- b\n- c", {"count": 3}, True),
    ("bullet_count", "- a\n- b", {"count": 3}, False),
    ("bullet_count", "・a\n・b\n・c", {"min": 2, "max": 4}, True),
    ("bullet_count", "・a", {"min": 2, "max": 4}, False),
    ("bullet_count", "・a\n・b\n・c\n・d\n・e", {"min": 2, "max": 4}, False),
    ("bullet_count", "1. a\n2. b\n3. c", {"count": 3}, True),
    ("bullet_count", "* a\n* b", {"count": 2}, True),
    # --- polite_form ---
    ("polite_form", "これはペンです。今日は晴れます。", {"style": "polite"}, True),
    ("polite_form", "これはペンです。今日は晴れる。", {"style": "polite"}, False),
    ("polite_form", "これはペンだ。今日は晴れる。", {"style": "plain"}, True),
    ("polite_form", "これはペンだ。今日は晴れます。", {"style": "plain"}, False),
    # Fable5 negation edge cases (must not be misclassified by naive substring match).
    ("polite_form", "それは正解ではない。", {"style": "plain"}, True),
    ("polite_form", "それは正解ではない。", {"style": "polite"}, False),
    ("polite_form", "それは正解ではないでしょうか。", {"style": "polite"}, True),
    ("polite_form", "それは正解ではないでしょうか。", {"style": "plain"}, False),
    ("polite_form", "明日は雨でしょう。", {"style": "polite"}, True),
    ("polite_form", "明日は雨でしょう。", {"style": "plain"}, False),
    # Bullet lines are excluded from the sentence check.
    ("polite_form", "以下の通りです。\n- 項目です\n- 別の項目だ", {"style": "polite"}, True),
    # Business-letter fragments must be exempted from polite judgment (Issue #104 rescore):
    # header labels, addressees, salutations, and placeholder brackets are non-sentences.
    (
        "polite_form",
        "件名：会議の変更について\n\nA社様\n\nいつもお世話になっております。\n"
        "打ち合わせの日程を変更させていただきたく、ご連絡いたします。\n\n敬具\n[署名]",
        {"style": "polite"},
        True,
    ),
    (
        "polite_form",
        "〇〇部長\n\nお世話になっております。\nよろしくお願いいたします。",
        {"style": "polite"},
        True,
    ),
    ("polite_form", "お客様各位\n\nこの度はありがとうございます。", {"style": "polite"}, True),
    ("polite_form", "拝啓\n\nお世話になっております。\n敬具", {"style": "polite"}, True),
    # But polite-exempt lines should NOT let a genuinely-plain body sneak through:
    ("polite_form", "件名：連絡事項\n\nA社様\n\n本文はここに書く。", {"style": "polite"}, False),
    # v1.1 (Issue #117 rescore): でした/ましょう/ませ are 敬体 sentence endings
    # (past/volitional/imperative forms of です・ます) -- their absence
    # false-flagged genuinely polite dpo-001 outputs.
    ("polite_form", "誠に申し訳ございませんでした。", {"style": "polite"}, True),
    ("polite_form", "共に新たな一歩を歩んでまいりましょう。", {"style": "polite"}, True),
    ("polite_form", "ぜひご覧くださいませ。", {"style": "polite"}, True),
    ("polite_form", "昨日は雨でした。", {"style": "plain"}, False),
    ("polite_form", "そろそろ行きましょう。", {"style": "plain"}, False),
    # v1.1: hiragana-less lines (signatures / role names like 幹事 〇〇) have
    # no predicate to judge -- exempt for polite style.
    ("polite_form", "ご確認をお願いいたします。\n幹事　〇〇", {"style": "polite"}, True),
    ("polite_form", "ご出席のご連絡をお待ちしております。\n担当：山田", {"style": "polite"}, True),
    # --- keyword ---
    ("keyword", "本日は東京で開催します。", {"include": ["東京"]}, True),
    ("keyword", "本日は大阪で開催します。", {"include": ["東京"]}, False),
    ("keyword", "本日は東京で開催します。", {"exclude": ["中止"]}, True),
    ("keyword", "本日は中止になりました。", {"exclude": ["中止"]}, False),
    (
        "keyword",
        "東京で開催、中止ではありません。",
        {"include": ["東京"], "exclude": ["延期"]},
        True,
    ),
    # NFKC normalization: full-width keyword matches half-width response text.
    ("keyword", "ABC123", {"include": ["ＡＢＣ"]}, True),
    # --- format_json ---
    ("format_json", '{"a": 1}', {}, True),
    ("format_json", '```json\n{"a": 1}\n```', {}, True),
    ("format_json", '```\n{"a": 1}\n```', {}, True),
    ("format_json", "{a: 1}", {}, False),
    ("format_json", "not json at all", {}, False),
    # --- format_markdown_table ---
    ("format_markdown_table", "| a | b |\n|---|---|\n| 1 | 2 |", {}, True),
    ("format_markdown_table", "| a | b |\n| 1 | 2 |", {}, False),
    ("format_markdown_table", "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", {"min_rows": 2}, True),
    ("format_markdown_table", "| a | b |\n|---|---|\n| 1 | 2 |", {"min_rows": 2}, False),
    (
        "format_markdown_table",
        "説明文\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n以上です。",
        {},
        True,
    ),
    # --- numeric_only ---
    ("numeric_only", "123", {}, True),
    ("numeric_only", "-12.5", {}, True),
    ("numeric_only", "12a", {}, False),
    # Raw (non-NFKC) check: full-width digits must fail, unlike char_count.
    ("numeric_only", "１２３", {}, False),
    ("numeric_only", "100円", {"allow_units": True}, True),
    ("numeric_only", "100円", {"allow_units": False}, False),
    ("numeric_only", "abc", {"allow_units": True}, False),
]


@pytest.mark.parametrize("verifier_id,response,params,expected", CASES)
def test_verifier_table(verifier_id, response, params, expected):
    passed, reason = VERIFIERS[verifier_id](response, params)
    assert passed is expected, f"reason={reason!r}"
    if passed:
        assert reason == ""
    else:
        assert reason != ""


def test_all_verifier_ids_referenced_in_registry():
    used_ids = {c[0] for c in CASES}
    assert used_ids == set(VERIFIERS.keys())


def test_char_count_requires_min_or_max():
    with pytest.raises(ValueError):
        verify_char_count("hello", {})


def test_bullet_count_requires_count_or_range():
    with pytest.raises(ValueError):
        verify_bullet_count("- a", {})


def test_polite_form_requires_valid_style():
    with pytest.raises(ValueError):
        verify_polite_form("これはペンです。", {"style": "invalid"})


def test_polite_form_empty_sentences_fail():
    passed, reason = verify_polite_form("", {"style": "polite"})
    assert passed is False
    assert reason != ""


def test_keyword_requires_include_or_exclude():
    with pytest.raises(ValueError):
        verify_keyword("hello", {})


def test_format_json_reason_mentions_parse_error():
    passed, reason = verify_format_json("{a: 1}", {})
    assert passed is False
    assert "JSON" in reason


def test_format_markdown_table_reason_when_missing():
    passed, reason = verify_format_markdown_table("no table here", {})
    assert passed is False
    assert reason != ""


def test_numeric_only_reason_when_invalid():
    passed, reason = verify_numeric_only("abc", {})
    assert passed is False
    assert reason != ""


# --- strip_preamble (used for loose scoring) ---


@pytest.mark.parametrize(
    "response,expected",
    [
        ("はい、承知しました。\n123", "123"),
        ("承知いたしました。\n本文です。", "本文です。"),
        ("回答: 42", "42"),
        ("```\n123\n```", "123"),
        ("```json\n{\"a\": 1}\n```", '{"a": 1}'),
        ("前置きなしの本文です。", "前置きなしの本文です。"),
    ],
)
def test_strip_preamble(response, expected):
    assert strip_preamble(response) == expected
