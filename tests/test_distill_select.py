"""sft-005 distillation-CSV selection + mix pipeline tests (Issue #109).

Verifier selection reuses the frozen ``lfm25_ja.eval.instruction_verifiers``
functions as-is (char_count / bullet_count / polite_form / keyword /
format_json) -- these tests exercise the real verifier functions plus the
distill_select-local additions (tight length margin, format_json detail
check, paragraph_count / start_word / forbidden_word / no_constraint), not
mocks, so acceptance only happens if the row genuinely passes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lfm25_ja.data.distill_select import (
    apply_degenerate_filters,
    check_eval_non_duplication,
    check_response_length_guard,
    compute_topic_stats,
    mix_distill_with_ichikara,
    prepare_distill_mix,
    read_distill_csv,
    render_distill_stats_report,
    row_to_messages,
    select_row,
    verify_format_json_detail,
    verify_no_constraint,
    verify_paragraph_count,
    verify_start_word,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV_PATH = REPO_ROOT / "datasets" / "sft" / "sft005_distill_candidateB_prompts.csv"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(
    category: str,
    detail: dict,
    prompt: str = "質問",
    response: str = "回答です。",
    topic: str = "topic",
    row_id: str = "distill-00001",
    instruction_id_list: str = "",
) -> dict:
    return {
        "id": row_id,
        "category": category,
        "instruction_id_list": instruction_id_list or category,
        "constraint_detail": json.dumps(detail, ensure_ascii=False),
        "detail": detail,
        "topic": topic,
        "prompt": prompt,
        "response": response,
        "response_char_count": str(len(response)),
    }


def _filler(n: int, offset: int = 0) -> str:
    """``n``-char filler text made of sequential CJK Unified Ideograph code
    points (starting at an ``offset``-dependent point) -- long enough to
    exercise char-count constraints while guaranteed not to trip the
    degenerate-repetition heuristic (``_has_repetition``), unlike a
    same-character run like ``"あ" * n``."""
    start = 0x4E00 + offset * 503
    return "".join(chr(start + (i % 4000)) for i in range(n))


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "distill.csv"
    fieldnames = [
        "id",
        "category",
        "instruction_id_list",
        "constraint_detail",
        "topic",
        "prompt",
        "response",
        "response_char_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    return path


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def test_read_distill_csv_parses_constraint_detail_json(tmp_path: Path) -> None:
    rows = [
        _row("char_count", {"max": 220, "band": "60-90%"}, response="あ" * 150),
        _row("no_constraint", {}, response="あ" * 40, row_id="distill-00002"),
    ]
    csv_path = _write_csv(tmp_path, rows)

    parsed = read_distill_csv(csv_path)

    assert len(parsed) == 2
    assert parsed[0]["detail"] == {"max": 220, "band": "60-90%"}
    assert parsed[1]["detail"] == {}


def test_read_distill_csv_handles_quoted_newlines_in_response(tmp_path: Path) -> None:
    response_with_newline = "1行目\n\n2行目"
    rows = [_row("paragraph_count", {"count": 2}, response=response_with_newline)]
    csv_path = _write_csv(tmp_path, rows)

    parsed = read_distill_csv(csv_path)

    assert parsed[0]["response"] == response_with_newline


# ---------------------------------------------------------------------------
# select_row: category-specific verifier selection (pass/reject)
# ---------------------------------------------------------------------------


def test_select_row_char_count_accepts_within_tight_band() -> None:
    # 70% of max=100 -> 70 chars, inside 60-90%.
    row = _row("char_count", {"max": 100}, response="あ" * 70)
    ok, _ = select_row(row)
    assert ok


def test_select_row_char_count_rejects_when_below_tight_band_though_verifier_passes() -> None:
    # 50 chars <= max=100 (verify_char_count passes) but below 60% band.
    row = _row("char_count", {"max": 100}, response="あ" * 50)
    ok, reason = select_row(row)
    assert not ok
    assert reason


def test_select_row_char_count_rejects_when_verifier_itself_fails() -> None:
    row = _row("char_count", {"max": 100}, response="あ" * 150)
    ok, _ = select_row(row)
    assert not ok


def test_select_row_char_count_rejects_when_max_collides_with_eval_values() -> None:
    row = _row("char_count", {"max": 100}, response="あ" * 70)
    ok, reason = select_row(row, eval_char_values={100})
    assert not ok
    assert "100" in reason


def test_select_row_char_count_respects_custom_tight_margin_band() -> None:
    # 50% of max=100 fails the default 60-90% band but passes an explicit
    # wider 40-90% band -- config-driven tight_margin (configs/data/mix_005.yaml)
    # must actually reach select_row, not just be documented.
    row = _row("char_count", {"max": 100}, response="あ" * 50)
    assert not select_row(row)[0]
    assert select_row(row, tight_margin_band=(0.40, 0.90))[0]


def test_select_row_bullet_count_accepts_matching_count() -> None:
    row = _row("bullet_count", {"count": 3}, response="- 項目1\n- 項目2\n- 項目3")
    ok, _ = select_row(row)
    assert ok


def test_select_row_bullet_count_rejects_mismatched_count() -> None:
    row = _row("bullet_count", {"count": 3}, response="- 項目1\n- 項目2")
    ok, _ = select_row(row)
    assert not ok


def test_select_row_format_json_accepts_matching_keys_and_count() -> None:
    payload = json.dumps(
        [{"名称": "A", "分類": "X", "説明": "d1"}, {"名称": "B", "分類": "Y", "説明": "d2"}],
        ensure_ascii=False,
    )
    row = _row("format_json", {"keys": ["名称", "分類", "説明"], "count": 2}, response=payload)
    ok, _ = select_row(row)
    assert ok


def test_select_row_format_json_rejects_missing_key() -> None:
    payload = json.dumps([{"名称": "A", "説明": "d1"}], ensure_ascii=False)
    row = _row("format_json", {"keys": ["名称", "分類"], "count": 1}, response=payload)
    ok, _ = select_row(row)
    assert not ok


def test_select_row_format_json_rejects_count_mismatch() -> None:
    payload = json.dumps([{"名称": "A"}], ensure_ascii=False)
    row = _row("format_json", {"keys": ["名称"], "count": 2}, response=payload)
    ok, _ = select_row(row)
    assert not ok


def test_select_row_keyword_include_accepts_when_keyword_present() -> None:
    row = _row(
        "keyword_include",
        {"include": ["現像"]},
        response="フィルムは現像するまで結果が分かりません。",
    )
    ok, _ = select_row(row)
    assert ok


def test_select_row_keyword_include_rejects_when_keyword_missing() -> None:
    row = _row("keyword_include", {"include": ["現像"]}, response="デジタルは便利です。")
    ok, _ = select_row(row)
    assert not ok


def test_select_row_paragraph_count_accepts_matching_blank_line_blocks() -> None:
    row = _row(
        "paragraph_count", {"count": 2}, response="1段落目の文章です。\n\n2段落目の文章です。"
    )
    ok, _ = select_row(row)
    assert ok


def test_select_row_paragraph_count_rejects_mismatched_blocks() -> None:
    row = _row("paragraph_count", {"count": 3}, response="1段落目です。\n\n2段落目です。")
    ok, _ = select_row(row)
    assert not ok


def test_select_row_forbidden_word_accepts_when_word_absent() -> None:
    row = _row("forbidden_word", {"exclude": ["インターネット"]}, response="波に乗る技術です。")
    ok, _ = select_row(row)
    assert ok


def test_select_row_forbidden_word_rejects_when_word_present() -> None:
    row = _row(
        "forbidden_word", {"exclude": ["インターネット"]}, response="インターネットで調べます。"
    )
    ok, _ = select_row(row)
    assert not ok


def test_select_row_start_word_accepts_matching_prefix() -> None:
    row = _row("start_word", {"start": "端的に言えば、"}, response="端的に言えば、そのとおりです。")
    ok, _ = select_row(row)
    assert ok


def test_select_row_start_word_rejects_non_matching_prefix() -> None:
    row = _row("start_word", {"start": "端的に言えば、"}, response="つまり、そのとおりです。")
    ok, _ = select_row(row)
    assert not ok


def test_select_row_compound_accepts_char_count_and_keyword() -> None:
    # max=100, band 60-90% -> 70 chars, and includes required keyword.
    response = "呼吸を意識します。" + "あ" * 61
    row = _row("compound", {"max": 100, "include": ["呼吸"]}, response=response)
    ok, _ = select_row(row)
    assert ok


def test_select_row_compound_rejects_when_keyword_missing() -> None:
    response = "あ" * 70
    row = _row("compound", {"max": 100, "include": ["呼吸"]}, response=response)
    ok, _ = select_row(row)
    assert not ok


def test_select_row_polite_form_accepts_polite_answer() -> None:
    row = _row("polite_form", {"style": "polite"}, response="東京は日本の首都です。")
    ok, _ = select_row(row)
    assert ok


def test_select_row_polite_form_rejects_plain_answer() -> None:
    row = _row("polite_form", {"style": "polite"}, response="東京は日本の首都だ。")
    ok, _ = select_row(row)
    assert not ok


def test_select_row_no_constraint_accepts_mid_length_answer() -> None:
    row = _row("no_constraint", {}, response="あ" * 100)
    ok, _ = select_row(row)
    assert ok


def test_select_row_no_constraint_rejects_too_short_answer() -> None:
    row = _row("no_constraint", {}, response="短い")
    ok, _ = select_row(row)
    assert not ok


def test_select_row_no_constraint_rejects_too_long_answer() -> None:
    row = _row("no_constraint", {}, response="あ" * 400)
    ok, _ = select_row(row)
    assert not ok


def test_select_row_unknown_category_is_rejected() -> None:
    row = _row("mystery_category", {}, response="あ" * 100)
    ok, reason = select_row(row)
    assert not ok
    assert reason


# ---------------------------------------------------------------------------
# tight length margin boundary (60-90% of max)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0.59, False),
        (0.60, True),
        (0.75, True),
        (0.90, True),
        (0.91, False),
    ],
)
def test_char_count_tight_margin_boundary(ratio: float, expected: bool) -> None:
    max_chars = 200
    length = round(max_chars * ratio)
    row = _row("char_count", {"max": max_chars}, response="あ" * length)
    ok, _ = select_row(row)
    assert ok == expected


# ---------------------------------------------------------------------------
# in-module verifiers used directly
# ---------------------------------------------------------------------------


def test_verify_paragraph_count_splits_on_blank_lines() -> None:
    ok, _ = verify_paragraph_count("一段落目。\n\n二段落目。\n\n三段落目。", {"count": 3})
    assert ok


def test_verify_paragraph_count_rejects_wrong_count() -> None:
    ok, _ = verify_paragraph_count("一段落目。\n\n二段落目。", {"count": 3})
    assert not ok


def test_verify_start_word_strips_before_matching() -> None:
    ok, _ = verify_start_word("  結論として、以上です。", {"start": "結論として、"})
    assert ok


def test_verify_no_constraint_default_bounds() -> None:
    assert verify_no_constraint("あ" * 20, {})[0]
    assert verify_no_constraint("あ" * 300, {})[0]
    assert not verify_no_constraint("あ" * 19, {})[0]
    assert not verify_no_constraint("あ" * 301, {})[0]


def test_verify_format_json_detail_accepts_when_no_keys_specified() -> None:
    payload = json.dumps([{"a": 1}, {"a": 2}], ensure_ascii=False)
    ok, _ = verify_format_json_detail(payload, {"count": 2})
    assert ok


def test_verify_format_json_detail_rejects_non_list_payload() -> None:
    payload = json.dumps({"a": 1}, ensure_ascii=False)
    ok, _ = verify_format_json_detail(payload, {"count": 1})
    assert not ok


def test_verify_format_json_detail_accepts_code_fenced_json() -> None:
    # The base model habitually wraps JSON in a ```json fence. The frozen
    # eval verifier (verify_format_json) extracts the fenced payload; the
    # detail check must judge the same payload, not the raw fenced text
    # (Issue #117 Phase V: this bug zeroed out all 466 format_json prompts).
    inner = json.dumps(
        [{"名称": "a", "分類": "b", "説明": "c"}, {"名称": "d", "分類": "e", "説明": "f"}],
        ensure_ascii=False,
    )
    payload = f"```json\n{inner}\n```"
    ok, msg = verify_format_json_detail(payload, {"keys": ["名称", "分類", "説明"], "count": 2})
    assert ok, msg


def test_verify_format_json_detail_fenced_json_still_checks_keys() -> None:
    inner = json.dumps([{"name": "a"}], ensure_ascii=False)
    payload = f"```json\n{inner}\n```"
    ok, msg = verify_format_json_detail(payload, {"keys": ["名称"], "count": 1})
    assert not ok
    assert "名称" in msg


# ---------------------------------------------------------------------------
# degenerate filters
# ---------------------------------------------------------------------------


def test_apply_degenerate_filters_drops_too_short_response() -> None:
    rows = [_row("no_constraint", {}, response="短い", row_id="a")]
    kept, rejects = apply_degenerate_filters(rows)
    assert kept == []
    assert rejects["too_short"] == 1


def test_apply_degenerate_filters_drops_repeated_line() -> None:
    response = "\n".join(["同じ行です。"] * 3 + ["あ" * 30])
    rows = [_row("no_constraint", {}, response=response, row_id="a")]
    kept, rejects = apply_degenerate_filters(rows)
    assert kept == []
    assert rejects["repetition"] == 1


def test_apply_degenerate_filters_exempts_format_json_from_repetition_check() -> None:
    # Structurally-repeated JSON key substrings (", "分類": ") must not be
    # treated as degenerate repetition -- validated against the real CSV
    # (271/466 format_json rows would otherwise be false-positive-flagged).
    payload = json.dumps(
        [{"名称": f"項目{i}", "分類": "X", "説明": "d"} for i in range(5)], ensure_ascii=False
    )
    rows = [_row("format_json", {"keys": ["名称"], "count": 5}, response=payload, row_id="a")]
    kept, rejects = apply_degenerate_filters(rows)
    assert len(kept) == 1
    assert rejects["repetition"] == 0


def test_apply_degenerate_filters_drops_exact_prompt_response_duplicate() -> None:
    response = _filler(30)
    rows = [
        _row("no_constraint", {}, prompt="質問A", response=response, row_id="a"),
        _row("no_constraint", {}, prompt="質問A", response=response, row_id="b"),
    ]
    kept, rejects = apply_degenerate_filters(rows)
    assert len(kept) == 1
    assert rejects["prompt_response_dup"] == 1


def test_apply_degenerate_filters_drops_same_category_detail_response_duplicate() -> None:
    response = _filler(70)
    rows = [
        _row(
            "char_count",
            {"max": 100},
            prompt="質問A",
            response=response,
            row_id="a",
        ),
        _row(
            "char_count",
            {"max": 100},
            prompt="質問B",  # different prompt, same category+detail+response
            response=response,
            row_id="b",
        ),
    ]
    kept, rejects = apply_degenerate_filters(rows)
    assert len(kept) == 1
    assert rejects["category_detail_dup"] == 1


def test_apply_degenerate_filters_keeps_prefix_related_responses_across_topics() -> None:
    # Same-topic prefix relationships across *different* constraint_detail
    # values are intended training signal (Issue #109 spec), not duplicates.
    prefix = _filler(65)
    rows = [
        _row("char_count", {"max": 100}, response=prefix, row_id="a", topic="植物"),
        _row(
            "char_count",
            {"max": 200},
            response=prefix + _filler(65, offset=100),
            row_id="b",
            topic="植物",
        ),
    ]
    kept, _ = apply_degenerate_filters(rows)
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# response length guard (hard gate)
# ---------------------------------------------------------------------------


def test_check_response_length_guard_passes_within_band() -> None:
    rows = [_row("no_constraint", {}, response="あ" * 176, row_id=str(i)) for i in range(10)]
    stats = check_response_length_guard(rows, base_mean=176, tolerance=0.20)
    assert stats["within_band"] is True
    assert stats["mean"] == pytest.approx(176.0)


def test_check_response_length_guard_raises_when_mean_too_low() -> None:
    rows = [_row("no_constraint", {}, response="あ" * 50, row_id=str(i)) for i in range(10)]
    with pytest.raises(ValueError):
        check_response_length_guard(rows, base_mean=176, tolerance=0.20)


def test_check_response_length_guard_raises_when_mean_too_high() -> None:
    rows = [_row("no_constraint", {}, response="あ" * 400, row_id=str(i)) for i in range(10)]
    with pytest.raises(ValueError):
        check_response_length_guard(rows, base_mean=176, tolerance=0.20)


def test_check_response_length_guard_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        check_response_length_guard([], base_mean=176, tolerance=0.20)


# ---------------------------------------------------------------------------
# eval non-duplication assertion (hard gate)
# ---------------------------------------------------------------------------


def _eval_prompts(char_count_values: list[int], prompt_texts: list[str]) -> list[dict]:
    prompts = []
    for i, mx in enumerate(char_count_values):
        prompts.append(
            {
                "id": f"ifja-{i}",
                "prompt": f"prompt {i}",
                "instruction_id_list": ["char_count"],
                "kwargs": {"char_count": {"max": mx}},
                "category": "依頼",
            }
        )
    for i, text in enumerate(prompt_texts):
        prompts.append(
            {
                "id": f"ifja-text-{i}",
                "prompt": text,
                "instruction_id_list": [],
                "kwargs": {},
                "category": "依頼",
            }
        )
    return prompts


def test_check_eval_non_duplication_passes_when_clean() -> None:
    rows = [_row("char_count", {"max": 100}, response="あ" * 70, topic="キャンプ")]
    eval_prompts = _eval_prompts([50, 60], ["別のプロンプトです。"])
    result = check_eval_non_duplication(rows, eval_prompts)
    assert result["value_overlap"] == []
    assert result["topic_hits"] == []


def test_check_eval_non_duplication_raises_on_char_count_value_collision() -> None:
    rows = [_row("char_count", {"max": 100}, response="あ" * 70)]
    eval_prompts = _eval_prompts([100], [])
    with pytest.raises(ValueError):
        check_eval_non_duplication(rows, eval_prompts)


def test_check_eval_non_duplication_raises_on_topic_substring_in_eval_prompt() -> None:
    rows = [_row("no_constraint", {}, response="あ" * 100, topic="家庭菜園")]
    eval_prompts = _eval_prompts([], ["家庭菜園についての質問です。"])
    with pytest.raises(ValueError):
        check_eval_non_duplication(rows, eval_prompts)


# ---------------------------------------------------------------------------
# topic reuse stats
# ---------------------------------------------------------------------------


def test_compute_topic_stats_counts_unique_and_reused_topics() -> None:
    rows = [
        _row("no_constraint", {}, topic="A", row_id="1"),
        _row("no_constraint", {}, topic="A", row_id="2"),
        _row("no_constraint", {}, topic="B", row_id="3"),
    ]
    stats = compute_topic_stats(rows)
    assert stats["unique_topics"] == 2
    assert stats["topics_reused"] == 1


# ---------------------------------------------------------------------------
# messages conversion
# ---------------------------------------------------------------------------


def test_row_to_messages_builds_chat_format() -> None:
    row = _row("no_constraint", {}, prompt="質問文", response="回答文")
    result = row_to_messages(row)
    assert result == {
        "messages": [
            {"role": "user", "content": "質問文"},
            {"role": "assistant", "content": "回答文"},
        ]
    }


# ---------------------------------------------------------------------------
# shuffle reproducibility
# ---------------------------------------------------------------------------


def test_mix_distill_with_ichikara_is_deterministic_for_same_seed() -> None:
    distill_rows = [
        row_to_messages(
            _row("no_constraint", {}, prompt=f"q{i}", response=f"a{i}" * 10, row_id=str(i))
        )
        for i in range(5)
    ]
    ichikara_rows = [
        {
            "messages": [
                {"role": "user", "content": f"i{i}"},
                {"role": "assistant", "content": f"ia{i}"},
            ]
        }
        for i in range(5)
    ]
    a = mix_distill_with_ichikara(distill_rows, ichikara_rows, seed=42)
    b = mix_distill_with_ichikara(distill_rows, ichikara_rows, seed=42)
    assert a == b
    assert len(a) == 10


def test_mix_distill_with_ichikara_differs_for_different_seed() -> None:
    distill_rows = [
        row_to_messages(
            _row("no_constraint", {}, prompt=f"q{i}", response=f"a{i}" * 10, row_id=str(i))
        )
        for i in range(8)
    ]
    ichikara_rows = [
        {
            "messages": [
                {"role": "user", "content": f"i{i}"},
                {"role": "assistant", "content": f"ia{i}"},
            ]
        }
        for i in range(8)
    ]
    a = mix_distill_with_ichikara(distill_rows, ichikara_rows, seed=42)
    b = mix_distill_with_ichikara(distill_rows, ichikara_rows, seed=1)
    assert a != b


# ---------------------------------------------------------------------------
# stats report rendering
# ---------------------------------------------------------------------------


def test_render_distill_stats_report_contains_key_sections() -> None:
    stats = {
        "seed": 42,
        "total": 100,
        "distill_accepted": 50,
        "distill_total_csv": 60,
        "category_totals": {"char_count": 10},
        "reject_reasons": {"char_count": {"タイトマージン": 2}},
        "degenerate_rejects": {
            "too_short": 0,
            "repetition": 0,
            "prompt_response_dup": 0,
            "category_detail_dup": 1,
        },
        "length_guard": {
            "mean": 150.0,
            "median": 148,
            "min": 54,
            "max": 300,
            "lower_bound": 140.8,
            "upper_bound": 211.2,
            "base_mean": 176,
            "tolerance": 0.20,
            "within_band": True,
        },
        "eval_non_duplication": {
            "eval_char_values": [50],
            "distill_max_values": [90],
            "value_overlap": [],
            "topic_hits": [],
        },
        "topic_stats": {"unique_topics": 5, "topics_reused": 3, "max_reuse": [("A", 3)]},
        "ichikara_count": 50,
        "mix_response_length": {"mean": 200.0},
        "ichikara_response_length": {"mean": 284.0},
        "output_path": "data/processed/sft/mix_005.jsonl",
        "csv_md5": "abc123",
    }
    report = render_distill_stats_report(stats)
    assert "sft-005" in report
    assert "176" in report
    assert "abc123" in report
    assert "前セッション成果物" in report
    assert "gen.py" in report
    assert "Qwen3-Swallow" in report


# ---------------------------------------------------------------------------
# end-to-end orchestration (small synthetic fixtures)
# ---------------------------------------------------------------------------


def test_prepare_distill_mix_end_to_end_smoke(tmp_path: Path) -> None:
    csv_rows = []
    for i in range(6):
        csv_rows.append(
            _row(
                "char_count",
                {"max": 250},
                prompt=f"質問{i}",
                response=_filler(180, offset=i),  # 72% of 250 -> inside 60-90% band
                topic=f"topic{i}",
                row_id=f"distill-{i:05d}",
            )
        )
    for i in range(6, 10):
        csv_rows.append(
            _row(
                "no_constraint",
                {},
                prompt=f"質問{i}",
                response=_filler(176, offset=i),
                topic=f"topic{i}",
                row_id=f"distill-{i:05d}",
            )
        )
    csv_path = _write_csv(tmp_path, csv_rows)

    ichikara_path = tmp_path / "ichikara.jsonl"
    with ichikara_path.open("w", encoding="utf-8") as f:
        for i in range(4):
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"ichikara質問{i}"},
                            {"role": "assistant", "content": "あ" * 284},
                        ]
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")

    eval_prompts_path = tmp_path / "eval_prompts.jsonl"
    with eval_prompts_path.open("w", encoding="utf-8") as f:
        for row in _eval_prompts([50, 60], ["まったく別のプロンプトです。"]):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    config = {
        "mix": {
            "seed": 42,
            "source_csv": str(csv_path),
            "ichikara_path": str(ichikara_path),
            "eval_prompts_path": str(eval_prompts_path),
            "output_path": str(tmp_path / "mix_005.jsonl"),
            "stats_report": str(tmp_path / "phase3_sft005_distill_stats.md"),
            "length_guard": {"base_mean_chars": 176, "tolerance": 0.20},
        }
    }
    import yaml

    config_path = tmp_path / "mix_005.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    stats = prepare_distill_mix(config_path)

    assert stats["distill_accepted"] == 10
    assert stats["ichikara_count"] == 4
    assert stats["total"] == 14

    output_path = Path(stats["output_path"])
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 14
    for line in lines:
        row = json.loads(line)
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][-1]["role"] == "assistant"

    report_path = Path(stats["report_path"])
    assert report_path.exists()


def test_prepare_distill_mix_raises_when_length_guard_fails(tmp_path: Path) -> None:
    csv_rows = [
        _row(
            "no_constraint",
            {},
            prompt=f"質問{i}",
            response=_filler(30, offset=i),  # way below 176*0.8
            topic=f"topic{i}",
            row_id=f"distill-{i:05d}",
        )
        for i in range(6)
    ]
    csv_path = _write_csv(tmp_path, csv_rows)

    ichikara_path = tmp_path / "ichikara.jsonl"
    ichikara_path.write_text("", encoding="utf-8")

    eval_prompts_path = tmp_path / "eval_prompts.jsonl"
    eval_prompts_path.write_text("", encoding="utf-8")

    config = {
        "mix": {
            "seed": 42,
            "source_csv": str(csv_path),
            "ichikara_path": str(ichikara_path),
            "eval_prompts_path": str(eval_prompts_path),
            "output_path": str(tmp_path / "mix_005.jsonl"),
            "stats_report": str(tmp_path / "phase3_sft005_distill_stats.md"),
            "length_guard": {"base_mean_chars": 176, "tolerance": 0.20},
        }
    }
    import yaml

    config_path = tmp_path / "mix_005.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        prepare_distill_mix(config_path)


# ---------------------------------------------------------------------------
# real CSV integration smoke test
# ---------------------------------------------------------------------------


def test_real_distill_csv_exists_and_parses() -> None:
    assert REAL_CSV_PATH.exists()
    rows = read_distill_csv(REAL_CSV_PATH)
    assert len(rows) == 4000
    for row in rows:
        assert isinstance(row["detail"], dict)
        assert row["category"]
        assert row["response"]
