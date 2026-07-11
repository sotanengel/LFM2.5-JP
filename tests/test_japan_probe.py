"""Japan-knowledge probe scoring: extract_answer_segment, score_answer,
build_report, model-spec parsing (Issue #76)."""

from __future__ import annotations

import torch
import transformers

from lfm25_ja.eval.japan_probe import (
    QUESTIONS,
    _legacy_score,
    _parse_model_specs,
    build_report,
    extract_answer_segment,
    run_japan_probe,
    score_answer,
)

# ---------------------------------------------------------------------------
# extract_answer_segment
# ---------------------------------------------------------------------------


def test_extract_answer_segment_empty_string() -> None:
    assert extract_answer_segment("") == ""


def test_extract_answer_segment_first_sentence_only() -> None:
    assert extract_answer_segment("富士山です。標高は3776メートルです。") == "富士山です。"


def test_extract_answer_segment_cuts_at_self_generated_question() -> None:
    text = "富士山です。質問: 次に高い山は?\n答え: 北岳です。"
    assert extract_answer_segment(text) == "富士山です。"


def test_extract_answer_segment_cuts_at_ascii_question_marker() -> None:
    text = "富士山です。Question: what about the second one?"
    assert extract_answer_segment(text) == "富士山です。"


def test_extract_answer_segment_cuts_at_choice_marker_before_sentence_end() -> None:
    # No "。" before the choice list -- boundary regex must still cut it off.
    text = "わかりません A:富士山 B:立山 C:御嶽山"
    assert extract_answer_segment(text) == "わかりません"


def test_extract_answer_segment_cuts_at_blank_line() -> None:
    text = "富士山です。\n\n次の質問もお願いします。"
    assert extract_answer_segment(text) == "富士山です。"


def test_extract_answer_segment_does_not_pick_up_choice_enumeration() -> None:
    # Wrong first answer (岐阜県), followed by a choice list that happens to
    # contain the correct answer (滋賀県) -- the segment must stop before it.
    text = "岐阜県です。A:岐阜県 B:富山県 C:滋賀県"
    segment = extract_answer_segment(text)
    assert "滋賀" not in segment
    assert segment == "岐阜県です。"


# ---------------------------------------------------------------------------
# score_answer
# ---------------------------------------------------------------------------


def test_score_answer_correct_at_start_is_true() -> None:
    assert score_answer("富士山です。", ["富士山", "富士"]) is True


def test_score_answer_choice_enumeration_case_is_false() -> None:
    # Regression case: 岐阜県 A:高尾山 B:立山 C:滋賀県 -- must NOT match "滋賀"
    # even though it appears later in the raw text.
    text = "岐阜県 A:高尾山 B:立山 C:滋賀県"
    assert score_answer(text, ["滋賀"]) is False


def test_score_answer_no_match_is_false() -> None:
    assert score_answer("わかりません。", ["富士山"]) is False


def test_legacy_score_would_incorrectly_match_choice_enumeration() -> None:
    # Documents *why* the fix was needed: the old scorer matches anywhere.
    text = "岐阜県 A:高尾山 B:立山 C:滋賀県"
    assert _legacy_score(text, ["滋賀"]) is True
    assert score_answer(text, ["滋賀"]) is False


# ---------------------------------------------------------------------------
# QUESTIONS sanity
# ---------------------------------------------------------------------------


def test_questions_has_ten_fields_of_five() -> None:
    fields: dict[str, int] = {}
    for field, _q, _answers in QUESTIONS:
        fields[field] = fields.get(field, 0) + 1
    assert len(fields) == 10
    assert all(count == 5 for count in fields.values())
    assert len(QUESTIONS) == 50


# ---------------------------------------------------------------------------
# _parse_model_specs
# ---------------------------------------------------------------------------


def test_parse_model_specs_parses_label_equals_path() -> None:
    specs = _parse_model_specs(
        ["base=LiquidAI/LFM2.5-1.2B-Base", "ckpt9000=outputs/x/checkpoint-9000"]
    )
    assert specs == [
        ("base", "LiquidAI/LFM2.5-1.2B-Base"),
        ("ckpt9000", "outputs/x/checkpoint-9000"),
    ]


def test_parse_model_specs_rejects_missing_equals() -> None:
    import pytest

    with pytest.raises(ValueError):
        _parse_model_specs(["base-only-no-equals"])


def test_parse_model_specs_rejects_empty_label_or_path() -> None:
    import pytest

    with pytest.raises(ValueError):
        _parse_model_specs(["=path/only"])
    with pytest.raises(ValueError):
        _parse_model_specs(["label="])


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


def test_build_report_marks_correct_and_incorrect() -> None:
    questions = [
        ("地理", "日本で一番高い山は何ですか?", ["富士山", "富士"]),
        ("地理", "日本の首都はどこですか?", ["東京"]),
    ]
    model_specs = [("base", "some/model")]
    raw_results = {
        ("base", 0): "富士山です。",
        ("base", 1): "わかりません。",
    }
    report = build_report(model_specs, questions, raw_results)

    assert "地理" in report
    assert "合計 (/2)" in report
    assert "O 富士山です。" in report
    assert "X わかりません。" in report


def test_build_report_flags_diff_between_old_and_new_scoring() -> None:
    questions = [("地理", "琵琶湖がある都道府県はどこですか?", ["滋賀"])]
    model_specs = [("base", "some/model")]
    raw_results = {("base", 0): "岐阜県 A:高尾山 B:立山 C:滋賀県"}

    report = build_report(model_specs, questions, raw_results)

    assert "旧O→新X" in report


def test_build_report_no_diff_column_when_scoring_agrees() -> None:
    questions = [("地理", "日本の首都はどこですか?", ["東京"])]
    model_specs = [("base", "some/model")]
    raw_results = {("base", 0): "東京です。"}

    report = build_report(model_specs, questions, raw_results)

    assert "旧O→新X" not in report
    assert "旧X→新O" not in report


# ---------------------------------------------------------------------------
# run_japan_probe (mocked transformers, CPU)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        return _FakeBatch()

    def decode(self, ids, skip_special_tokens=True):
        return "富士山です。"


class _FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=torch.tensor([[1, 2, 3]]))

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _FakeModel:
    device = "cpu"

    def eval(self) -> None:
        return None

    def generate(
        self,
        input_ids=None,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=None,
        repetition_penalty=1.05,
    ):
        extra = torch.zeros((1, max_new_tokens), dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


def test_run_japan_probe_returns_raw_text_per_model_and_question(monkeypatch) -> None:
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeModel()),
    )

    results = run_japan_probe([("base", "fake/model")], max_new_tokens=4)

    assert len(results) == len(QUESTIONS)
    assert results[("base", 0)] == "富士山です。"
