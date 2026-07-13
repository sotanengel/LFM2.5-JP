"""Answer-extraction fix for the jsem / jmmlu(JP) ool bug (Issue #66).

Fixtures below are lifted verbatim from the real WSL baseline result records
(``~/llm-jp-eval/local_files/results/result_baseline-*.json``), where both
models' answers were format-mismatched against llm-jp-eval's built-in
``AnswerPatternId.ANSWER_TAGS_JP`` (jsem) / ``AnswerPatternId.CHOICE_ONLY_JP``
(jmmlu) patterns. The fix went through two rounds against the real harness:

Round 1 (pattern only): jsem answers with a bare label (``"yes"``) or
LaTeX-boxed label (``"\\boxed{yes}"``) followed by an explanation, never
wrapped in the ``<answer></answer>`` tags the built-in pattern requires ->
100% ool. jmmlu's own dataset processor caps generation at
``output_length=1`` token, so a response that would start "\\boxed{A}" gets
cut down to a lone ``"\\"`` -> ool (55% for JP-202606 vs 2% for Instruct).
Switching to a lenient `custom` regex + a modest output_length bump
(jsem=48, jmmlu=16) dropped these a lot, but a real rerun still showed
jsem(JP) 37%, jmmlu(JP) 15%, jmmlu(Instruct) 10% ool -- both models,
especially JP-202606 on math/logic jmmlu questions, often write a long
Japanese chain-of-thought explanation before ever stating the label, running
out of budget before reaching it.

Round 2 (budget + last-match): output_length widened further (jsem=160,
jmmlu=200, still within generation.max_new_tokens=256), and each regex
prefixed with a greedy ``.*`` so ``re.search``'s backtracking finds the
*last* occurrence of the label (the model's actual conclusion) instead of
the first thing that looks like one -- long explanations sometimes mention
and reject an option (e.g. "option A: ... this is false") before concluding,
which a leftmost-match regex would wrongly grab. Verified ool on a real
rerun (2026-07-13, 100 samples/task): jsem Instruct 0% / JP-202606 1%,
jmmlu Instruct 4% / JP-202606 5% -- all under the 10% acceptance threshold.
"""

from __future__ import annotations

import re
from pathlib import Path

from lfm25_ja.eval.run_llm_jp_eval import (
    DATASET_INFO_OVERRIDES,
    build_eval_dataset_config,
    extract_custom_answer,
    load_eval_config,
)


def _cfg():
    root = Path(__file__).resolve().parents[1]
    return load_eval_config(root / "configs" / "eval" / "llm_jp_eval.yaml")


# --- jsem -------------------------------------------------------------


def test_jsem_pattern_extracts_boxed_label():
    pattern = DATASET_INFO_OVERRIDES["jsem"]["answer_extract_pattern"]
    text = "\\boxed{yes}\n\n### 説明:\n前提は「"
    assert extract_custom_answer(text, pattern) == "yes"


def test_jsem_pattern_extracts_bare_label_with_trailing_explanation():
    pattern = DATASET_INFO_OVERRIDES["jsem"]["answer_extract_pattern"]
    text = "yes\n\n### 説明:\n前提と仮説は同じ文を"
    assert extract_custom_answer(text, pattern) == "yes"


def test_jsem_pattern_extracts_undef_from_boxed():
    pattern = DATASET_INFO_OVERRIDES["jsem"]["answer_extract_pattern"]
    text = "\\boxed{undef}\n\n### 説明:\n前提は"
    assert extract_custom_answer(text, pattern) == "undef"


def test_jsem_pattern_is_ool_when_no_label_token_present():
    # A genuine model failure (no label anywhere) must still count as ool,
    # not be silently coerced into a match.
    pattern = DATASET_INFO_OVERRIDES["jsem"]["answer_extract_pattern"]
    text = "前提と仮説は同じ主張を述べているため、"
    assert extract_custom_answer(text, pattern) == ""


def test_jsem_pattern_prefers_final_conclusion_over_earlier_mention():
    # Round 2 regression: a long chain-of-thought explanation can restate a
    # label mid-reasoning before its actual conclusion. The greedy `.*`
    # prefix must pick the LAST occurrence, not the first.
    pattern = DATASET_INFO_OVERRIDES["jsem"]["answer_extract_pattern"]
    text = (
        "前提はyesの可能性を示唆するが、断定はできない。よく検討すると、"
        "前提は仮説を含意しないため、結論としてはno"
    )
    assert extract_custom_answer(text, pattern) == "no"


def test_jsem_pattern_still_ool_on_truncated_reasoning_with_no_label():
    # A real WSL sample: the model's explanation runs out of budget before
    # ever stating a label. Widening output_length must not make this
    # falsely match -- it should remain ool.
    pattern = DATASET_INFO_OVERRIDES["jsem"]["answer_extract_pattern"]
    text = (
        "この前提は、商品が取り上げられたことと売り上げの増加との関係を示していますが、"
        "取り上げられたことが好ましいことであるという仮説を直接裏付ける情報は含まれていません。取り"
    )
    assert extract_custom_answer(text, pattern) == ""


def test_jsem_builtin_answer_tags_pattern_fails_on_real_samples():
    """Documents the bug: llm-jp-eval's builtin ANSWER_TAGS_JP regex requires
    literal <answer></answer> tags, which neither model ever emits."""
    text = "yes\n\n### 説明:\n前提と仮説は同じ文を"
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    assert match is None


# --- jmmlu --------------------------------------------------------------


def test_jmmlu_pattern_extracts_letter_from_boxed_prefix():
    pattern = DATASET_INFO_OVERRIDES["jmmlu"]["answer_extract_pattern"]
    text = "\\boxed{A}\n\n### 説明:\n選択肢Aが正しい"
    assert extract_custom_answer(text, pattern) == "A"


def test_jmmlu_pattern_extracts_bare_letter():
    pattern = DATASET_INFO_OVERRIDES["jmmlu"]["answer_extract_pattern"]
    assert extract_custom_answer("D", pattern) == "D"


def test_jmmlu_pattern_ool_on_truncated_backslash_with_no_letter():
    # Even with a wider output_length, a response that never reaches a
    # letter (e.g. still truncated) must remain ool rather than false-match.
    pattern = DATASET_INFO_OVERRIDES["jmmlu"]["answer_extract_pattern"]
    assert extract_custom_answer("\\", pattern) == ""


def test_jmmlu_pattern_prefers_final_conclusion_over_discussed_option():
    # Round 2 regression, taken from a real WSL sample (JP-202606, id=20):
    # the model discusses and rejects option A and B before running out of
    # budget. A leftmost-match regex would wrongly extract "A" here; the
    # greedy `.*` prefix must instead prefer the last (most recent) letter
    # actually mentioned.
    pattern = DATASET_INFO_OVERRIDES["jmmlu"]["answer_extract_pattern"]
    text = (
        "各文を分析します：\n\n選択肢A: これは必ずしも真ではない。\n\n"
        "選択肢B: これも真ではない。したがって消去法により正しい答えは D"
    )
    assert extract_custom_answer(text, pattern) == "D"


def test_jmmlu_pattern_still_ool_on_truncated_reasoning_with_no_letter():
    # Real WSL sample: a long derivation that never reaches a lettered
    # conclusion within budget must remain ool, not false-match on a stray
    # character.
    pattern = DATASET_INFO_OVERRIDES["jmmlu"]["answer_extract_pattern"]
    text = (
        "完全グラフ $ K_n $ の辺の数は次の式で与えられる："
        "$$ \\binom{n}{2} = \\frac{n(n-1)}{2} $$"
    )
    assert extract_custom_answer(text, pattern) == ""


def test_jmmlu_builtin_choice_only_pattern_captures_bare_backslash_verbatim():
    """Documents the bug: with output_length=1 the JP-202606 model's first
    token was a bare "\\" (the start of its habitual \\boxed{...} answer
    style). llm-jp-eval's builtin CHOICE_ONLY_JP first-line regex captures it
    verbatim, which is not in label_list=["A","B","C","D"] -> ool."""
    text = "\\"
    match = re.search(r"(?s)^(.*?)(?=\n|\Z)", text, re.DOTALL)
    pred = "".join(match.groups()).strip() if match else ""
    assert pred == "\\"
    assert pred not in ["A", "B", "C", "D"]


# --- override values themselves -----------------------------------------


def test_dataset_info_overrides_widen_output_length_budget():
    # jsem was 15 tokens, jmmlu was 1 token upstream (llm-jp-eval dataset
    # processors) -- both too small for a model that reasons/boxes before
    # answering. Round 1 (48/16) still left double-digit ool; round 2 widens
    # further (within generation.max_new_tokens=256) to give room for a full
    # chain-of-thought explanation to reach its conclusion.
    assert DATASET_INFO_OVERRIDES["jsem"]["output_length"] >= 128
    assert DATASET_INFO_OVERRIDES["jmmlu"]["output_length"] >= 128


def test_dataset_info_overrides_switch_to_custom_pattern():
    for task in ("jsem", "jmmlu"):
        assert DATASET_INFO_OVERRIDES[task]["answer_pattern_id"] == "custom"


def test_eval_config_yaml_carries_the_same_overrides():
    cfg = _cfg()
    overrides = cfg["eval"]["dataset_info_overrides"]
    assert overrides["jsem"]["answer_extract_pattern"] == DATASET_INFO_OVERRIDES["jsem"][
        "answer_extract_pattern"
    ]
    assert overrides["jmmlu"]["output_length"] == DATASET_INFO_OVERRIDES["jmmlu"]["output_length"]


def test_build_eval_dataset_config_includes_overrides_and_tasks():
    cfg = _cfg()
    dataset_cfg = build_eval_dataset_config(cfg)
    assert dataset_cfg["datasets"] == cfg["eval"]["tasks"]
    assert "jsem" in dataset_cfg["dataset_info_overrides"]
    assert "jmmlu" in dataset_cfg["dataset_info_overrides"]
    assert dataset_cfg["dataset_info_overrides"]["jsem"]["output_length"] == 160
