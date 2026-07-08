"""Data pipeline tests: download (Issue #16), clean + contamination (Issue #17 / #22),
mixing (Issue #18), ChatML formatting (Issue #19), end-to-end prepare (Issue #20).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lfm25_ja.data.clean import (
    MinHashDeduplicator,
    _read_jsonl,
    clean_corpus,
    detect_language,
    length_filter,
    ngram_contamination_checker,
    normalize_nfkc,
    remove_control_chars,
    render_stats_report,
)
from lfm25_ja.data.download import download_all, download_corpus, load_corpus_config
from lfm25_ja.data.format_chat import (
    build_sft_example,
    decode_for_inspection,
    to_chatml,
    token_count_stats,
)
from lfm25_ja.data.mix import mix_corpora, render_mix_report
from lfm25_ja.data.prepare import prepare_data

CORPUS_YAML = """
cache_dir: data/raw

corpora:
  - name: wikipedia_ja
    hf_id: wikimedia/wikipedia
    hf_config: 20231101.ja
    split: train
    language: ja
  - name: aozora
    hf_id: globis-university/aozorabunko-clean
    split: train
    language: ja

clean:
  min_chars: 10
  max_chars: 1000
  lang_threshold: 0.5
  minhash:
    num_perm: 64
    threshold: 0.7
    ngram: 5
  contamination:
    ngram: 5
    threshold: 0.5
"""


@pytest.fixture
def corpus_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.yaml"
    path.write_text(CORPUS_YAML, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# download.py (Issue #16)
# ---------------------------------------------------------------------------


def test_load_corpus_config_reads_corpora_and_clean_sections(corpus_config_path: Path) -> None:
    cfg = load_corpus_config(corpus_config_path)
    assert cfg["cache_dir"] == "data/raw"
    assert len(cfg["corpora"]) == 2
    assert cfg["clean"]["min_chars"] == 10


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_passes_hf_config_as_second_positional_arg(mock_load: MagicMock) -> None:
    entry = {
        "name": "wikipedia_ja",
        "hf_id": "wikimedia/wikipedia",
        "hf_config": "20231101.ja",
        "split": "train",
    }
    download_corpus(entry, cache_dir="data/raw")
    args, kwargs = mock_load.call_args
    assert args[0] == "wikimedia/wikipedia"
    assert args[1] == "20231101.ja"
    assert kwargs["split"] == "train"
    assert kwargs["cache_dir"] == "data/raw"


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_without_hf_config_omits_second_arg(mock_load: MagicMock) -> None:
    entry = {"name": "aozora", "hf_id": "globis-university/aozorabunko-clean", "split": "train"}
    download_corpus(entry, cache_dir="data/raw")
    args, kwargs = mock_load.call_args
    assert args == ("globis-university/aozorabunko-clean",)
    assert kwargs["split"] == "train"


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_streaming_flag_forwarded(mock_load: MagicMock) -> None:
    entry = {"name": "aozora", "hf_id": "globis-university/aozorabunko-clean", "split": "train"}
    download_corpus(entry, cache_dir="data/raw", streaming=True)
    _, kwargs = mock_load.call_args
    assert kwargs["streaming"] is True


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_corpus_error_includes_corpus_name(mock_load: MagicMock) -> None:
    mock_load.side_effect = OSError("network unreachable")
    entry = {"name": "wikipedia_ja", "hf_id": "wikimedia/wikipedia", "hf_config": "20231101.ja"}
    with pytest.raises(RuntimeError, match="wikipedia_ja"):
        download_corpus(entry, cache_dir="data/raw")


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_all_downloads_only_requested_names(
    mock_load: MagicMock, corpus_config_path: Path
) -> None:
    mock_load.return_value = "dataset-stub"
    result = download_all(corpus_config_path, names=["aozora"])
    assert list(result.keys()) == ["aozora"]
    assert mock_load.call_count == 1


@patch("lfm25_ja.data.download.datasets.load_dataset")
def test_download_all_downloads_all_when_names_is_none(
    mock_load: MagicMock, corpus_config_path: Path
) -> None:
    mock_load.return_value = "dataset-stub"
    result = download_all(corpus_config_path)
    assert set(result.keys()) == {"wikipedia_ja", "aozora"}
    assert mock_load.call_count == 2


def test_download_all_unknown_name_raises(corpus_config_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown_corpus"):
        download_all(corpus_config_path, names=["unknown_corpus"])


# ---------------------------------------------------------------------------
# clean.py: normalization / control chars (Issue #17)
# ---------------------------------------------------------------------------


def test_normalize_nfkc_fullwidth_alnum_to_halfwidth() -> None:
    assert normalize_nfkc("ABC１２３") == "ABC123"


def test_normalize_nfkc_halfwidth_katakana_to_fullwidth() -> None:
    assert normalize_nfkc("ﾃｽﾄ") == "テスト"


def test_remove_control_chars_keeps_newline_and_tab() -> None:
    text = "line1\nline2\tindented"
    assert remove_control_chars(text) == text


def test_remove_control_chars_strips_control_and_private_use() -> None:
    text = "clean" + chr(0) + "text" + chr(7) + chr(0xE000) + "withjunk"
    result = remove_control_chars(text)
    assert chr(0) not in result
    assert chr(7) not in result
    assert chr(0xE000) not in result
    assert "clean" in result and "text" in result and "junk" in result


def test_detect_language_japanese_sentence() -> None:
    assert detect_language("これは日本語の文章です。テストを行います。") == "ja"


def test_detect_language_english_sentence() -> None:
    assert detect_language("This is a plain English sentence for testing purposes.") == "en"


def test_detect_language_empty_string_is_other() -> None:
    assert detect_language("") == "other"


# ---------------------------------------------------------------------------
# clean.py: length filter (Issue #17)
# ---------------------------------------------------------------------------


def test_length_filter_boundaries() -> None:
    assert length_filter("a" * 50, min_chars=50, max_chars=100) is True
    assert length_filter("a" * 100, min_chars=50, max_chars=100) is True
    assert length_filter("a" * 49, min_chars=50, max_chars=100) is False
    assert length_filter("a" * 101, min_chars=50, max_chars=100) is False


# ---------------------------------------------------------------------------
# clean.py: MinHash near-duplicate removal (Issue #17)
# ---------------------------------------------------------------------------


def test_minhash_deduplicator_flags_near_duplicate_and_keeps_unrelated() -> None:
    dedup = MinHashDeduplicator(num_perm=64, threshold=0.7, ngram=5)
    base = "吾輩は猫である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。" * 3
    near_duplicate = base + "。"  # trivially near-identical
    unrelated = "本日は晴天なり。遠足のお知らせをいたします。持ち物を確認してください。" * 3

    assert dedup.add_and_check("doc1", base) is False
    assert dedup.add_and_check("doc2", near_duplicate) is True
    assert dedup.add_and_check("doc3", unrelated) is False


# ---------------------------------------------------------------------------
# clean.py: n-gram contamination check (Issue #22)
# ---------------------------------------------------------------------------


def test_ngram_contamination_checker_flags_eval_overlap() -> None:
    eval_texts = ["日本の首都は東京都です。人口は約1400万人です。"]
    checker = ngram_contamination_checker(eval_texts, ngram=5)

    contaminated = (
        "日本の首都は東京都です。人口は約1400万人です。" + "これは学習データに混入した文です。"
    )
    clean_doc = "猫は液体である説が有名だ。柔軟な体を持つ動物として知られている。"

    assert checker.check(contaminated) > 0.5
    assert checker.check(clean_doc) < 0.1


def test_ngram_contamination_checker_empty_text_is_zero() -> None:
    checker = ngram_contamination_checker(["some eval text"], ngram=5)
    assert checker.check("") == 0.0


# ---------------------------------------------------------------------------
# clean.py: clean_corpus end-to-end (Issue #17 + #22)
# ---------------------------------------------------------------------------


def test_clean_corpus_end_to_end_stats_are_consistent() -> None:
    docs = [
        {"id": "1", "text": "これはテスト文書です。" * 5},  # valid ja, long enough
        {"id": "2", "text": "これはテスト文書です。" * 5},  # near-duplicate of #1
        {"id": "3", "text": "short"},  # too short
        {"id": "4", "text": "This is a valid English document for the mixing corpus." * 2},
        {"id": "5", "text": "1234567890"},  # digits only -> not ja/en, likely "other"
    ]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.7, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.5},
    }
    clean_docs, stats = clean_corpus(docs, cfg)

    assert stats["input_count"] == 5
    assert stats["output_count"] == len(clean_docs)
    assert stats["output_count"] < stats["input_count"]
    total_removed = sum(stage["removed"] for stage in stats["stages"])
    assert stats["input_count"] - total_removed == stats["output_count"]


def test_clean_corpus_removes_eval_contaminated_documents() -> None:
    eval_texts = ["日本の首都は東京都です。人口は約1400万人です。"]
    docs = [
        {"id": "1", "text": "日本の首都は東京都です。人口は約1400万人です。" * 2},
        {"id": "2", "text": "猫は液体である説が有名だ。柔軟な体を持つ動物として知られている。" * 2},
    ]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.9, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.3},
    }
    clean_docs, stats = clean_corpus(docs, cfg, eval_texts=eval_texts)
    remaining_ids = {d["id"] for d in clean_docs}
    assert "1" not in remaining_ids
    assert "2" in remaining_ids
    assert any(stage["name"] == "contamination_filter" for stage in stats["stages"])


def test_clean_corpus_language_mix_ratio_in_stats() -> None:
    """Stats should make it possible to compute the ja/en mixing ratio downstream."""
    docs = [
        {"id": "1", "text": "これは日本語のテスト文書です。" * 4},
        {"id": "2", "text": "This is an English test document used for the mix." * 3},
    ]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.9, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.5},
    }
    clean_docs, stats = clean_corpus(docs, cfg)
    languages = {d["language"] for d in clean_docs}
    assert languages == {"ja", "en"}
    assert stats["output_count"] == 2


def test_clean_corpus_output_docs_retain_text_field_for_chatml_stage() -> None:
    """Downstream ChatML formatting (format_chat.py) consumes the 'text' field untouched."""
    docs = [{"id": "1", "text": "これはテスト文書です。" * 5}]
    cfg = {
        "min_chars": 10,
        "max_chars": 1000,
        "lang_threshold": 0.5,
        "minhash": {"num_perm": 64, "threshold": 0.7, "ngram": 5},
        "contamination": {"ngram": 5, "threshold": 0.5},
    }
    clean_docs, _ = clean_corpus(docs, cfg)
    assert clean_docs[0]["text"] == docs[0]["text"]


# ---------------------------------------------------------------------------
# clean.py: markdown stats report
# ---------------------------------------------------------------------------


def test_render_stats_report_contains_key_numbers() -> None:
    stats = {
        "input_count": 10,
        "output_count": 6,
        "stages": [
            {"name": "length_filter", "removed": 2, "remaining": 8, "removal_rate": 0.2},
            {"name": "language_filter", "removed": 1, "remaining": 7, "removal_rate": 0.125},
            {"name": "dedup", "removed": 1, "remaining": 6, "removal_rate": 0.142857},
        ],
    }
    report = render_stats_report(stats)
    assert "length_filter" in report
    assert "10" in report
    assert "6" in report
    assert "|" in report  # markdown table


# ---------------------------------------------------------------------------
# mix.py: corpus mixing by language ratio (Issue #18)
# ---------------------------------------------------------------------------


def _make_docs(lang: str, n: int, tokens_each: int | None = None) -> list[dict]:
    docs = []
    for i in range(n):
        doc = {"id": f"{lang}-{i}", "lang": lang, "text": f"{lang} doc {i}"}
        if tokens_each is not None:
            doc["n_tokens"] = tokens_each
        docs.append(doc)
    return docs


def test_mix_corpora_selects_by_document_ratio() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 1000), "en": _make_docs("en", 150)}
    mixed, stats = mix_corpora(
        docs_by_lang, ratios={"ja": 0.85, "en": 0.15}, seed=42, unit="documents"
    )

    selected_ja = sum(1 for d in mixed if d["lang"] == "ja")
    selected_en = sum(1 for d in mixed if d["lang"] == "en")

    assert selected_ja == 850
    assert selected_en == 150
    assert len(mixed) == 1000

    assert stats["languages"]["ja"]["selected"] == 850
    assert stats["languages"]["en"]["selected"] == 150
    assert stats["languages"]["ja"]["available"] == 1000
    assert stats["languages"]["en"]["available"] == 150
    assert stats["total_selected"] == 1000


def test_mix_corpora_same_seed_is_deterministic() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 200), "en": _make_docs("en", 100)}
    mixed1, _ = mix_corpora(docs_by_lang, ratios={"ja": 0.7, "en": 0.3}, seed=7)
    mixed2, _ = mix_corpora(docs_by_lang, ratios={"ja": 0.7, "en": 0.3}, seed=7)

    ids1 = [d["id"] for d in mixed1]
    ids2 = [d["id"] for d in mixed2]
    assert ids1 == ids2


def test_mix_corpora_different_seed_changes_order() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 200), "en": _make_docs("en", 100)}
    mixed1, _ = mix_corpora(docs_by_lang, ratios={"ja": 0.7, "en": 0.3}, seed=1)
    mixed2, _ = mix_corpora(docs_by_lang, ratios={"ja": 0.7, "en": 0.3}, seed=2)

    ids1 = [d["id"] for d in mixed1]
    ids2 = [d["id"] for d in mixed2]
    # Selection AND ordering both derive from the seed, so a different seed is
    # expected to change which documents are picked as well as their order.
    assert len(ids1) == len(ids2) > 0
    assert ids1 != ids2


def test_mix_corpora_empty_ratios_raises() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 10)}
    with pytest.raises(ValueError):
        mix_corpora(docs_by_lang, ratios={}, seed=1)


def test_mix_corpora_negative_ratio_raises() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 10), "en": _make_docs("en", 10)}
    with pytest.raises(ValueError):
        mix_corpora(docs_by_lang, ratios={"ja": -0.1, "en": 1.1}, seed=1)


def test_mix_corpora_ratios_need_not_sum_to_one() -> None:
    # ratios are normalized internally; 85:15 should behave the same as 0.85:0.15
    docs_by_lang = {"ja": _make_docs("ja", 1000), "en": _make_docs("en", 150)}
    mixed, stats = mix_corpora(docs_by_lang, ratios={"ja": 85, "en": 15}, seed=42)
    assert stats["languages"]["ja"]["selected"] == 850
    assert stats["languages"]["en"]["selected"] == 150
    assert len(mixed) == 1000


def test_mix_corpora_unit_tokens_allocates_by_token_budget() -> None:
    docs_by_lang = {
        "ja": _make_docs("ja", 100, tokens_each=100),  # 10,000 tokens available
        "en": _make_docs("en", 100, tokens_each=10),  # 1,000 tokens available
    }
    mixed, stats = mix_corpora(docs_by_lang, ratios={"ja": 0.5, "en": 0.5}, seed=3, unit="tokens")

    ja_tokens = sum(d["n_tokens"] for d in mixed if d["lang"] == "ja")
    en_tokens = sum(d["n_tokens"] for d in mixed if d["lang"] == "en")

    # en is the limiting language (1,000 tokens / 0.5 ratio => 2,000 token budget total,
    # but ja can only supply 10,000 tokens / 0.5 ratio => 20,000 -> en is the bottleneck).
    assert en_tokens <= 1000
    assert ja_tokens <= 10000
    # ratio should roughly hold (within one document's worth of tokens)
    assert abs(ja_tokens - en_tokens) <= 100
    assert stats["languages"]["ja"]["available"] == 10000
    assert stats["languages"]["en"]["available"] == 1000


def test_mix_corpora_unit_tokens_missing_field_raises() -> None:
    docs_by_lang = {
        "ja": [{"id": "ja-0", "lang": "ja", "text": "no tokens field"}],
        "en": _make_docs("en", 5, tokens_each=10),
    }
    with pytest.raises(ValueError):
        mix_corpora(docs_by_lang, ratios={"ja": 0.5, "en": 0.5}, seed=1, unit="tokens")


def test_mix_corpora_stats_ratio_actual_matches_selection() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 1000), "en": _make_docs("en", 150)}
    _, stats = mix_corpora(docs_by_lang, ratios={"ja": 0.85, "en": 0.15}, seed=42)

    assert stats["languages"]["ja"]["ratio_actual"] == pytest.approx(0.85, abs=1e-6)
    assert stats["languages"]["en"]["ratio_actual"] == pytest.approx(0.15, abs=1e-6)
    assert stats["languages"]["ja"]["ratio_target"] == pytest.approx(0.85, abs=1e-6)
    assert stats["languages"]["en"]["ratio_target"] == pytest.approx(0.15, abs=1e-6)


def test_render_mix_report_contains_key_numbers() -> None:
    docs_by_lang = {"ja": _make_docs("ja", 1000), "en": _make_docs("en", 150)}
    _, stats = mix_corpora(docs_by_lang, ratios={"ja": 0.85, "en": 0.15}, seed=42)
    report = render_mix_report(stats)
    assert "ja" in report
    assert "en" in report
    assert "850" in report
    assert "150" in report
    assert "|" in report


# ---------------------------------------------------------------------------
# format_chat.py: ChatML conversion + loss masking (Issue #19)
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Deterministic whitespace-level tokenizer for tests (no HF download).

    Treats ``<|im_start|>`` / ``<|im_end|>`` as standalone tokens (mimicking how a
    real tokenizer's special-token pre-tokenization keeps them separate from
    surrounding text), and splits everything else on whitespace. Intentionally has
    no ``apply_chat_template`` attribute so callers exercise the to_chatml() fallback.
    """

    _SPECIAL_RE = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inverse: dict[int, str] = {}

    def _token_id(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse[idx] = token
        return self.vocab[token]

    def encode(self, text: str) -> list[int]:
        tokens: list[str] = []
        for chunk in self._SPECIAL_RE.split(text):
            if not chunk:
                continue
            if chunk in ("<|im_start|>", "<|im_end|>"):
                tokens.append(chunk)
            else:
                tokens.extend(chunk.split())
        return [self._token_id(t) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inverse[i] for i in ids)

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}


def test_to_chatml_formats_tags_in_order() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "こんにちは、元気です。"},
    ]
    text = to_chatml(messages)
    assert text.index("<|im_start|>system") < text.index("<|im_start|>user")
    assert text.index("<|im_start|>user") < text.index("<|im_start|>assistant")
    assert "You are helpful.<|im_end|>" in text
    assert "こんにちは、元気です。<|im_end|>" in text


def test_to_chatml_invalid_role_raises() -> None:
    with pytest.raises(ValueError):
        to_chatml([{"role": "tool", "content": "x"}])


def test_build_sft_example_unmasks_only_assistant_content() -> None:
    tokenizer = MockTokenizer()
    messages = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi friend"},
    ]
    example = build_sft_example(messages, tokenizer, max_seq_len=100)

    input_ids = example["input_ids"]
    labels = example["labels"]
    assert len(input_ids) == len(labels) == len(example["attention_mask"])

    learned_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    learned_text = tokenizer.decode(learned_tokens)
    assert learned_text == "Hi friend"

    # Everything outside the assistant content must be masked.
    masked_positions = [i for i, lab in enumerate(labels) if lab == -100]
    masked_text = tokenizer.decode([input_ids[i] for i in masked_positions])
    assert "Hi" not in masked_text.split()
    assert "friend" not in masked_text.split()


def test_build_sft_example_masks_system_and_user_but_not_assistant() -> None:
    tokenizer = MockTokenizer()
    messages = [
        {"role": "system", "content": "Be nice"},
        {"role": "user", "content": "What is 2+2"},
        {"role": "assistant", "content": "It is 4"},
    ]
    example = build_sft_example(messages, tokenizer, max_seq_len=200)
    input_ids, labels = example["input_ids"], example["labels"]

    learned = tokenizer.decode([t for t, lab in zip(input_ids, labels) if lab != -100])
    assert learned == "It is 4"


def test_build_sft_example_multiturn_unmasks_each_assistant_turn() -> None:
    tokenizer = MockTokenizer()
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    example = build_sft_example(messages, tokenizer, max_seq_len=200)
    input_ids, labels = example["input_ids"], example["labels"]
    learned = tokenizer.decode([t for t, lab in zip(input_ids, labels) if lab != -100])
    assert learned == "first answer second answer"


def test_build_sft_example_truncates_to_max_seq_len() -> None:
    tokenizer = MockTokenizer()
    messages = [
        {"role": "user", "content": "one two three four five"},
        {"role": "assistant", "content": "six seven eight nine ten"},
    ]
    example = build_sft_example(messages, tokenizer, max_seq_len=5)
    assert len(example["input_ids"]) == 5
    assert len(example["labels"]) == 5
    assert len(example["attention_mask"]) == 5


def test_decode_for_inspection_marks_learned_span() -> None:
    tokenizer = MockTokenizer()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    example = build_sft_example(messages, tokenizer, max_seq_len=100)
    inspection = decode_for_inspection(example, tokenizer)
    assert "【learned】" in inspection
    assert "【/learned】" in inspection
    learned_start = inspection.index("【learned】")
    learned_end = inspection.index("【/learned】")
    assert "world" in inspection[learned_start:learned_end]
    assert "world" not in inspection[:learned_start]


def test_token_count_stats_basic_values() -> None:
    tokenizer = MockTokenizer()
    messages_a = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    messages_b = [
        {"role": "user", "content": "a longer question here"},
        {"role": "assistant", "content": "a somewhat longer answer here too"},
    ]
    examples = [
        build_sft_example(messages_a, tokenizer, max_seq_len=100),
        build_sft_example(messages_b, tokenizer, max_seq_len=100),
    ]
    stats = token_count_stats(examples)

    assert stats["count"] == 2
    lengths = [len(ex["input_ids"]) for ex in examples]
    assert stats["input_length"]["min"] == min(lengths)
    assert stats["input_length"]["max"] == max(lengths)
    assert stats["input_length"]["mean"] == pytest.approx(sum(lengths) / 2)

    learned_counts = [sum(1 for lab in ex["labels"] if lab != -100) for ex in examples]
    assert stats["learned_tokens"]["min"] == min(learned_counts)
    assert stats["learned_tokens"]["max"] == max(learned_counts)


# ---------------------------------------------------------------------------
# prepare.py: end-to-end download -> clean -> mix -> report orchestrator (Issue #20)
# ---------------------------------------------------------------------------

PREPARE_CORPUS_YAML = """
cache_dir: data/raw

corpora:
  - name: wikipedia_ja
    hf_id: wikimedia/wikipedia
    hf_config: 20231101.ja
    split: train
    language: ja
  - name: wikipedia_en
    hf_id: wikimedia/wikipedia
    hf_config: 20231101.en
    split: train
    language: en

clean:
  min_chars: 5
  max_chars: 1000
  lang_threshold: 0.5
  minhash:
    num_perm: 64
    threshold: 0.9
    ngram: 5
  contamination:
    ngram: 5
    threshold: 0.5

mix:
  seed: 1
  ratios:
    ja: 0.5
    en: 0.5
  unit: documents
"""


@pytest.fixture
def prepare_corpus_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.yaml"
    path.write_text(PREPARE_CORPUS_YAML, encoding="utf-8")
    return path


def _fake_dataset(lang: str, n: int) -> list[dict]:
    if lang == "ja":
        base = "これは日本語のテスト文書です。" * 4
    else:
        base = "This is an English test document used for the preparation pipeline." * 2
    return [{"text": f"{base} {i}"} for i in range(n)]


@patch("lfm25_ja.data.prepare.download_corpus")
def test_prepare_data_end_to_end_writes_mixture_and_report(
    mock_download: MagicMock, prepare_corpus_config_path: Path, tmp_path: Path
) -> None:
    def fake_download(entry, cache_dir, streaming=False):
        return _fake_dataset(entry["language"], 20)

    mock_download.side_effect = fake_download
    output_dir = tmp_path / "processed"

    result = prepare_data(str(prepare_corpus_config_path), output_dir=str(output_dir))

    assert mock_download.call_count == 2
    assert Path(result["output_path"]).exists()
    assert Path(result["report_path"]).exists()

    mixed_docs = _read_jsonl(result["output_path"])
    assert len(mixed_docs) > 0
    langs = {d["language"] for d in mixed_docs}
    assert langs <= {"ja", "en"}

    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "wikipedia_ja" in report_text
    assert "wikipedia_en" in report_text

    assert "wikipedia_ja" in result["corpora"]
    assert "wikipedia_en" in result["corpora"]
    assert result["mix"]["total_selected"] > 0


@patch("lfm25_ja.data.prepare.download_corpus")
def test_prepare_data_sample_limit_caps_rows_per_corpus(
    mock_download: MagicMock, prepare_corpus_config_path: Path, tmp_path: Path
) -> None:
    def fake_download(entry, cache_dir, streaming=False):
        return _fake_dataset(entry["language"], 50)

    mock_download.side_effect = fake_download
    output_dir = tmp_path / "processed"

    result = prepare_data(
        str(prepare_corpus_config_path), sample_limit=5, output_dir=str(output_dir)
    )
    for stats in result["corpora"].values():
        assert stats["downloaded_count"] <= 5


@patch("lfm25_ja.data.prepare.download_corpus")
def test_prepare_data_names_filters_corpora(
    mock_download: MagicMock, prepare_corpus_config_path: Path, tmp_path: Path
) -> None:
    mock_download.return_value = _fake_dataset("ja", 20)
    output_dir = tmp_path / "processed"

    result = prepare_data(
        str(prepare_corpus_config_path),
        names=["wikipedia_ja"],
        output_dir=str(output_dir),
    )
    assert list(result["corpora"].keys()) == ["wikipedia_ja"]
    assert mock_download.call_count == 1


def test_prepare_data_unknown_name_raises(prepare_corpus_config_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown_corpus"):
        prepare_data(
            str(prepare_corpus_config_path),
            names=["unknown_corpus"],
            output_dir=str(tmp_path / "processed"),
        )


@patch("lfm25_ja.data.prepare.download_corpus")
def test_prepare_data_download_failure_includes_corpus_and_stage(
    mock_download: MagicMock, prepare_corpus_config_path: Path, tmp_path: Path
) -> None:
    mock_download.side_effect = RuntimeError("Failed to download corpus 'wikipedia_ja' (x): boom")
    with pytest.raises(RuntimeError, match="wikipedia_ja"):
        prepare_data(str(prepare_corpus_config_path), output_dir=str(tmp_path / "processed"))


@patch("lfm25_ja.data.prepare.download_corpus")
def test_prepare_data_eval_texts_triggers_contamination_filter(
    mock_download: MagicMock, prepare_corpus_config_path: Path, tmp_path: Path
) -> None:
    def fake_download(entry, cache_dir, streaming=False):
        return _fake_dataset(entry["language"], 20)

    mock_download.side_effect = fake_download

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        '{"text": "これは日本語のテスト文書です。これは日本語のテスト文書です。"}\n',
        encoding="utf-8",
    )

    result = prepare_data(
        str(prepare_corpus_config_path),
        output_dir=str(tmp_path / "processed"),
        eval_texts_path=str(eval_path),
    )
    ja_stages = result["corpora"]["wikipedia_ja"]["clean"]["stages"]
    assert any(stage["name"] == "contamination_filter" for stage in ja_stages)
