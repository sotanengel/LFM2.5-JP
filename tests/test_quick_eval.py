"""Quick held-out PPL eval helpers: build_heldout, measure_ppl (Issue #29 lead-in)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import transformers

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.eval.quick_eval import build_heldout, measure_ppl

# ---------------------------------------------------------------------------
# build_heldout
# ---------------------------------------------------------------------------


def _write_corpus_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "corpus.yaml"
    config_path.write_text(
        """
cache_dir: data/raw
corpora:
  - name: wikipedia_ja
    hf_id: wikimedia/wikipedia
    hf_config: 20231101.ja
    split: train
    language: ja
""",
        encoding="utf-8",
    )
    return config_path


def test_build_heldout_skips_and_takes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_corpus_config(tmp_path)
    fake_rows = [{"text": f"doc {i}"} for i in range(10)]
    monkeypatch.setattr(
        "lfm25_ja.eval.quick_eval.download_corpus", lambda *_a, **_k: fake_rows
    )

    out_path = tmp_path / "heldout.jsonl"
    result_path = build_heldout(config_path, out_path, skip=3, take=4)

    assert result_path == out_path
    docs = _read_jsonl(out_path)
    assert [d["text"] for d in docs] == ["doc 3", "doc 4", "doc 5", "doc 6"]


def test_build_heldout_normalizes_nfkc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_corpus_config(tmp_path)
    # full-width "Ａ" should be NFKC-normalized to half-width "A"
    fake_rows = [{"text": "Ａ" * 3}]
    monkeypatch.setattr(
        "lfm25_ja.eval.quick_eval.download_corpus", lambda *_a, **_k: fake_rows
    )

    out_path = tmp_path / "heldout.jsonl"
    build_heldout(config_path, out_path, skip=0, take=1)

    docs = _read_jsonl(out_path)
    assert docs[0]["text"] == "AAA"


def test_build_heldout_missing_wikipedia_ja_entry_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "corpus.yaml"
    config_path.write_text("cache_dir: data/raw\ncorpora: []\n", encoding="utf-8")
    monkeypatch.setattr(
        "lfm25_ja.eval.quick_eval.download_corpus", lambda *_a, **_k: []
    )

    with pytest.raises(ValueError, match="wikipedia_ja"):
        build_heldout(config_path, tmp_path / "heldout.jsonl")


def test_build_heldout_empty_result_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_corpus_config(tmp_path)
    monkeypatch.setattr(
        "lfm25_ja.eval.quick_eval.download_corpus", lambda *_a, **_k: []
    )

    with pytest.raises(ValueError):
        build_heldout(config_path, tmp_path / "heldout.jsonl", skip=0, take=5)


# ---------------------------------------------------------------------------
# measure_ppl
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class _FakeCausalLM:
    """Stand-in causal LM returning a fixed loss regardless of input, so the
    weighted-average math in measure_ppl can be checked deterministically."""

    device = "cpu"

    def eval(self) -> None:
        return None

    def __call__(self, input_ids: torch.Tensor, labels: torch.Tensor) -> _FakeOutput:
        return _FakeOutput(torch.tensor(0.5))


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors=None, truncation=None, max_length=None):
        n = min(len(text.split()), max_length or 10**9)
        n = max(n, 2)
        return {"input_ids": torch.arange(n).unsqueeze(0)}


def test_measure_ppl_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        measure_ppl("fake/model", tmp_path / "does_not_exist.jsonl")


def test_measure_ppl_empty_file_raises(tmp_path: Path) -> None:
    heldout = tmp_path / "empty.jsonl"
    heldout.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        measure_ppl("fake/model", heldout)


def test_measure_ppl_weighted_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(
        heldout,
        [{"text": "one two three"}, {"text": "four five"}],
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeCausalLM()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeTokenizer()),
    )

    result = measure_ppl("fake/model", heldout, max_docs=None)

    assert result["n_docs"] == 2
    assert result["n_tokens"] == (3 - 1) + (2 - 1)
    assert result["ppl"] == pytest.approx(math.exp(0.5), rel=1e-4)


def test_measure_ppl_respects_max_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(
        heldout,
        [{"text": "one two"}, {"text": "three four"}, {"text": "five six"}],
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeCausalLM()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeTokenizer()),
    )

    result = measure_ppl("fake/model", heldout, max_docs=1)

    assert result["n_docs"] == 1


def test_measure_ppl_model_load_failure_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(heldout, [{"text": "one two"}])

    def _boom(cls, *a, **k):
        raise OSError("not found")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(_boom))

    with pytest.raises(RuntimeError, match="tokenizer"):
        measure_ppl("does/not-exist", heldout)


def test_measure_ppl_defaults_tokenizer_to_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When tokenizer_path is not given, the tokenizer is loaded from
    model_path (existing/default behavior)."""
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(heldout, [{"text": "one two"}])

    seen: dict[str, Any] = {}

    def _fake_tok_from_pretrained(cls, name_or_path, *a, **k):
        seen["tokenizer_source"] = name_or_path
        return _FakeTokenizer()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeCausalLM()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(_fake_tok_from_pretrained),
    )

    measure_ppl("some/model-path", heldout)

    assert seen["tokenizer_source"] == "some/model-path"


def test_measure_ppl_uses_explicit_tokenizer_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPT checkpoint dir may have no tokenizer files. --tokenizer lets the
    caller point at a separate location (e.g. the base HF model id) instead
    of failing or silently falling back."""
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(heldout, [{"text": "one two"}])

    seen: dict[str, Any] = {}

    def _fake_tok_from_pretrained(cls, name_or_path, *a, **k):
        seen["tokenizer_source"] = name_or_path
        return _FakeTokenizer()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeCausalLM()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(_fake_tok_from_pretrained),
    )

    measure_ppl(
        "outputs/sweep/L0/checkpoint",
        heldout,
        tokenizer_path="LiquidAI/LFM2.5-1.2B-Base",
    )

    assert seen["tokenizer_source"] == "LiquidAI/LFM2.5-1.2B-Base"


def test_measure_ppl_explicit_tokenizer_failure_names_tokenizer_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error message should reference the tokenizer_path actually used,
    not the (possibly different) model_path, so failures are easy to debug."""
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(heldout, [{"text": "one two"}])

    def _boom(cls, *a, **k):
        raise OSError("not found")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(_boom))

    with pytest.raises(RuntimeError, match="some/explicit-tokenizer"):
        measure_ppl(
            "does/not-exist",
            heldout,
            tokenizer_path="some/explicit-tokenizer",
        )
