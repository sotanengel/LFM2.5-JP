"""Text cleaning pipeline: normalization, language ID, dedup, contamination check.

Implements the mandatory items from docs/lfm2_5-ja-plan.md sec 3:
NFKC normalization, language detection, MinHash near-duplicate removal,
control/private-use character stripping, length filtering, and n-gram
contamination checking against the evaluation set (Issue #17, #22).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

_HIRAGANA_RE = re.compile(r"[぀-ゟ]")
_KATAKANA_RE = re.compile(r"[゠-ヿ]")
_KANJI_RE = re.compile(r"[一-鿿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Private-use area / gaiji ranges to strip (BMP PUA + supplementary PUA-A/B).
_PRIVATE_USE_RANGES: tuple[tuple[int, int], ...] = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


def normalize_nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization (fullwidth/halfwidth folding etc.)."""
    return unicodedata.normalize("NFKC", text)


def _is_private_use(codepoint: int) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in _PRIVATE_USE_RANGES)


def remove_control_chars(text: str) -> str:
    """Strip control characters (keeping \\t/\\n) and private-use/gaiji characters."""
    kept = []
    for ch in text:
        if ch in ("\n", "\t"):
            kept.append(ch)
            continue
        if unicodedata.category(ch) == "Cc":
            continue
        if _is_private_use(ord(ch)):
            continue
        kept.append(ch)
    return "".join(kept)


def detect_language(text: str, lang_threshold: float = 0.5) -> str:
    """Heuristic language detection using script character ratios.

    Counts hiragana/katakana/kanji vs. Latin letters (digits and punctuation
    are ignored). Returns "ja" if the Japanese-script ratio meets
    ``lang_threshold``, "en" if the Latin ratio does, otherwise "other"
    (including empty/whitespace-only or non-letter text).
    """
    if not text:
        return "other"
    ja_chars = len(_HIRAGANA_RE.findall(text)) + len(_KATAKANA_RE.findall(text))
    ja_chars += len(_KANJI_RE.findall(text))
    latin_chars = len(_LATIN_RE.findall(text))
    total = ja_chars + latin_chars
    if total == 0:
        return "other"
    if ja_chars / total >= lang_threshold:
        return "ja"
    if latin_chars / total >= lang_threshold:
        return "en"
    return "other"


def length_filter(text: str, min_chars: int, max_chars: int) -> bool:
    """Return True if ``text`` length is within [min_chars, max_chars]."""
    length = len(text)
    return min_chars <= length <= max_chars


def _char_ngrams(text: str, n: int) -> set[str]:
    """Character n-grams of ``text``. Short texts yield a single shingle."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


class MinHashDeduplicator:
    """Approximate near-duplicate detector using MinHash + LSH over char n-grams."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.8, ngram: int = 5) -> None:
        self.num_perm = num_perm
        self.ngram = ngram
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def _minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for shingle in _char_ngrams(text, self.ngram):
            m.update(shingle.encode("utf-8"))
        return m

    def add_and_check(self, doc_id: str, text: str) -> bool:
        """Return True (and skip insertion) if ``text`` is a near-duplicate of a
        previously seen document; otherwise insert it and return False."""
        m = self._minhash(text)
        if self.lsh.query(m):
            return True
        self.lsh.insert(doc_id, m)
        return False


class _ContaminationChecker:
    """Checks documents for n-gram overlap against an evaluation-set n-gram pool."""

    def __init__(self, eval_texts: list[str], ngram: int) -> None:
        self.ngram = ngram
        self._eval_ngrams: set[str] = set()
        for text in eval_texts:
            self._eval_ngrams.update(_char_ngrams(text, ngram))

    def check(self, text: str) -> float:
        """Fraction of ``text``'s n-grams that also appear in the eval set."""
        doc_ngrams = _char_ngrams(text, self.ngram)
        if not doc_ngrams:
            return 0.0
        overlap = sum(1 for g in doc_ngrams if g in self._eval_ngrams)
        return overlap / len(doc_ngrams)


def ngram_contamination_checker(eval_texts: list[str], ngram: int) -> _ContaminationChecker:
    """Build a contamination checker against ``eval_texts`` (Issue #22)."""
    return _ContaminationChecker(eval_texts, ngram)


def _stage_stats(name: str, removed: int, remaining: int, before: int) -> dict[str, Any]:
    removal_rate = removed / before if before else 0.0
    return {"name": name, "removed": removed, "remaining": remaining, "removal_rate": removal_rate}


def clean_corpus(
    docs: Iterable[dict[str, Any]],
    cfg: dict[str, Any],
    eval_texts: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full cleaning pipeline over ``docs``.

    ``cfg`` is the ``clean`` section of corpus.yaml (min_chars, max_chars,
    lang_threshold, minhash{}, contamination{}). Returns (clean_docs, stats).
    """
    docs = list(docs)
    input_count = len(docs)

    min_chars = cfg.get("min_chars", 50)
    max_chars = cfg.get("max_chars", 50000)
    lang_threshold = cfg.get("lang_threshold", 0.5)
    minhash_cfg = cfg.get("minhash", {})
    contamination_cfg = cfg.get("contamination", {})

    stages: list[dict[str, Any]] = []

    # Normalize + strip control/private-use chars (transformation, not filtering).
    normalized: list[dict[str, Any]] = []
    for doc in docs:
        new_doc = dict(doc)
        text = normalize_nfkc(doc.get("text", ""))
        text = remove_control_chars(text)
        new_doc["text"] = text
        normalized.append(new_doc)

    # 1. Length filter.
    before = len(normalized)
    after_length = [d for d in normalized if length_filter(d["text"], min_chars, max_chars)]
    stages.append(
        _stage_stats("length_filter", before - len(after_length), len(after_length), before)
    )

    # 2. Language filter (keep only ja/en; drop "other").
    before = len(after_length)
    after_lang: list[dict[str, Any]] = []
    for d in after_length:
        lang = detect_language(d["text"], lang_threshold=lang_threshold)
        if lang in ("ja", "en"):
            d = dict(d)
            d["language"] = lang
            after_lang.append(d)
    stages.append(
        _stage_stats("language_filter", before - len(after_lang), len(after_lang), before)
    )

    # 3. MinHash near-duplicate removal.
    before = len(after_lang)
    dedup = MinHashDeduplicator(
        num_perm=minhash_cfg.get("num_perm", 128),
        threshold=minhash_cfg.get("threshold", 0.8),
        ngram=minhash_cfg.get("ngram", 5),
    )
    after_dedup: list[dict[str, Any]] = []
    for i, d in enumerate(after_lang):
        doc_id = str(d.get("id", i))
        if not dedup.add_and_check(doc_id, d["text"]):
            after_dedup.append(d)
    stages.append(_stage_stats("dedup", before - len(after_dedup), len(after_dedup), before))

    result = after_dedup

    # 4. Contamination filter against evaluation set (optional, Issue #22).
    if eval_texts:
        before = len(result)
        checker = ngram_contamination_checker(eval_texts, ngram=contamination_cfg.get("ngram", 13))
        threshold = contamination_cfg.get("threshold", 0.5)
        after_contam = [d for d in result if checker.check(d["text"]) < threshold]
        stages.append(
            _stage_stats(
                "contamination_filter", before - len(after_contam), len(after_contam), before
            )
        )
        result = after_contam

    stats = {
        "input_count": input_count,
        "output_count": len(result),
        "stages": stages,
    }
    return result, stats


def render_stats_report(stats: dict[str, Any]) -> str:
    """Render a markdown summary table of cleaning stats."""
    lines = [
        "# Cleaning pipeline report",
        "",
        f"- Input documents: {stats.get('input_count', 0)}",
        f"- Output documents: {stats.get('output_count', 0)}",
        "",
        "| stage | removed | remaining | removal_rate |",
        "|---|---|---|---|",
    ]
    for stage in stats.get("stages", []):
        lines.append(
            f"| {stage['name']} | {stage['removed']} | {stage['remaining']} "
            f"| {stage['removal_rate']:.2%} |"
        )
    return "\n".join(lines)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    docs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _write_jsonl(path: str | Path, docs: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Clean a JSONL corpus (Issue #17 / #22)")
    parser.add_argument("--config", required=True, help="Path to configs/data/corpus.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL path (with a 'text' field)")
    parser.add_argument("--output", required=True, help="Output JSONL path for cleaned docs")
    parser.add_argument(
        "--eval-texts", default=None, help="Optional eval-set JSONL for contamination check"
    )
    parser.add_argument("--report", default=None, help="Optional markdown report output path")
    args = parser.parse_args()

    config = load_config(args.config)
    clean_cfg = config.get("clean", config)

    docs = _read_jsonl(args.input)
    eval_texts = None
    if args.eval_texts:
        eval_docs = _read_jsonl(args.eval_texts)
        eval_texts = [d["text"] for d in eval_docs if "text" in d]

    clean_docs, stats = clean_corpus(docs, clean_cfg, eval_texts=eval_texts)
    _write_jsonl(args.output, clean_docs)
    logger.info("Cleaned %d -> %d documents", stats["input_count"], stats["output_count"])

    if args.report:
        report = render_stats_report(stats)
        Path(args.report).write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.report)


if __name__ == "__main__":
    main()
