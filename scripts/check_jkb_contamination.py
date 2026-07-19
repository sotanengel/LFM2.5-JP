"""Smoke-check JKB v1 contamination against a small CPT corpus sample (Issue #121).

The K2 CPT preparation pass is the real contamination filter — it streams the
full corpora through `clean.clean_corpus` with `eval_texts=<JKB eval_texts>`
and drops docs whose n-gram overlap with JKB exceeds `contamination.threshold`
(see `configs/data/corpus.yaml`).

This script is the K0-time smoke check: it exercises the same
`ngram_contamination_checker` mechanism on a synthetic corpus + optionally a
locally-available sample JSONL, and reports the overlap distribution so
we know K2 will actually filter something meaningful.

Usage:
    uv run --no-sync python scripts/check_jkb_contamination.py \
        --sample-jsonl data/raw/wikipedia_ja/*.jsonl.gz --max-docs 500
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from lfm25_ja.data.clean import _char_ngrams, ngram_contamination_checker


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _iter_sample(paths: list[Path], max_docs: int) -> list[str]:
    docs: list[str] = []
    for p in paths:
        if len(docs) >= max_docs:
            break
        try:
            f = _open_maybe_gz(p)
        except FileNotFoundError:
            continue
        with f:
            for line in f:
                if len(docs) >= max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text") or ""
                if text:
                    docs.append(text)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="JKB v1 contamination smoke check")
    parser.add_argument("--jkb-eval-texts", default="datasets/eval/jkb/eval_texts.jsonl")
    parser.add_argument("--sample-jsonl", nargs="*", default=[], help="Optional corpus sample paths")
    parser.add_argument("--max-docs", type=int, default=500)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    eval_texts = []
    with open(args.jkb_eval_texts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eval_texts.append(json.loads(line)["text"])

    print(f"JKB rows loaded: {len(eval_texts)}")
    total_ngrams = sum(len(_char_ngrams(t, args.ngram)) for t in eval_texts)
    print(f"JKB n-gram pool size (n={args.ngram}): ~{total_ngrams}")

    checker = ngram_contamination_checker(eval_texts, ngram=args.ngram)

    # 1. Synthetic sanity check: a document quoting a JKB answer verbatim
    #    should return non-zero overlap; unrelated text should return ~0.
    hit_doc = eval_texts[0] + "。" + eval_texts[3]
    miss_doc = "本日は晴天なりまさか本当にお元気ですかそうですかありがとうございました"
    print(f"synthetic hit-doc overlap: {checker.check(hit_doc):.4f}  (should be ~1.0)")
    print(f"synthetic miss-doc overlap: {checker.check(miss_doc):.4f}  (should be ~0.0)")

    sample_paths = [Path(p) for p in args.sample_jsonl]
    if not sample_paths:
        print("no --sample-jsonl provided; skipping corpus-sample scan")
        return

    docs = _iter_sample(sample_paths, args.max_docs)
    if not docs:
        print("no docs loaded from sample paths")
        return

    print(f"sample docs scanned: {len(docs)}")
    overlaps = [checker.check(d) for d in docs]
    over_thresh = sum(1 for o in overlaps if o >= args.threshold)
    print(f"overlap distribution: mean={sum(overlaps)/len(overlaps):.4f}"
          f" max={max(overlaps):.4f} >={args.threshold}: {over_thresh}")


if __name__ == "__main__":
    main()
