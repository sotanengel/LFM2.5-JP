"""Assemble JKB v1 train/eval splits from the 12 per-domain _authored/*.jsonl files.

Split is deterministic: SHA1(id) % 5 == 0 → train; else eval. This keeps
the domain × difficulty distribution roughly balanced across splits without
per-cell stratification (each cell has 14 rows, ~1/5 = 2-3 rows into train
and 11-12 into eval).

Also emits `eval_texts.jsonl` (each row's `prompt + " " + best_answer`
under a `text` field) so `prepare.py --eval-texts` can use it to filter
contaminated docs out of CPT corpora during K2 (Issue #121).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DOMAINS = ("geo", "hist", "lit", "food", "trad", "pol", "life",
           "region", "sport", "sci", "relig", "lang")


def _split_bucket(row_id: str) -> str:
    return "train" if int(hashlib.sha1(row_id.encode("utf-8")).hexdigest(), 16) % 5 == 0 else "eval"


def _best_answer(row: dict) -> str:
    if row["format"] == "short_answer":
        return row["answers"][0] if row["answers"] else ""
    for c in row["choices"]:
        if c["label"] == row["correct_choice"]:
            return c["text"]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble JKB v1 splits")
    parser.add_argument("--authored-dir", default="datasets/eval/jkb/_authored")
    parser.add_argument("--out-dir", default="datasets/eval/jkb")
    args = parser.parse_args()

    authored = Path(args.authored_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for slug in DOMAINS:
        p = authored / f"{slug}.jsonl"
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_rows.append(json.loads(line))

    train, ev = [], []
    for r in all_rows:
        (train if _split_bucket(r["id"]) == "train" else ev).append(r)

    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (out / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ev) + "\n", encoding="utf-8"
    )

    eval_texts = [
        {"id": r["id"], "text": r["prompt"] + " " + _best_answer(r)}
        for r in all_rows
    ]
    (out / "eval_texts.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_texts) + "\n", encoding="utf-8"
    )

    def _dist(rows: list[dict]) -> dict:
        d: dict = {}
        for r in rows:
            key = (r["domain"], r["difficulty"])
            d[key] = d.get(key, 0) + 1
        return d

    print(f"total: {len(all_rows)}  train: {len(train)}  eval: {len(ev)}")
    print("train per-cell:", sorted(_dist(train).items()))
    print("eval per-cell:", sorted(_dist(ev).items()))


if __name__ == "__main__":
    main()
