"""ichikara-instruction dataset acquisition + chat-format conversion (Issue #33).

ichikara-instruction is a Japanese instruction dataset released by
LIAT-AIP/RIKEN under the CC-BY-NC-SA license (see
http://liat-aip.sakura.ne.jp/wp/llm%e3%81%ae%e3%81%9f%e3%82%81%e3%81%ae%e6%97%a5%e6%9c%ac%e8%aa%9e%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%a9%e3%82%af%e3%82%b7%e3%83%a7%e3%83%b3%e3%83%87%e3%83%bc%e3%82%bf%e4%bd%9c%e6%88%90/).
This project uses the ``kinokokoro/ichikara-instruction-003`` community
mirror on the HuggingFace Hub, which republishes the official 003-batch
release under the same license. That repo is *not* gated (verified via
``GET https://huggingface.co/api/datasets/kinokokoro/ichikara-instruction-003``
-> ``"gated": false``), so it can be pulled with the same frictionless
``datasets.load_dataset`` flow already used by ``lfm25_ja.data.download`` --
no manual license click-through / form submission is required.

Each raw record is a single-turn Q/A pair (``text`` = question/instruction,
``output`` = answer); :func:`convert_ichikara_record` maps that into the
``{"messages": [...]}`` chat schema consumed by
``lfm25_ja.data.format_chat.build_sft_example``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import datasets

from lfm25_ja.data.clean import _write_jsonl

logger = logging.getLogger(__name__)

ICHIKARA_HF_REPO = "kinokokoro/ichikara-instruction-003"

# Batch 001 shards (1/2/5-annotator-response variants) from the 003 release
# (see the dataset repo's README.md). These parse cleanly as JSON (verified
# with Python's stdlib json.loads) and total 2,903 Q/A pairs.
ICHIKARA_FILES: tuple[str, ...] = (
    "ichikara-instruction-003-001-1.json",
    "ichikara-instruction-003-001-2.1.json",
    "ichikara-instruction-003-001-2.2.json",
    "ichikara-instruction-003-001-5.1.json",
    "ichikara-instruction-003-001-5.2.json",
)

# NOTE (Issue #33): batches 002/003 (single-response, published 2024-12-21;
# ichikara-instruction-003-002-1.json / -003-1.json, ~1,999 additional Q/A
# pairs) are intentionally EXCLUDED from ICHIKARA_FILES. Both shards contain
# malformed JSON in this mirror -- confirmed independently with Python's
# stdlib json.loads (not just datasets' stricter pandas/ujson backend),
# failing on invalid `\` escape sequences around embedded code-sample text
# (e.g. a BASIC/VBA snippet using `\` as a path separator). The invalid
# escapes are interleaved with legitimate `\\` escapes in the same string,
# so a generic "double every stray backslash" repair silently corrupts
# adjacent valid escapes instead of fixing the file -- this would require a
# hand-written, string-aware repair of specific corrupted records, which is
# out of scope here. If a future issue wants that content, fix each file's
# JSON by hand (or find a non-corrupted mirror) and add it back explicitly.
ICHIKARA_MALFORMED_FILES: tuple[str, ...] = (
    "ichikara-instruction-003-002-1.json",
    "ichikara-instruction-003-003-1.json",
)


def _resolve_urls(
    files: tuple[str, ...] = ICHIKARA_FILES, repo: str = ICHIKARA_HF_REPO
) -> list[str]:
    return [f"https://huggingface.co/datasets/{repo}/resolve/main/{name}" for name in files]


def download_ichikara_raw(
    cache_dir: str | Path = "data/raw/ichikara",
    files: tuple[str, ...] = ICHIKARA_FILES,
    repo: str = ICHIKARA_HF_REPO,
) -> list[dict[str, Any]]:
    """Download every ichikara-instruction-003 JSON shard and return the
    concatenated raw records (each a ``{"ID", "text", "output"}`` dict).
    """
    urls = _resolve_urls(files, repo)
    try:
        dataset = datasets.load_dataset(
            "json", data_files=urls, split="train", cache_dir=str(cache_dir)
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with dataset context below
        raise RuntimeError(f"Failed to download ichikara-instruction ({repo}): {exc}") from exc
    return [dict(row) for row in dataset]


def convert_ichikara_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw ichikara record (``text``=question, ``output``=answer)
    into the ``{"messages": [...]}`` chat schema ``build_sft_example`` expects.
    """
    missing = [key for key in ("text", "output") if key not in record]
    if missing:
        raise ValueError(
            f"ichikara record missing field(s) {missing}: keys={list(record.keys())}"
        )
    return {
        "messages": [
            {"role": "user", "content": record["text"]},
            {"role": "assistant", "content": record["output"]},
        ]
    }


def build_ichikara_dataset(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw ichikara records into chat-format rows, skipping (and
    logging) any record missing ``text``/``output`` rather than aborting the
    whole run.
    """
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(raw_records):
        try:
            rows.append(convert_ichikara_record(record))
        except ValueError as exc:
            logger.warning("Skipping malformed ichikara record %d: %s", i, exc)
    return rows


def prepare_ichikara(
    output_path: str | Path = "data/processed/sft/ichikara.jsonl",
    cache_dir: str | Path = "data/raw/ichikara",
    files: tuple[str, ...] = ICHIKARA_FILES,
    repo: str = ICHIKARA_HF_REPO,
) -> dict[str, Any]:
    """End-to-end: download the ichikara-instruction-003 shards, convert them
    to chat-format JSONL, and write the result to ``output_path``.

    Returns ``{"input_count", "output_count", "output_path"}`` where
    ``input_count`` is the number of raw records downloaded and
    ``output_count`` is the number successfully converted (``<= input_count``
    if any malformed records were skipped).
    """
    raw_records = download_ichikara_raw(cache_dir=cache_dir, files=files, repo=repo)
    rows = build_ichikara_dataset(raw_records)
    _write_jsonl(output_path, rows)
    logger.info(
        "ichikara: %d raw records -> %d chat-format rows -> %s",
        len(raw_records),
        len(rows),
        output_path,
    )
    return {
        "input_count": len(raw_records),
        "output_count": len(rows),
        "output_path": str(output_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare the ichikara-instruction SFT dataset (Issue #33)"
    )
    parser.add_argument(
        "--output", default="data/processed/sft/ichikara.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--cache-dir", default="data/raw/ichikara", help="HF datasets cache dir"
    )
    args = parser.parse_args()

    result = prepare_ichikara(output_path=args.output, cache_dir=args.cache_dir)
    logger.info("Done: %d examples -> %s", result["output_count"], result["output_path"])


if __name__ == "__main__":
    main()
