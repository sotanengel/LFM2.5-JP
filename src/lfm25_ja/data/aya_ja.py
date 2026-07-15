"""Aya Dataset (Japanese subset) acquisition + chat-format conversion (Issue #105).

Data-diversification component of the sft-002 mix (Fable5 consult, agentId
``a2cdb09320bf6f720``): sft-002 adds a randomly-sampled slice of the
Japanese-language portion of Cohere Labs' multilingual Aya Dataset.

**License/gating check (verified at implementation time via
``GET https://huggingface.co/api/datasets/CohereForAI/aya_dataset``):**
``gated: false``, license ``apache-2.0``. Note the repo id has since been
renamed to ``CohereLabs/aya_dataset`` on the Hub (the HF API 307-redirects
``CohereForAI/aya_dataset`` requests there transparently); this module keeps
the originally-specified ``CohereForAI/aya_dataset`` id since it still
resolves correctly, and ``datasets.load_dataset`` follows the redirect.

The ``default`` config's schema (confirmed via the same API call) is
``{"inputs": str, "targets": str, "language": str, "language_code": str,
"annotation_type": str, "user_id": str}`` -- a flat, single-turn
prompt/completion pair with no ``instruction``/``input``/``output`` split.
Japanese rows are selected via ``language_code == "jpn"`` (falling back to
``language == "Japanese"`` for robustness against schema drift), then mapped
into the ``{"messages": [...]}`` chat schema consumed by
``lfm25_ja.data.format_chat.build_sft_example``.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import datasets

from lfm25_ja.data.clean import _write_jsonl

logger = logging.getLogger(__name__)

AYA_HF_REPO = "CohereForAI/aya_dataset"


def download_aya_raw(
    cache_dir: str | Path = "data/raw/aya_ja",
    repo: str = AYA_HF_REPO,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Download the Aya Dataset ``default`` config and return the raw records
    (all languages -- filtering to Japanese happens in :func:`filter_japanese_records`)."""
    try:
        dataset = datasets.load_dataset(repo, split=split, cache_dir=str(cache_dir))
    except Exception as exc:  # noqa: BLE001 - re-raised with dataset context below
        raise RuntimeError(f"Failed to download aya_dataset ({repo}): {exc}") from exc
    return [dict(row) for row in dataset]


def _is_japanese(record: dict[str, Any]) -> bool:
    if record.get("language_code") == "jpn":
        return True
    return record.get("language") == "Japanese"


def filter_japanese_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only Japanese-language rows (``language_code == "jpn"``)."""
    return [r for r in raw_records if _is_japanese(r)]


def convert_aya_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw Aya Dataset record (``inputs``=prompt, ``targets``=completion)
    into the ``{"messages": [...]}`` chat schema ``build_sft_example`` expects."""
    missing = [key for key in ("inputs", "targets") if key not in record]
    if missing:
        raise ValueError(f"aya record missing field(s) {missing}: keys={list(record.keys())}")
    return {
        "messages": [
            {"role": "user", "content": record["inputs"]},
            {"role": "assistant", "content": record["targets"]},
        ]
    }


def build_aya_ja_dataset(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to Japanese rows and convert them into chat-format rows, skipping
    (and logging) any malformed record rather than aborting the whole run."""
    ja_records = filter_japanese_records(raw_records)
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(ja_records):
        try:
            rows.append(convert_aya_record(record))
        except ValueError as exc:
            logger.warning("Skipping malformed aya record %d: %s", i, exc)
    return rows


def sample_rows(rows: list[dict[str, Any]], n_samples: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically sample up to ``n_samples`` rows (all of them, shuffled,
    if fewer are available than requested)."""
    if n_samples >= len(rows):
        if n_samples > len(rows):
            logger.warning(
                "Requested n_samples=%d but only %d rows are available; using all of them",
                n_samples,
                len(rows),
            )
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    return random.Random(seed).sample(rows, n_samples)


def prepare_aya_ja(
    output_path: str | Path = "data/processed/sft/aya_ja.jsonl",
    cache_dir: str | Path = "data/raw/aya_ja",
    n_samples: int = 1500,
    seed: int = 42,
    repo: str = AYA_HF_REPO,
) -> dict[str, Any]:
    """End-to-end: download the Aya Dataset, filter to Japanese, convert to
    chat-format JSONL, randomly sample ``n_samples`` rows (seeded), and write
    the result to ``output_path``.

    Returns ``{"input_count", "ja_count", "output_count", "output_path"}``.
    """
    raw_records = download_aya_raw(cache_dir=cache_dir, repo=repo)
    rows = build_aya_ja_dataset(raw_records)
    sampled = sample_rows(rows, n_samples, seed)
    _write_jsonl(output_path, sampled)
    logger.info(
        "aya_ja: %d raw records -> %d ja chat-format rows -> %d sampled -> %s",
        len(raw_records),
        len(rows),
        len(sampled),
        output_path,
    )
    return {
        "input_count": len(raw_records),
        "ja_count": len(rows),
        "output_count": len(sampled),
        "output_path": str(output_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare the aya_dataset (Japanese subset) SFT dataset slice (Issue #105)"
    )
    parser.add_argument(
        "--output", default="data/processed/sft/aya_ja.jsonl", help="Output JSONL path"
    )
    parser.add_argument("--cache-dir", default="data/raw/aya_ja", help="HF datasets cache dir")
    parser.add_argument("--n-samples", type=int, default=1500, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    result = prepare_aya_ja(
        output_path=args.output, cache_dir=args.cache_dir, n_samples=args.n_samples, seed=args.seed
    )
    logger.info("Done: %d examples -> %s", result["output_count"], result["output_path"])


if __name__ == "__main__":
    main()
