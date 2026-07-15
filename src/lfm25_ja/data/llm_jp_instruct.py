"""llm-jp-instruct dataset acquisition + chat-format conversion (Issue #105).

Data-diversification component of the sft-002 mix (Fable5 consult, agentId
``a2cdb09320bf6f720``): sft-002 adds a randomly-sampled slice of a llm-jp
instruction corpus alongside the existing ichikara-instruction-003 data.

**Fallback note (verified at implementation time via
``GET https://huggingface.co/api/datasets/<repo>``):**

- ``llm-jp/llm-jp-instruct-v1`` (the repo id originally specified for this
  task) **does not exist** on the HuggingFace Hub -- the API returns no match
  and a plain HTTP fetch of the dataset page 404s. There is no gating
  involved; the repo id is simply not a real dataset.
- The closest same-family real dataset, ``llm-jp/llm-jp-instructions``
  (``gated: false``, license ``cc-by-4.0``), only has 1,000 total rows across
  its train/dev/test splits -- far short of the ``n_samples=4000`` target, so
  it was rejected too.
- This module therefore falls back to **``llm-jp/oasst1-21k-ja``**
  (``gated: false``, license ``apache-2.0`` -- a Japanese DeepL translation of
  an English oasst1 subset, published by the llm-jp collaborative project).
  It is non-gated, ~21k rows (comfortably above the 4,000-sample target), and
  Apache-2.0 is strictly more permissive than the CC-BY-SA family requested
  for this slot (no NC clause, no share-alike obligation) -- see
  ``https://huggingface.co/api/datasets/llm-jp/oasst1-21k-ja``.

Each raw record uses oasst1-21k-ja's ``conversations`` schema: a list of
``{"from": "human"|"gpt", "value": ...}`` turns. :func:`convert_llm_jp_instruct_record`
also accepts a generic ``instruction``/``input``/``output`` single-turn schema
(some llm-jp-family mirrors use that shape instead) so this loader is portable
if the source repo is swapped again later. Both are mapped into the
``{"messages": [...]}`` chat schema consumed by
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

LLM_JP_INSTRUCT_HF_REPO = "llm-jp/oasst1-21k-ja"
LLM_JP_INSTRUCT_FILE = "oasst1-21k-ja.jsonl"

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
}


def _resolve_url(
    filename: str = LLM_JP_INSTRUCT_FILE, repo: str = LLM_JP_INSTRUCT_HF_REPO
) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


def download_llm_jp_instruct_raw(
    cache_dir: str | Path = "data/raw/llm_jp_instruct",
    filename: str = LLM_JP_INSTRUCT_FILE,
    repo: str = LLM_JP_INSTRUCT_HF_REPO,
) -> list[dict[str, Any]]:
    """Download the llm-jp-instruct source file and return the raw records."""
    url = _resolve_url(filename, repo)
    try:
        dataset = datasets.load_dataset(
            "json", data_files=[url], split="train", cache_dir=str(cache_dir)
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with dataset context below
        raise RuntimeError(f"Failed to download llm-jp-instruct ({repo}): {exc}") from exc
    return [dict(row) for row in dataset]


def _convert_conversations_record(record: dict[str, Any]) -> dict[str, Any]:
    turns = record["conversations"]
    if not turns:
        raise ValueError("record['conversations'] is empty")
    messages: list[dict[str, str]] = []
    for turn in turns:
        from_ = turn.get("from")
        role = _ROLE_MAP.get(from_)
        if role is None:
            raise ValueError(f"Unknown conversation turn role: {from_!r}")
        messages.append({"role": role, "content": turn.get("value", "")})
    if not any(m["role"] == "user" for m in messages) or not any(
        m["role"] == "assistant" for m in messages
    ):
        raise ValueError("record['conversations'] has no user/assistant turn pair")
    return {"messages": messages}


def _convert_instruction_record(record: dict[str, Any]) -> dict[str, Any]:
    instruction = record["instruction"]
    input_ = record.get("input", "")
    output = record["output"]
    user_content = f"{instruction}\n\n{input_}" if input_ else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def convert_llm_jp_instruct_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw llm-jp-instruct record into the ``{"messages": [...]}``
    chat schema ``build_sft_example`` expects.

    Supports both the ``conversations`` (multi-turn ``from``/``value`` list)
    schema used by ``llm-jp/oasst1-21k-ja`` and a generic
    ``instruction``/``input``/``output`` single-turn schema, in case the
    source repo is swapped for a different llm-jp-family mirror later.
    """
    if "conversations" in record:
        return _convert_conversations_record(record)
    if "instruction" in record and "output" in record:
        return _convert_instruction_record(record)
    raise ValueError(
        "llm-jp-instruct record has neither 'conversations' nor "
        f"'instruction'/'output' fields: keys={list(record.keys())}"
    )


def build_llm_jp_instruct_dataset(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw llm-jp-instruct records into chat-format rows, skipping (and
    logging) any malformed record rather than aborting the whole run."""
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(raw_records):
        try:
            rows.append(convert_llm_jp_instruct_record(record))
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping malformed llm-jp-instruct record %d: %s", i, exc)
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


def prepare_llm_jp_instruct(
    output_path: str | Path = "data/processed/sft/llm_jp_instruct.jsonl",
    cache_dir: str | Path = "data/raw/llm_jp_instruct",
    n_samples: int = 4000,
    seed: int = 42,
    filename: str = LLM_JP_INSTRUCT_FILE,
    repo: str = LLM_JP_INSTRUCT_HF_REPO,
) -> dict[str, Any]:
    """End-to-end: download llm-jp-instruct, convert to chat-format JSONL,
    randomly sample ``n_samples`` rows (seeded), and write the result to
    ``output_path``.

    Returns ``{"input_count", "converted_count", "output_count", "output_path"}``.
    """
    raw_records = download_llm_jp_instruct_raw(cache_dir=cache_dir, filename=filename, repo=repo)
    rows = build_llm_jp_instruct_dataset(raw_records)
    sampled = sample_rows(rows, n_samples, seed)
    _write_jsonl(output_path, sampled)
    logger.info(
        "llm-jp-instruct: %d raw records -> %d chat-format rows -> %d sampled -> %s",
        len(raw_records),
        len(rows),
        len(sampled),
        output_path,
    )
    return {
        "input_count": len(raw_records),
        "converted_count": len(rows),
        "output_count": len(sampled),
        "output_path": str(output_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare the llm-jp-instruct SFT dataset slice (Issue #105)"
    )
    parser.add_argument(
        "--output", default="data/processed/sft/llm_jp_instruct.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--cache-dir", default="data/raw/llm_jp_instruct", help="HF datasets cache dir"
    )
    parser.add_argument("--n-samples", type=int, default=4000, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    result = prepare_llm_jp_instruct(
        output_path=args.output, cache_dir=args.cache_dir, n_samples=args.n_samples, seed=args.seed
    )
    logger.info("Done: %d examples -> %s", result["output_count"], result["output_path"])


if __name__ == "__main__":
    main()
