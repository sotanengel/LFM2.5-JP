"""ChatML formatting and SFT loss-masking (Issue #19).

LFM2.5's chat template is ChatML-like: ``<|im_start|>{role}\\n{content}<|im_end|>\\n``
(docs/lfm2_5-ja-plan.md sec 1.1 / sec 3). This module renders ``messages`` lists into
that format, tokenizes them into SFT training examples with a loss mask that only
scores assistant responses, and provides tools to sanity-check the mask by decoding it.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = {"system", "user", "assistant"}
_ASSISTANT_TAG = "<|im_start|>assistant\n"
_LEARNED_START = "【learned】"  # 【learned】
_LEARNED_END = "【/learned】"  # 【/learned】


def to_chatml(messages: list[dict[str, Any]]) -> str:
    """Render a ``[{"role": ..., "content": ...}, ...]`` list as a ChatML string.

    Only ``system``/``user``/``assistant`` roles are allowed; anything else raises
    ``ValueError``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {role!r}. Allowed roles: {sorted(_ALLOWED_ROLES)}")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text)["input_ids"])


def _encode_full(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Encode the full ``messages`` sequence.

    Uses the tokenizer's own ``apply_chat_template`` when available (the real HF
    tokenizer path); otherwise falls back to rendering ``to_chatml`` and calling the
    tokenizer directly.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return list(ids)
    return _encode_text(tokenizer, to_chatml(messages))


def build_sft_example(
    messages: list[dict[str, Any]], tokenizer: Any, max_seq_len: int
) -> dict[str, list[int]]:
    """Build a tokenized SFT example with a loss mask over assistant responses only.

    Returns ``{"input_ids": [...], "labels": [...], "attention_mask": [...]}`` where
    ``labels`` is ``-100`` everywhere except the token span(s) covering each
    assistant message's content, truncated to ``max_seq_len``.

    Assistant span detection: for each assistant message we independently encode
    (a) the ChatML prefix up to and including that message's opening
    ``<|im_start|>assistant\\n`` tag, and (b) the raw message content. The token
    counts of (a) and (b) give the start/end offsets of that message's content
    inside the fully-encoded sequence. This works for any tokenizer exposing the
    standard ``tokenizer(text) -> {"input_ids": [...]}`` call convention and does
    not require the tokenizer to support ``apply_chat_template``.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    for msg in messages:
        if msg.get("role") not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {msg.get('role')!r}")

    full_ids = _encode_full(tokenizer, messages)
    labels = [-100] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_text = to_chatml(messages[:i]) + _ASSISTANT_TAG
        content_text = msg.get("content", "")
        prefix_ids = _encode_text(tokenizer, prefix_text)
        content_ids = _encode_text(tokenizer, content_text)

        start = min(len(prefix_ids), len(full_ids))
        end = min(start + len(content_ids), len(full_ids))
        for j in range(start, end):
            labels[j] = full_ids[j]

    full_ids = full_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    attention_mask = [1] * len(full_ids)

    return {"input_ids": full_ids, "labels": labels, "attention_mask": attention_mask}


def decode_for_inspection(example: dict[str, list[int]], tokenizer: Any) -> str:
    """Decode ``example`` for visual verification of the loss mask (Issue #19).

    Spans where ``labels != -100`` (i.e. tokens that contribute to the loss) are
    wrapped in ``【learned】...【/learned】`` markers.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    pieces: list[str] = []
    i = 0
    n = len(input_ids)
    while i < n:
        learned = labels[i] != -100
        j = i
        while j < n and (labels[j] != -100) == learned:
            j += 1
        segment_text = tokenizer.decode(input_ids[i:j])
        if learned:
            pieces.append(f"{_LEARNED_START}{segment_text}{_LEARNED_END}")
        else:
            pieces.append(segment_text)
        i = j
    return "".join(pieces)


def token_count_stats(examples: list[dict[str, list[int]]]) -> dict[str, Any]:
    """Compute min/max/mean/median stats for input length and learned-token count."""

    def _stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    if not examples:
        return {"count": 0, "input_length": _stats([]), "learned_tokens": _stats([])}

    lengths = [len(ex["input_ids"]) for ex in examples]
    learned_counts = [sum(1 for lab in ex["labels"] if lab != -100) for ex in examples]
    return {
        "count": len(examples),
        "input_length": _stats(lengths),
        "learned_tokens": _stats(learned_counts),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert a JSONL of chat messages into tokenized SFT examples (Issue #19)"
    )
    parser.add_argument("--input", required=True, help="Input JSONL path (with a 'messages' field)")
    parser.add_argument("--output", required=True, help="Output JSONL path for SFT examples")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer id or local path")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Truncation length")
    parser.add_argument(
        "--inspect",
        type=int,
        default=0,
        help="Print decode_for_inspection() for the first N examples",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    docs = _read_jsonl(args.input)
    examples = [build_sft_example(doc["messages"], tokenizer, args.max_seq_len) for doc in docs]
    _write_jsonl(args.output, examples)

    stats = token_count_stats(examples)
    logger.info("Formatted %d examples -> %s", len(examples), args.output)
    logger.info("Token count stats: %s", json.dumps(stats, ensure_ascii=False))

    if args.inspect:
        for example in examples[: args.inspect]:
            print(decode_for_inspection(example, tokenizer))
            print("---")


if __name__ == "__main__":
    main()
