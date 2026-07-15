"""Format-constraint synthesis for the sft-002 mix (Issue #105).

Fable5 consult (agentId ``a2cdb09320bf6f720``) recommended base self-distillation
to generate format-constrained (char_count / bullet_count / format_json /
polite_form / keyword) training examples, but that requires GPU generation.
This module implements the agreed selection+synthesis hybrid instead: existing
Q/A pairs already drawn from the upstream mix (ichikara + llm-jp-instruct +
aya-ja) that *already satisfy* a target verifier are selected, the verifier's
instruction is appended to the prompt, and the (unmodified, already-compliant)
answer is kept as-is. Each candidate is re-checked against the corresponding
``lfm25_ja.eval.instruction_verifiers`` function before being accepted --
only verifier-passing examples are used. This sacrifices the on-policy
benefit of self-distillation but guarantees on-distribution, verifier-clean
training signal without any GPU generation step.

``format_markdown_table`` and ``numeric_only`` are intentionally excluded
(Fable5 instruction: kept as a held-out generalization probe for the ifeval_ja
harness, see Issue #104's ``configs/eval/ifeval_ja.yaml``).
"""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from typing import Any, Callable

from lfm25_ja.eval.instruction_verifiers import (
    verify_bullet_count,
    verify_char_count,
    verify_format_json,
    verify_keyword,
    verify_polite_form,
)

logger = logging.getLogger(__name__)

# Same bullet-line shape as instruction_verifiers._BULLET_LINE (kept as an
# independent literal here rather than importing a private name across
# modules -- instruction_verifiers.py is frozen/do-not-touch, Issue #105).
_BULLET_LINE = re.compile(r"^\s*[-・*]|^\s*\d+\.")

_KATAKANA_RUN = re.compile(r"[ァ-ヴー]{2,}")
_KANJI_RUN = re.compile(r"[一-鿿]{2,4}")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _extract_length(response: str) -> int:
    """Character count of ``response`` after NFKC normalization."""
    return len(_nfkc(response))


def _count_bullets(response: str) -> int:
    """Number of lines in ``response`` matching the bullet-line shape."""
    text = _nfkc(response)
    return sum(1 for line in text.splitlines() if _BULLET_LINE.match(line))


def _extract_keyword_noun(response: str) -> str | None:
    """Extract a machine-identifiable "noun phrase" from ``response``: a
    katakana run of length >= 2, or a kanji run of length 2-4. Returns the
    most frequent candidate (ties broken by longer length, then first
    occurrence), or ``None`` if no candidate is found."""
    text = _nfkc(response)
    candidates = _KATAKANA_RUN.findall(text) + _KANJI_RUN.findall(text)
    if not candidates:
        return None
    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for idx, cand in enumerate(candidates):
        counts[cand] = counts.get(cand, 0) + 1
        first_index.setdefault(cand, idx)
    best = max(
        counts,
        key=lambda c: (counts[c], len(c), -first_index[c]),
    )
    return best


def _extract_qa(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Return ``(question, answer)`` = the last user turn immediately preceding
    the final assistant turn, and that assistant turn's content. Returns
    ``None`` if ``messages`` doesn't end in an assistant turn with a preceding
    user turn (multi-turn conversations are reduced to their final exchange)."""
    if not messages or messages[-1].get("role") != "assistant":
        return None
    answer = messages[-1].get("content", "")
    question = None
    for msg in reversed(messages[:-1]):
        if msg.get("role") == "user":
            question = msg.get("content", "")
            break
    if question is None:
        return None
    return question, answer


def _build_result(
    question: str, answer: str, suffix: str, verifier_name: str, source_row: dict[str, Any]
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"{question}\n\n{suffix}"},
            {"role": "assistant", "content": answer},
        ],
        "verifier": verifier_name,
        "origin": source_row.get("origin"),
    }


def synthesize_char_count(source_row: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Select ``source_row`` for the ``char_count`` verifier if its answer is
    20-300 characters, appending a "{length+buffer}文字以内で答えてください"
    instruction (buffer randomly 0-20, seeded via ``rng``)."""
    qa = _extract_qa(source_row.get("messages", []))
    if qa is None:
        return None
    question, answer = qa
    length = _extract_length(answer)
    if not (20 <= length <= 300):
        return None
    buffer = rng.randint(0, 20)
    limit = length + buffer
    ok, msg = verify_char_count(answer, {"max": limit})
    if not ok:
        logger.warning("synthesize_char_count: constructed constraint failed verify (%s)", msg)
        return None
    suffix = f"{limit}文字以内で答えてください。"
    return _build_result(question, answer, suffix, "char_count", source_row)


def synthesize_bullet_count(
    source_row: dict[str, Any], rng: random.Random
) -> dict[str, Any] | None:
    """Select ``source_row`` for the ``bullet_count`` verifier if its answer is
    already a 2-5 item bulleted list, appending a "箇条書き{N}項目で答えて
    ください" instruction."""
    qa = _extract_qa(source_row.get("messages", []))
    if qa is None:
        return None
    question, answer = qa
    count = _count_bullets(answer)
    if not (2 <= count <= 5):
        return None
    ok, msg = verify_bullet_count(answer, {"count": count})
    if not ok:
        logger.warning("synthesize_bullet_count: constructed constraint failed verify (%s)", msg)
        return None
    suffix = f"箇条書き{count}項目で答えてください。"
    return _build_result(question, answer, suffix, "bullet_count", source_row)


def synthesize_format_json(
    source_row: dict[str, Any], rng: random.Random
) -> dict[str, Any] | None:
    """Select ``source_row`` for the ``format_json`` verifier if its answer is
    50-200 characters, wrapping it as ``{"answer": "..."}`` and appending a
    JSON-format instruction."""
    qa = _extract_qa(source_row.get("messages", []))
    if qa is None:
        return None
    question, answer = qa
    length = _extract_length(answer)
    if not (50 <= length <= 200):
        return None
    wrapped = json.dumps({"answer": answer}, ensure_ascii=False)
    ok, msg = verify_format_json(wrapped, {})
    if not ok:
        logger.warning("synthesize_format_json: constructed constraint failed verify (%s)", msg)
        return None
    suffix = 'JSON 形式で回答してください。schema: {"answer": string}'
    return _build_result(question, wrapped, suffix, "format_json", source_row)


def synthesize_polite_form(
    source_row: dict[str, Any], rng: random.Random
) -> dict[str, Any] | None:
    """Select ``source_row`` for the ``polite_form`` verifier if its answer
    already passes ``verify_polite_form(style="polite")``, appending a
    敬体(です・ます調)instruction. Caller is responsible for restricting
    ``source_row`` candidates to the aya-ja/ichikara origin tags (llm-jp
    excluded, see Issue #105 mix design)."""
    qa = _extract_qa(source_row.get("messages", []))
    if qa is None:
        return None
    question, answer = qa
    ok, _msg = verify_polite_form(answer, {"style": "polite"})
    if not ok:
        return None
    suffix = "敬体(です・ます調)で答えてください。"
    return _build_result(question, answer, suffix, "polite_form", source_row)


def synthesize_keyword(source_row: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Select ``source_row`` for the ``keyword`` verifier if a noun phrase can
    be machine-extracted from its answer, appending a "回答には「{noun}」を
    必ず含めてください" instruction."""
    qa = _extract_qa(source_row.get("messages", []))
    if qa is None:
        return None
    question, answer = qa
    noun = _extract_keyword_noun(answer)
    if noun is None:
        return None
    ok, msg = verify_keyword(answer, {"include": [noun]})
    if not ok:
        logger.warning("synthesize_keyword: constructed constraint failed verify (%s)", msg)
        return None
    suffix = f"回答には「{noun}」を必ず含めてください。"
    return _build_result(question, answer, suffix, "keyword", source_row)


_SYNTH_DISPATCH: dict[str, Callable[[dict[str, Any], random.Random], dict[str, Any] | None]] = {
    "char_count": synthesize_char_count,
    "bullet_count": synthesize_bullet_count,
    "format_json": synthesize_format_json,
    "polite_form": synthesize_polite_form,
    "keyword": synthesize_keyword,
}

# Fixed evaluation order (matches the table in Issue #105 / the orchestrator
# prompt) -- kept independent of dict iteration order for determinism even if
# a config passes ``targets`` with different key ordering.
_VERIFIER_ORDER: tuple[str, ...] = (
    "char_count",
    "bullet_count",
    "format_json",
    "polite_form",
    "keyword",
)


def build_format_constrained_samples(
    source_rows: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
    polite_form_origins: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build format-constrained SFT rows by selecting+synthesizing from
    ``source_rows`` (each a ``{"messages": [...], "origin": ...}`` dict drawn
    from the upstream ichikara/llm-jp-instruct/aya-ja mix).

    For each verifier in ``targets`` (evaluated in the fixed order
    ``char_count, bullet_count, format_json, polite_form, keyword``),
    ``source_rows`` is shuffled (seeded) and each row is passed to the
    matching ``synthesize_*`` function until ``targets[verifier]`` samples are
    collected or the pool is exhausted. If a verifier's target can't be
    reached, a WARN is logged with the actual count and processing continues
    with the remaining verifiers.

    ``polite_form_origins`` restricts the ``polite_form`` verifier's candidate
    pool to rows whose ``origin`` is in that set (Issue #105: llm-jp-instruct
    is excluded from polite_form selection, low keigo ratio).
    """
    rng = random.Random(seed)
    collected: list[dict[str, Any]] = []

    for verifier_name in _VERIFIER_ORDER:
        target_n = targets.get(verifier_name)
        if not target_n:
            continue
        synth_fn = _SYNTH_DISPATCH[verifier_name]

        if verifier_name == "polite_form" and polite_form_origins:
            pool = [r for r in source_rows if r.get("origin") in polite_form_origins]
        else:
            pool = source_rows

        shuffled = list(pool)
        rng.shuffle(shuffled)

        matched: list[dict[str, Any]] = []
        for row in shuffled:
            result = synth_fn(row, rng)
            if result is not None:
                matched.append(result)
                if len(matched) >= target_n:
                    break

        if len(matched) < target_n:
            logger.warning(
                "format constraint '%s': only found %d/%d passing candidates",
                verifier_name,
                len(matched),
                target_n,
            )

        collected.extend(matched)

    return collected
