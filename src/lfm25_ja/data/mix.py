"""Language-ratio corpus mixing for the CPT mixture (Issue #18).

Given per-language document pools, samples a mixture that honors a target
language ratio (e.g. 85% Japanese / 15% English to guard against catastrophic
forgetting, see docs/lfm2_5-ja-plan.md sec 3). The overall mixture size is
capped by whichever language is scarcest relative to its target ratio, so the
requested ratio is always respected exactly (in document/token counts) and
any surplus from the more abundant language(s) is simply left unused.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)


def _normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("ratios must not contain negative values")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("ratios must sum to a positive value")
    return {lang: value / total for lang, value in ratios.items()}


def _lang_volume(docs: list[dict[str, Any]], unit: str, token_field: str, lang: str) -> float:
    if unit == "documents":
        return float(len(docs))
    if unit == "tokens":
        total = 0
        for doc in docs:
            if token_field not in doc:
                raise ValueError(
                    f"Document missing token field '{token_field}' for language '{lang}'"
                )
            total += doc[token_field]
        return float(total)
    raise ValueError(f"Unknown unit: {unit!r} (expected 'documents' or 'tokens')")


def mix_corpora(
    docs_by_lang: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    seed: int,
    unit: str = "documents",
    token_field: str = "n_tokens",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample a language-balanced mixture from ``docs_by_lang``.

    ``ratios`` need not sum to 1 (normalized internally). The overall mixture
    volume is bounded by the most-constrained language relative to its ratio,
    so the target ratio is respected exactly and leftover documents from
    other languages are simply not used. Selection and ordering are both
    deterministic given ``seed``.

    Returns ``(mixed_docs, stats)`` where ``stats`` reports, per language,
    ``selected``/``available`` (in the chosen ``unit``) and
    ``ratio_target``/``ratio_actual``, plus the overall ``total_selected``.
    """
    norm_ratios = _normalize_ratios(ratios)
    languages = sorted(norm_ratios.keys())

    available: dict[str, float] = {
        lang: _lang_volume(docs_by_lang.get(lang, []), unit, token_field, lang)
        for lang in languages
    }

    capacities = [
        available[lang] / norm_ratios[lang] for lang in languages if norm_ratios[lang] > 0
    ]
    total_volume = min(capacities) if capacities else 0.0

    target_volume = {lang: total_volume * norm_ratios[lang] for lang in languages}

    rng = random.Random(seed)
    selected_docs: list[dict[str, Any]] = []
    selected_volume: dict[str, float] = {}

    for lang in languages:
        lang_docs = list(docs_by_lang.get(lang, []))
        idxs = list(range(len(lang_docs)))
        rng.shuffle(idxs)

        if unit == "documents":
            n_select = int(target_volume[lang])
            chosen = idxs[:n_select]
            selected_volume[lang] = float(len(chosen))
        else:  # tokens
            budget = target_volume[lang]
            chosen = []
            running = 0.0
            for idx in idxs:
                tokens = lang_docs[idx][token_field]
                if running + tokens > budget:
                    break
                chosen.append(idx)
                running += tokens
            selected_volume[lang] = running

        for idx in chosen:
            selected_docs.append(lang_docs[idx])

    rng.shuffle(selected_docs)

    total_selected = sum(selected_volume.values())
    languages_stats: dict[str, Any] = {}
    for lang in languages:
        sel = selected_volume[lang]
        avail = available[lang]
        ratio_actual = sel / total_selected if total_selected else 0.0
        languages_stats[lang] = {
            "selected": int(sel) if unit == "documents" else sel,
            "available": int(avail) if unit == "documents" else avail,
            "ratio_target": norm_ratios[lang],
            "ratio_actual": ratio_actual,
        }

    stats = {
        "unit": unit,
        "seed": seed,
        "total_selected": int(total_selected) if unit == "documents" else total_selected,
        "languages": languages_stats,
    }
    return selected_docs, stats


def render_mix_report(stats: dict[str, Any]) -> str:
    """Render a markdown summary table of mixing stats."""
    lines = [
        "# Corpus mixing report",
        "",
        f"- Unit: {stats.get('unit', 'documents')}",
        f"- Seed: {stats.get('seed')}",
        f"- Total selected: {stats.get('total_selected', 0)}",
        "",
        "| language | selected | available | ratio_target | ratio_actual |",
        "|---|---|---|---|---|",
    ]
    for lang, s in stats.get("languages", {}).items():
        lines.append(
            f"| {lang} | {s['selected']} | {s['available']} "
            f"| {s['ratio_target']:.2%} | {s['ratio_actual']:.2%} |"
        )
    return "\n".join(lines)


def _parse_inputs(pairs: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --inputs entry (expected lang=path): {pair}")
        lang, path = pair.split("=", 1)
        result[lang] = path
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Mix per-language corpora by ratio (Issue #18)")
    parser.add_argument("--config", required=True, help="Path to configs/data/corpus.yaml")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="lang=path.jsonl pairs, e.g. --inputs ja=ja.jsonl en=en.jsonl",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path for the mixed corpus")
    parser.add_argument("--report", default=None, help="Optional markdown report output path")
    args = parser.parse_args()

    config = load_config(args.config)
    mix_cfg = config.get("mix", config)

    input_paths = _parse_inputs(args.inputs)
    docs_by_lang = {lang: _read_jsonl(path) for lang, path in input_paths.items()}

    mixed_docs, stats = mix_corpora(
        docs_by_lang,
        ratios=mix_cfg["ratios"],
        seed=mix_cfg.get("seed", 42),
        unit=mix_cfg.get("unit", "documents"),
    )
    _write_jsonl(args.output, mixed_docs)
    logger.info("Mixed corpus written: %d documents -> %s", len(mixed_docs), args.output)

    if args.report:
        report = render_mix_report(stats)
        Path(args.report).write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.report)


if __name__ == "__main__":
    main()
