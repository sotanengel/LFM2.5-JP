"""End-to-end data preparation pipeline: download -> clean -> mix -> report (Issue #20).

Orchestrates the existing per-stage modules (``download.py``, ``clean.py``,
``mix.py``) into a single entry point that turns ``configs/data/corpus.yaml``
into a final, language-mixed training corpus under ``data/processed/``,
along with a markdown report summarizing every stage.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any, Iterable

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl, clean_corpus, render_stats_report
from lfm25_ja.data.download import download_corpus, load_corpus_config
from lfm25_ja.data.mix import mix_corpora, render_mix_report

logger = logging.getLogger(__name__)


def _extract_text_rows(dataset: Iterable[Any], sample_limit: int | None) -> list[dict[str, Any]]:
    """Convert a loaded dataset (HF ``Dataset`` or any iterable of mapping rows)
    into a list of ``{"id": ..., "text": ...}`` dicts.

    ``sample_limit`` (if given) caps how many leading rows are consumed, which
    keeps pilot/verification runs fast.
    """
    rows: list[dict[str, Any]] = []
    iterator: Iterable[Any] = iter(dataset)
    if sample_limit is not None:
        iterator = itertools.islice(iterator, sample_limit)
    for i, row in enumerate(iterator):
        if "text" not in row:
            raise ValueError(f"Row {i} is missing a 'text' field (keys={list(row.keys())})")
        rows.append({"id": str(i), "text": row["text"]})
    return rows


def _default_output_dir(config: dict[str, Any]) -> Path:
    """Derive the default ``data/processed``-style output dir from the corpus config."""
    if config.get("data_dir"):
        return Path(config["data_dir"]) / "processed"
    cache_dir = Path(config.get("cache_dir", "data/raw"))
    return cache_dir.parent / "processed"


def prepare_data(
    config_path: str,
    names: list[str] | None = None,
    sample_limit: int | None = None,
    output_dir: str | None = None,
    eval_texts_path: str | None = None,
    streaming: bool = False,
) -> dict[str, Any]:
    """Run the end-to-end data prep pipeline: download -> clean -> mix -> report.

    For each selected corpus in ``config_path`` (all of them if ``names`` is
    None): download it, extract its ``text`` field (optionally capped to the
    first ``sample_limit`` rows), clean it, and bucket the cleaned documents
    by detected language. The per-language buckets are then mixed according
    to the config's ``mix`` section and written to ``<output_dir>/mixture.jsonl``
    (``output_dir`` defaults to a ``data/processed``-style directory derived
    from the config). A markdown report covering every stage is written to
    ``<output_dir>/prepare_report.md``.

    ``streaming`` forwards to ``download_corpus`` (Issue #69) so that large
    corpora (wikipedia_ja, cc100_ja) are pulled as a HF ``IterableDataset``
    instead of being fully materialized on disk. This is a single global CLI
    flag applied to every selected corpus -- there is intentionally no
    per-entry ``streaming:`` override in ``corpus.yaml`` (simplicity over
    flexibility; revisit if a mixed streaming/non-streaming run is needed).
    Because an ``IterableDataset`` has no defined length, ``streaming=True``
    requires ``sample_limit`` to be set, otherwise ``_extract_text_rows``
    would consume the stream forever; this raises ``ValueError`` up front.

    Returns a stats dict: ``{"corpora": {name: {"downloaded_count", "clean"}},
    "mix": <mix stats>, "output_path": str, "report_path": str}``.

    Raises ``RuntimeError`` (with the offending corpus name and stage) if any
    per-corpus stage fails, and ``ValueError`` if ``names`` references an
    unknown corpus, or if ``streaming=True`` is passed without ``sample_limit``.
    """
    if streaming and sample_limit is None:
        raise ValueError(
            "streaming=True requires sample_limit to be set; otherwise an "
            "IterableDataset would be consumed indefinitely."
        )

    config = load_corpus_config(config_path)
    cache_dir = config.get("cache_dir", "data/raw")
    corpora: list[dict[str, Any]] = config.get("corpora", [])
    clean_cfg = config.get("clean", {})
    mix_cfg = config.get("mix", {})

    out_dir = Path(output_dir) if output_dir else _default_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_name = {entry["name"]: entry for entry in corpora}
    if names:
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"Unknown corpus name(s): {', '.join(missing)}")
        selected = [by_name[n] for n in names]
    else:
        selected = corpora

    eval_texts: list[str] | None = None
    if eval_texts_path:
        eval_docs = _read_jsonl(eval_texts_path)
        eval_texts = [d["text"] for d in eval_docs if "text" in d]

    corpus_stats: dict[str, Any] = {}
    docs_by_lang: dict[str, list[dict[str, Any]]] = {}

    total = len(selected)
    for i, entry in enumerate(selected, start=1):
        name = entry["name"]
        logger.info("[%d/%d] Preparing corpus '%s'...", i, total, name)

        try:
            dataset = download_corpus(entry, cache_dir=cache_dir, streaming=streaming)
        except Exception as exc:  # noqa: BLE001 - re-raised with corpus/stage context
            raise RuntimeError(
                f"prepare_data failed for corpus '{name}' at stage 'download': {exc}"
            ) from exc

        try:
            docs = _extract_text_rows(dataset, sample_limit)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"prepare_data failed for corpus '{name}' at stage 'extract': {exc}"
            ) from exc

        try:
            clean_docs, clean_stats = clean_corpus(docs, clean_cfg, eval_texts=eval_texts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"prepare_data failed for corpus '{name}' at stage 'clean': {exc}"
            ) from exc

        corpus_stats[name] = {
            "downloaded_count": len(docs),
            "clean": clean_stats,
        }
        for doc in clean_docs:
            lang = doc.get("language", "other")
            docs_by_lang.setdefault(lang, []).append(doc)

        logger.info(
            "[%d/%d] '%s': %d -> %d documents after cleaning",
            i,
            total,
            name,
            clean_stats["input_count"],
            clean_stats["output_count"],
        )

    try:
        mixed_docs, mix_stats = mix_corpora(
            docs_by_lang,
            ratios=mix_cfg.get("ratios", {"ja": 1.0}),
            seed=mix_cfg.get("seed", 42),
            unit=mix_cfg.get("unit", "documents"),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"prepare_data failed at stage 'mix': {exc}") from exc

    mixture_path = out_dir / "mixture.jsonl"
    _write_jsonl(mixture_path, mixed_docs)

    report_lines = ["# Data preparation report (Issue #20)", ""]
    for name, stats in corpus_stats.items():
        report_lines.append(f"## {name}")
        report_lines.append("")
        report_lines.append(render_stats_report(stats["clean"]))
        report_lines.append("")
    report_lines.append(render_mix_report(mix_stats))
    report = "\n".join(report_lines) + "\n"

    report_path = out_dir / "prepare_report.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Mixture written: %d documents -> %s", len(mixed_docs), mixture_path)
    logger.info("Report written to %s", report_path)

    return {
        "corpora": corpus_stats,
        "mix": mix_stats,
        "output_path": str(mixture_path),
        "report_path": str(report_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="End-to-end data preparation pipeline: download -> clean -> mix (Issue #20)"
    )
    parser.add_argument("--config", required=True, help="Path to configs/data/corpus.yaml")
    parser.add_argument(
        "--names", nargs="+", default=None, help="Subset of corpus names to prepare"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Process only the first N rows of each corpus (pilot/verification runs)",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory for the mixture + report"
    )
    parser.add_argument(
        "--eval-texts", default=None, help="Optional eval-set JSONL for contamination check"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Stream corpora via HF IterableDataset instead of downloading them in full "
            "(requires --sample-limit; large corpora like wikipedia_ja/cc100_ja otherwise "
            "download tens of GB)"
        ),
    )
    args = parser.parse_args()

    result = prepare_data(
        args.config,
        names=args.names,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
        eval_texts_path=args.eval_texts,
        streaming=args.streaming,
    )
    logger.info("Done. Mixture: %s | Report: %s", result["output_path"], result["report_path"])


if __name__ == "__main__":
    main()
