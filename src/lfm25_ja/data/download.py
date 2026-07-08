"""HuggingFace dataset acquisition for Phase 1 corpora (Issue #16)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import datasets

from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)


def load_corpus_config(path: str | Path) -> dict[str, Any]:
    """Load the corpus definition YAML (see configs/data/corpus.yaml)."""
    return load_config(path)


def download_corpus(entry: dict[str, Any], cache_dir: str, streaming: bool = False) -> Any:
    """Thin wrapper around ``datasets.load_dataset`` for a single corpus entry.

    ``entry`` follows the ``corpora[]`` schema in configs/data/corpus.yaml:
    ``name``, ``hf_id``, optional ``hf_config``, ``split``.
    """
    name = entry.get("name", "<unknown>")
    hf_id = entry["hf_id"]
    hf_config = entry.get("hf_config")
    split = entry.get("split", "train")
    try:
        if hf_config:
            return datasets.load_dataset(
                hf_id, hf_config, split=split, cache_dir=cache_dir, streaming=streaming
            )
        return datasets.load_dataset(hf_id, split=split, cache_dir=cache_dir, streaming=streaming)
    except Exception as exc:  # noqa: BLE001 - re-raised with corpus context below
        raise RuntimeError(f"Failed to download corpus '{name}' ({hf_id}): {exc}") from exc


def download_all(
    config_path: str | Path,
    names: list[str] | None = None,
    streaming: bool = False,
) -> dict[str, Any]:
    """Download the requested corpora (or all of them if ``names`` is None).

    Returns a mapping of corpus name -> loaded dataset.
    """
    config = load_corpus_config(config_path)
    cache_dir = config.get("cache_dir", "data/raw")
    corpora: list[dict[str, Any]] = config.get("corpora", [])
    by_name = {entry["name"]: entry for entry in corpora}

    if names:
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"Unknown corpus name(s): {', '.join(missing)}")
        selected = [by_name[n] for n in names]
    else:
        selected = corpora

    results: dict[str, Any] = {}
    total = len(selected)
    for i, entry in enumerate(selected, start=1):
        name = entry["name"]
        logger.info("[%d/%d] Downloading corpus '%s' (%s)...", i, total, name, entry["hf_id"])
        results[name] = download_corpus(entry, cache_dir=cache_dir, streaming=streaming)
        logger.info("[%d/%d] Finished downloading '%s'", i, total, name)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download Phase 1 HF corpora")
    parser.add_argument("--config", required=True, help="Path to configs/data/corpus.yaml")
    parser.add_argument(
        "--names", nargs="+", default=None, help="Subset of corpus names to download"
    )
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    results = download_all(args.config, names=args.names, streaming=args.streaming)
    logger.info("Downloaded %d corpora: %s", len(results), ", ".join(results.keys()))


if __name__ == "__main__":
    main()
