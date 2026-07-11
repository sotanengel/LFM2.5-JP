"""Disk cache for tokenized + packed CPT training rows (Issues #71 / #72)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

PACKAGES = ("full", "centi", "deci")
DEFAULT_CACHE_ROOT = Path("data/processed/packed")

# Deterministic subset denominator for each non-"full" package: the first
# len(packed) // N rows are kept (at least one, for smoke runs).
_PACKAGE_DENOMINATORS: dict[str, int] = {"centi": 100, "deci": 10}


def packed_cache_dir(
    source_path: str | Path,
    model_name: str,
    seq_len: int,
    cache_root: str | Path | None = None,
) -> Path:
    """Return the deterministic cache directory for a source + model + seq_len."""
    source = Path(source_path)
    root = Path(cache_root or DEFAULT_CACHE_ROOT)
    model_slug = model_name.replace("/", "__")
    return root / f"{source.stem}__{model_slug}__seq{seq_len}"


def _source_fingerprint(source_path: Path) -> dict[str, float | int]:
    stat = source_path.stat()
    return {"source_mtime": stat.st_mtime, "source_size": stat.st_size}


def cache_is_valid(
    cache_dir: Path,
    source_path: str | Path,
    model_name: str,
    seq_len: int,
) -> bool:
    """True when manifest matches the current source and training settings."""
    manifest_path = cache_dir / "manifest.json"
    packed_path = cache_dir / "packed.pt"
    source = Path(source_path)
    if not manifest_path.is_file() or not packed_path.is_file() or not source.is_file():
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = _source_fingerprint(source)
    return (
        manifest.get("source_path") == str(source)
        and manifest.get("model_name") == model_name
        and manifest.get("seq_len") == seq_len
        and manifest.get("source_mtime") == fingerprint["source_mtime"]
        and manifest.get("source_size") == fingerprint["source_size"]
    )


def save_packed_cache(
    cache_dir: Path,
    packed: list[dict[str, list[int]]],
    source_path: str | Path,
    model_name: str,
    seq_len: int,
) -> None:
    """Persist packed rows and a manifest for later cache hits."""
    source = Path(source_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(packed, cache_dir / "packed.pt")
    manifest = {
        "source_path": str(source),
        "model_name": model_name,
        "seq_len": seq_len,
        "num_sequences": len(packed),
        **_source_fingerprint(source),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved packed cache: %s (%d sequences)",
        cache_dir,
        len(packed),
    )


def load_packed_cache(cache_dir: Path) -> list[dict[str, list[int]]]:
    """Load packed rows previously written by :func:`save_packed_cache`."""
    return torch.load(cache_dir / "packed.pt", weights_only=False)


def apply_package(
    packed: list[dict[str, list[int]]],
    package: str,
) -> list[dict[str, list[int]]]:
    """Select a training subset from fully packed rows."""
    if package not in PACKAGES:
        raise ValueError(f"package must be one of {PACKAGES}, got {package!r}")
    if package == "full":
        return packed
    # centi = 1/100, deci = 1/10 of packed sequences (at least one row for smoke runs)
    n = max(1, len(packed) // _PACKAGE_DENOMINATORS[package])
    return packed[:n]


def build_or_load_packed(
    jsonl_path: str | Path,
    tokenizer: Any,
    seq_len: int,
    model_name: str,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
) -> list[dict[str, list[int]]]:
    """Tokenize + pack once, then reuse ``packed.pt`` on subsequent runs."""
    source = Path(jsonl_path)
    cache_dir = packed_cache_dir(source, model_name, seq_len, cache_root)

    if not rebuild and cache_is_valid(cache_dir, source, model_name, seq_len):
        packed = load_packed_cache(cache_dir)
        logger.info(
            "Loaded packed cache: %s (%d sequences)",
            cache_dir,
            len(packed),
        )
        return packed

    logger.info("Building packed dataset from %s (seq_len=%d)", source, seq_len)
    from lfm25_ja.train.train_cpt import build_cpt_dataset

    packed = build_cpt_dataset(source, tokenizer, seq_len)
    if not packed:
        raise ValueError(
            f"No packed training sequences produced from {source!r} (seq_len={seq_len})"
        )
    save_packed_cache(cache_dir, packed, source, model_name, seq_len)
    return packed
