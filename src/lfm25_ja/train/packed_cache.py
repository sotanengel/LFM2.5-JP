"""Disk cache for tokenized + packed CPT training rows (Issues #71 / #72 / #132)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from lfm25_ja.data.clean import _iter_jsonl

logger = logging.getLogger(__name__)

PACKAGES = ("full", "centi", "deci")
DEFAULT_CACHE_ROOT = Path("data/processed/packed")

# Deterministic subset denominator for each non-"full" package.
# Doc-level subsample before tokenize (Issue #132): centi=1/100, deci=1/10 of
# JSONL documents. Legacy ``apply_package`` still prefixes packed rows the same way.
_PACKAGE_DENOMINATORS: dict[str, int] = {"centi": 100, "deci": 10}


def package_doc_limit(num_docs: int, package: str) -> int | None:
    """Return max documents to tokenize for ``package``, or None for full."""
    if package not in PACKAGES:
        raise ValueError(f"package must be one of {PACKAGES}, got {package!r}")
    if package == "full":
        return None
    if num_docs <= 0:
        return 1
    return max(1, num_docs // _PACKAGE_DENOMINATORS[package])


def count_jsonl_docs(path: str | Path) -> int:
    """Count non-empty JSONL records without materializing them."""
    return sum(1 for _ in _iter_jsonl(path))


def packed_cache_dir(
    source_path: str | Path,
    model_name: str,
    seq_len: int,
    cache_root: str | Path | None = None,
    package: str = "full",
) -> Path:
    """Return the deterministic cache directory for a source + model + seq_len + package."""
    if package not in PACKAGES:
        raise ValueError(f"package must be one of {PACKAGES}, got {package!r}")
    source = Path(source_path)
    root = Path(cache_root or DEFAULT_CACHE_ROOT)
    model_slug = model_name.replace("/", "__")
    return root / f"{source.stem}__{model_slug}__seq{seq_len}__{package}"


def _source_fingerprint(source_path: Path) -> dict[str, float | int]:
    stat = source_path.stat()
    return {"source_mtime": stat.st_mtime, "source_size": stat.st_size}


def cache_is_valid(
    cache_dir: Path,
    source_path: str | Path,
    model_name: str,
    seq_len: int,
    package: str = "full",
) -> bool:
    """True when manifest matches the current source and training settings.

    A manifest that fails to parse as JSON (e.g. a partial write left behind
    by a process that crashed mid-write) is treated as a cache miss rather
    than raising, so a corrupted cache triggers a rebuild instead of crashing
    training.
    """
    manifest_path = cache_dir / "manifest.json"
    packed_path = cache_dir / "packed.pt"
    source = Path(source_path)
    if not manifest_path.is_file() or not packed_path.is_file() or not source.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable/corrupt packed cache manifest: %s", manifest_path)
        return False
    fingerprint = _source_fingerprint(source)
    return (
        manifest.get("source_path") == str(source)
        and manifest.get("model_name") == model_name
        and manifest.get("seq_len") == seq_len
        and manifest.get("package", "full") == package
        and manifest.get("source_mtime") == fingerprint["source_mtime"]
        and manifest.get("source_size") == fingerprint["source_size"]
    )


def compact_packed_rows(
    packed: list[dict[str, list[int]]] | list[torch.Tensor],
) -> list[torch.Tensor]:
    """Store only int32 ``input_ids`` tensors (labels/mask rebuilt in Dataset)."""
    out: list[torch.Tensor] = []
    for row in packed:
        if isinstance(row, torch.Tensor):
            out.append(row.to(dtype=torch.int32).contiguous())
            continue
        ids = row["input_ids"]
        out.append(torch.tensor(ids, dtype=torch.int32))
    return out


def save_packed_cache(
    cache_dir: Path,
    packed: list[dict[str, list[int]]] | list[torch.Tensor],
    source_path: str | Path,
    model_name: str,
    seq_len: int,
    package: str = "full",
) -> None:
    """Persist compact packed rows and a manifest for later cache hits."""
    source = Path(source_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    compact = compact_packed_rows(packed)
    torch.save(compact, cache_dir / "packed.pt")
    manifest = {
        "source_path": str(source),
        "model_name": model_name,
        "seq_len": seq_len,
        "package": package,
        "format": "input_ids_int32_v1",
        "num_sequences": len(compact),
        **_source_fingerprint(source),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved packed cache: %s (%d sequences, package=%s)",
        cache_dir,
        len(compact),
        package,
    )


def load_packed_cache(cache_dir: Path) -> list[torch.Tensor]:
    """Load packed rows previously written by :func:`save_packed_cache`."""
    packed = torch.load(cache_dir / "packed.pt", weights_only=False)
    return compact_packed_rows(packed)


def apply_package(
    packed: list[Any],
    package: str,
) -> list[Any]:
    """Select a training subset from fully packed rows (legacy / post-hoc).

    Prefer building with ``package=`` in :func:`build_or_load_packed` so
    tokenization itself is bounded (Issue #132).
    """
    if package not in PACKAGES:
        raise ValueError(f"package must be one of {PACKAGES}, got {package!r}")
    if package == "full":
        return packed
    n = max(1, len(packed) // _PACKAGE_DENOMINATORS[package])
    return packed[:n]


def build_or_load_packed(
    jsonl_path: str | Path,
    tokenizer: Any,
    seq_len: int,
    model_name: str,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
    package: str = "full",
) -> list[torch.Tensor]:
    """Tokenize + pack once (package-scoped), then reuse ``packed.pt``.

    Non-full packages subsample JSONL documents *before* tokenization so peak
    RAM stays proportional to the package size (Issue #132).
    """
    if package not in PACKAGES:
        raise ValueError(f"package must be one of {PACKAGES}, got {package!r}")

    source = Path(jsonl_path)
    cache_dir = packed_cache_dir(source, model_name, seq_len, cache_root, package=package)

    if not rebuild and cache_is_valid(cache_dir, source, model_name, seq_len, package=package):
        packed = load_packed_cache(cache_dir)
        logger.info(
            "Loaded packed cache: %s (%d sequences, package=%s)",
            cache_dir,
            len(packed),
            package,
        )
        return packed

    max_docs = package_doc_limit(count_jsonl_docs(source), package)
    logger.info(
        "Building packed dataset from %s (seq_len=%d, package=%s, max_docs=%s)",
        source,
        seq_len,
        package,
        max_docs,
    )
    from lfm25_ja.train.train_cpt import build_cpt_dataset

    packed = build_cpt_dataset(source, tokenizer, seq_len, max_docs=max_docs)
    if not packed:
        raise ValueError(
            f"No packed training sequences produced from {source!r} "
            f"(seq_len={seq_len}, package={package})"
        )
    compact = compact_packed_rows(packed)
    save_packed_cache(cache_dir, compact, source, model_name, seq_len, package=package)
    return compact
