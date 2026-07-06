"""YAML configuration loading and merging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into a copy of base."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for experiment reproducibility."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def project_root() -> Path:
    """Return repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def load_project_config(name: str = "base.yaml") -> dict[str, Any]:
    """Load a config from the project's configs/ directory."""
    return load_config(project_root() / "configs" / name)
