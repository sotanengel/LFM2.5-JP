#!/usr/bin/env bash
# End-to-end data preparation pipeline wrapper (Issue #20)
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m lfm25_ja.data.prepare --config configs/data/corpus.yaml "$@"
