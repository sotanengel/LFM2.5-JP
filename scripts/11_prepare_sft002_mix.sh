#!/usr/bin/env bash
# sft-002 mix data preparation wrapper (Issue #105).
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV || true
uv run --no-sync python -m lfm25_ja.data.prepare_sft_mix \
  --config configs/data/mix_002.yaml "$@"
