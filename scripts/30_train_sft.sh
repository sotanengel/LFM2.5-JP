#!/usr/bin/env bash
# SFT training wrapper (Issue #31).
# CONFIG=configs/sft/sft_001.yaml make train-sft
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m lfm25_ja.train.train_sft \
  --config "${CONFIG:-configs/sft/sft_001.yaml}" \
  "$@"
