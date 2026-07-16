#!/usr/bin/env bash
# DPO training wrapper (Issue #115 / #42).
# CONFIG=configs/dpo/dpo_001_beta01.yaml make train-dpo
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
uv run python -m lfm25_ja.train.train_dpo \
  --config "${CONFIG:-configs/dpo/dpo_001_beta01.yaml}" \
  "$@"
