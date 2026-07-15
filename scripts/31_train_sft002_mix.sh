#!/usr/bin/env bash
# sft-002 mix training wrapper (Issue #105). See docs/sft_training_runbook.md.
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run --no-sync python -m lfm25_ja.train.train_sft \
  --config configs/sft/sft_002_mix.yaml \
  --no-checkpoints "$@"
