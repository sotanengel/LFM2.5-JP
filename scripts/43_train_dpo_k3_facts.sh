#!/usr/bin/env bash
# K3 factual DPO training wrapper (Issue #124 / #145).
# CONFIG=configs/dpo/dpo_k3_facts_beta005.yaml bash scripts/43_train_dpo_k3_facts.sh
set -euo pipefail
unset VIRTUAL_ENV 2>/dev/null || true
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

uv run --no-sync python -m lfm25_ja.train.train_dpo \
  --config "${CONFIG:-configs/dpo/dpo_k3_facts_beta005.yaml}" \
  "$@"
