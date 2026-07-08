#!/usr/bin/env bash
# 350M pilot CPT training wrapper (Issue #23). Override the config via CONFIG=...
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m lfm25_ja.train.train_cpt --config "${CONFIG:-configs/cpt/cpt_350m_pilot.yaml}" "$@"
