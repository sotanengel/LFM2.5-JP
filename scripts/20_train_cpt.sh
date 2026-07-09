#!/usr/bin/env bash
# CPT training wrapper (Issue #23 / #30 / #71 / #72).
# CONFIG=configs/cpt/cpt_1.2b_layerft.yaml PACKAGE=centi make train-cpt
set -euo pipefail
cd "$(dirname "$0")/.."
PACKAGE="${PACKAGE:-full}"
case "$PACKAGE" in
  full|centi) ;;
  *)
    echo "PACKAGE must be 'full' or 'centi', got: $PACKAGE" >&2
    exit 1
    ;;
esac
uv run python -m lfm25_ja.train.train_cpt \
  --config "${CONFIG:-configs/cpt/cpt_350m_pilot.yaml}" \
  --package "$PACKAGE" \
  "$@"

