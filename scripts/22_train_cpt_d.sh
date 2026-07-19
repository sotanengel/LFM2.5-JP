#!/usr/bin/env bash
# K2 cpt-D launcher (Issue #132 / #136).
# One-liner: bash scripts/22_train_cpt_d.sh deci
# Always unsets PYTORCH_CUDA_ALLOC_CONF (WSL2 dxgkrnl + expandable_segments
# causes CUDA "device not ready"; see experiments/reports/k2_cptD.md).
set -euo pipefail
unset VIRTUAL_ENV 2>/dev/null || true
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

PACKAGE="${1:-}"
shift || true

BG=0
REBUILD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bg)
      BG=1
      shift
      ;;
    --rebuild-cache)
      REBUILD+=(--rebuild-cache)
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "$PACKAGE" in
  full | centi | deci) ;;
  *)
    echo "Usage: $0 {full|centi|deci} [--bg] [--rebuild-cache]" >&2
    exit 1
    ;;
esac

SRC_MIXTURE="data/processed_cptD/mixture.jsonl"
if [[ ! -f "$SRC_MIXTURE" ]]; then
  echo "Missing prepared data: $SRC_MIXTURE (run cpt-D prepare / scratchpad/build_cptD_corpus.py first)" >&2
  exit 1
fi

mkdir -p scratchpad

LOG="scratchpad/k2_cpt_d_${PACKAGE}.log"
CMD=(
  uv run --no-sync python -m lfm25_ja.train.train_cpt
  --config configs/cpt/cpt_1.2b_layerft_cptD_L9.yaml
  --package "$PACKAGE"
)
if ((${#REBUILD[@]})); then
  CMD+=("${REBUILD[@]}")
fi

if [[ "$BG" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$LOG" 2>&1 &
  echo "PID=$!"
  echo "LOG=$LOG"
else
  exec "${CMD[@]}"
fi
