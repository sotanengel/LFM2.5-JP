#!/usr/bin/env bash
# 8B-A1B-Base 中央層 CPT launcher (Issue #94 / #95 / #97).
# One-liner: bash scripts/22_train_cpt_8b.sh centi --bg
set -euo pipefail
unset VIRTUAL_ENV 2>/dev/null || true
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

SRC_MIXTURE="data/processed_phase2/mixture.jsonl"
if [[ ! -f "$SRC_MIXTURE" ]]; then
  # Fall back to the shared processed mixture used by 1.2B CPT.
  SRC_MIXTURE="data/processed/mixture.jsonl"
fi
if [[ ! -f "$SRC_MIXTURE" ]]; then
  echo "Missing prepared data: data/processed_phase2/mixture.jsonl or data/processed/mixture.jsonl" >&2
  exit 1
fi

mkdir -p data/processed scratchpad
if [[ "$SRC_MIXTURE" != "data/processed/mixture.jsonl" ]]; then
  cp -f "$SRC_MIXTURE" data/processed/mixture.jsonl
fi

LOG="scratchpad/cpt_8b_a1b_${PACKAGE}.log"
CMD=(
  uv run --no-sync python -m lfm25_ja.train.train_cpt
  --config configs/cpt/cpt_8b_a1b_layerft.yaml
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
