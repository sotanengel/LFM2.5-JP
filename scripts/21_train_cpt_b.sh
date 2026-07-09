#!/usr/bin/env bash
# Phase 2 cpt-B launcher (Issue #27 / #71 / #72).
# One-liner: bash scripts/21_train_cpt_b.sh full --bg
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
  full | centi) ;;
  *)
    echo "Usage: $0 {full|centi} [--bg] [--rebuild-cache]" >&2
    exit 1
    ;;
esac

SRC_MIXTURE="data/processed_phase2/mixture.jsonl"
if [[ ! -f "$SRC_MIXTURE" ]]; then
  echo "Missing prepared data: $SRC_MIXTURE (run phase2 prepare first)" >&2
  exit 1
fi

mkdir -p data/processed scratchpad
cp -f "$SRC_MIXTURE" data/processed/mixture.jsonl
if [[ -f data/processed_phase2/prepare_report.md ]]; then
  cp -f data/processed_phase2/prepare_report.md data/processed/prepare_report_phase2.md
fi

LOG="scratchpad/phase2_cpt_b_${PACKAGE}.log"
CMD=(
  uv run --no-sync python -m lfm25_ja.train.train_cpt
  --config configs/cpt/cpt_1.2b_layerft.yaml
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
