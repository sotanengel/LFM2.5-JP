#!/usr/bin/env bash
# WSL disk cleanup after cpt-D rejection (Issue #145).
# Removes cpt-D weights, eval intermediates, and processed_cptD caches on
# ~/lfm25-ja-k2. Preserves K1/K3 reference assets under ~/lfm25-ja.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

K2_ROOT="${K2_ROOT:-$HOME/lfm25-ja-k2}"
MAIN_ROOT="${MAIN_ROOT:-$HOME/lfm25-ja}"

log_section() { echo ""; echo "=== $1 ==="; }

log_section "Disk before cleanup"
df -h /mnt/c "$HOME" 2>/dev/null || df -h "$HOME"

REMOVE_PATHS=(
  "$K2_ROOT/outputs/cpt-1.2b-layerft-cptD-L9-deci-L9"
  "$K2_ROOT/outputs/eval/jkb/k2-deci"
  "$K2_ROOT/outputs/eval/ifeval_ja/cptD-deci"
  "$K2_ROOT/outputs/eval/cptD_deci_llmjp"
  "$K2_ROOT/outputs/eval/k2_gate"
  "$K2_ROOT/data/processed_cptD"
)

# Glob patterns for centi runs
for d in "$K2_ROOT"/outputs/cpt-1.2b-layerft-cptD-L9-centi* "$K2_ROOT"/outputs/*centi-manual*; do
  if [[ -e "$d" ]]; then
    REMOVE_PATHS+=("$d")
  fi
done

log_section "Removing cpt-D artifacts"
for p in "${REMOVE_PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    echo "rm -rf $p"
    rm -rf "$p"
  else
    echo "skip (missing): $p"
  fi
done

# Optional: dpo-001 intermediates on main root if disk is tight
if [[ "${CLEAN_DPO001_INTERMEDIATES:-0}" == "1" ]]; then
  log_section "Removing dpo-001 intermediates (optional)"
  rm -rf "$MAIN_ROOT/data/processed/dpo"
fi

log_section "Disk after cleanup"
df -h /mnt/c "$HOME" 2>/dev/null || df -h "$HOME"

log_section "Preserved (must remain)"
for p in \
  "$MAIN_ROOT/outputs/eval/jkb/k1-full/base" \
  "$MAIN_ROOT/outputs/eval/ifeval_ja/base-jp202606" \
  "$MAIN_ROOT/outputs/dpo-001-b005" \
  "$MAIN_ROOT/outputs/sft-005-distill" \
  "$MAIN_ROOT/outputs/cpt-1.2b-layerft"; do
  if [[ -e "$p" ]]; then
    echo "OK: $p"
  else
    echo "WARN missing (may be OK): $p"
  fi
done

echo ""
echo "Cleanup complete."
