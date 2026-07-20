#!/usr/bin/env bash
# K3 factual DPO pref pipeline (Issue #124 / #145).
# Runs P0 -> G -> V -> J -> P sequentially. Start judge vLLM before Phase J.
set -euo pipefail
unset VIRTUAL_ENV 2>/dev/null || true
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/data/dpo_pairs_k3_facts.yaml}"

echo "=== K3 Phase P0: pref_prompts_jkb ==="
uv run --no-sync python -m lfm25_ja.data.pref_prompts_jkb --config "$CONFIG"

echo "=== K3 Phase G: pref_generate ==="
uv run --no-sync python -m lfm25_ja.data.pref_generate --config "$CONFIG"

echo "=== K3 Phase V: pref_verify_facts ==="
uv run --no-sync python -m lfm25_ja.data.pref_verify_facts --config "$CONFIG"

echo "=== K3 Phase J: judge_swallow (factual) ==="
echo "Ensure scripts/41_serve_judge_vllm.sh is running before this step."
uv run --no-sync python -m lfm25_ja.eval.judge_swallow --config "$CONFIG"

echo "=== K3 Phase P: pref_pairs_facts ==="
uv run --no-sync python -m lfm25_ja.data.pref_pairs_facts --config "$CONFIG"

echo "K3 pref pipeline complete."
