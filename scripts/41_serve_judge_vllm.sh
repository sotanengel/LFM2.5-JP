#!/usr/bin/env bash
# Native-WSL vLLM server for the dpo-001 judge (Issue #115): official
# Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4 on the OpenAI-compatible API.
#
# Setup (one-time, separate venv -- never touches ~/lfm25-ja/.venv):
#   uv venv ~/qwen-vllm --python 3.12
#   uv pip install --python ~/qwen-vllm/bin/python vllm
#
# Serving knobs ported from the joryu pipeline's RTX 3060 Ti (8GB) tuning:
#   - dtype bfloat16 is REQUIRED (fp16 makes awq_marlin emit token id 0)
#   - kv-cache fp8 roughly doubles effective KV capacity, negligible quality
#   - VLLM_USE_FLASHINFER_SAMPLER=0
# max-num-seqs is raised from joryu's 1 (sequential distillation) to 8: the
# judge issues concurrent short requests (judge.server_concurrency).
#
# The server must not share the GPU with the LFM model (phase exclusivity):
# stop it before Phase G / training runs.
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
exec ~/qwen-vllm/bin/vllm serve tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4 \
  --dtype bfloat16 \
  --quantization awq_marlin \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --kv-cache-dtype fp8 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --swap-space 4 \
  --enforce-eager \
  --port "${PORT:-8100}" \
  "$@"
