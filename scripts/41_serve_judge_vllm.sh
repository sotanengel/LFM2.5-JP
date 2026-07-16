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
#   - VLLM_USE_FLASHINFER_SAMPLER=0
#   - joryu also used --kv-cache-dtype fp8, but on vLLM 0.25 fp8 KV forces
#     the FlashInfer attention backend whose JIT needs nvcc (absent on this
#     WSL) -- so KV stays "auto" (bf16, ~5.4k KV tokens: plenty for the
#     judge's short prompt+96-token verdicts)
# max-num-seqs is raised from joryu's 1 (sequential distillation) to 8: the
# judge issues concurrent short requests (judge.server_concurrency).
#
# The server must not share the GPU with the LFM model (phase exclusivity):
# stop it before Phase G / training runs.
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM disables pinned memory on WSL by default (conservative driver check),
# which makes the v1 engine's UVA buffers fail with "UVA is not available".
# Pinned memory demonstrably works on this WSL2 setup (torch pin_memory=True
# succeeds; the transformers/bnb stack relies on it daily), so opt back in.
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
# FlashInfer JIT needs nvcc (absent on this WSL); pin attention to the
# prebuilt flash-attn backend instead.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
exec ~/qwen-vllm/bin/vllm serve tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4 \
  --dtype bfloat16 \
  --quantization awq_marlin \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --enforce-eager \
  --port "${PORT:-8100}" \
  "$@"
