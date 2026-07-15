#!/usr/bin/env bash
# ifeval_ja evaluation wrapper (Issue #104).
#
# Standalone transformers generation (GPU) + rule-based scoring (CPU) --
# unlike scripts/50_eval_all.sh this does not touch ~/llm-jp-eval or
# ~/llm-jp-eval-inference. Must run inside WSL2 Ubuntu against the WSL
# checkout of this repo (~/lfm25-ja), not from Windows:
#   wsl -d Ubuntu -- bash -lc '~/lfm25-ja/scripts/60_eval_ifeval_ja.sh'
# On Windows (or without a GPU), pass --dry-run to only print the plan.
set -euo pipefail
cd "$(dirname "$0")/.."

# VIRTUAL_ENV leaking in from a Windows-side shell breaks uv's venv
# resolution inside WSL; `uv run --no-sync` must run against this repo's
# own WSL venv.
unset VIRTUAL_ENV || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --no-sync python -m lfm25_ja.eval.run_ifeval_ja all --config configs/eval/ifeval_ja.yaml "$@"
