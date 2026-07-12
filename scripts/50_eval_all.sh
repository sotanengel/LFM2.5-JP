#!/usr/bin/env bash
# Full evaluation wrapper (Issue #66).
#
# The real llm-jp-eval v2 pipeline (preprocess -> dump -> llm-jp-eval-inference
# -> evaluate) needs the GPU and the ~/llm-jp-eval / ~/llm-jp-eval-inference
# checkouts, so this MUST run inside WSL2 Ubuntu against the WSL checkout of
# this repo (~/lfm25-ja), not from Windows. Run it from a WSL bash shell:
#   wsl -d Ubuntu -- bash -lc '~/lfm25-ja/scripts/50_eval_all.sh'
# On Windows (or when ~/llm-jp-eval isn't present), lfm25_ja.eval.run_llm_jp_eval
# falls back to a dry run that only prints the pipeline commands.
set -euo pipefail
cd "$(dirname "$0")/.."

# VIRTUAL_ENV leaking in from a Windows-side shell breaks uv's venv
# resolution inside WSL; `uv run --no-sync` (used by `make eval-baseline`)
# must run against this repo's own WSL venv.
unset VIRTUAL_ENV || true

make eval-baseline
