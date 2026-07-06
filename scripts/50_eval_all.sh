#!/usr/bin/env bash
# Full evaluation wrapper (WSL / Git Bash)
set -euo pipefail
cd "$(dirname "$0")/.."
make eval-baseline
