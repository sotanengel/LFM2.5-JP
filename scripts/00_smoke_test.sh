#!/usr/bin/env bash
# Phase 0 smoke test wrapper (WSL / Git Bash)
set -euo pipefail
cd "$(dirname "$0")/.."
make smoke-test
