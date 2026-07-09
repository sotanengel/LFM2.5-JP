.PHONY: setup setup-gpu setup-gpu-linux lint test test-gpu smoke-test probe-memory eval-baseline data train-cpt train-sft train-dpo

# GPU targets use --no-sync to avoid uv reverting to CPU torch from lock on partial sync
PYTHON ?= uv run --no-sync python
PYTHON_CPU ?= uv run python
PIP ?= uv pip

setup:
	uv venv --python 3.11
	uv sync --extra dev

# Windows-only: installs CUDA torch via PyTorch's cu126 index (scripts/setup_gpu.ps1).
setup-gpu:
	powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1

# Linux / WSL2: no PowerShell script needed. The gpu extra resolves to a PyPI
# torch wheel (CUDA + flash SDP bundled), so a plain `uv sync` is enough
# (Issue #64).
setup-gpu-linux:
	uv sync --extra dev --extra gpu

lint:
	$(PYTHON_CPU) -m ruff check src tests

test:
	$(PYTHON_CPU) -m pytest -m "not gpu"

test-gpu:
	$(PYTHON) -m pytest -m gpu

smoke-test:
	$(PYTHON) -m lfm25_ja.train.smoke

probe-memory:
	$(PYTHON) -m lfm25_ja.utils.memory_probe

eval-baseline:
	$(PYTHON) -m lfm25_ja.eval.run_llm_jp_eval

data:
	@test -f scripts/10_prepare_data.sh || (echo "scripts/10_prepare_data.sh not implemented yet (see GitHub issues)" && exit 1)
	bash scripts/10_prepare_data.sh

train-cpt:
	@test -f scripts/20_train_cpt.sh || (echo "scripts/20_train_cpt.sh not implemented yet (see GitHub issues)" && exit 1)
	bash scripts/20_train_cpt.sh

train-sft:
	@test -f scripts/30_train_sft.sh || (echo "scripts/30_train_sft.sh not implemented yet (see GitHub issues)" && exit 1)
	bash scripts/30_train_sft.sh

train-dpo:
	@test -f scripts/40_train_dpo.sh || (echo "scripts/40_train_dpo.sh not implemented yet (see GitHub issues)" && exit 1)
	bash scripts/40_train_dpo.sh
