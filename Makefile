.PHONY: setup setup-gpu lint test test-gpu smoke-test probe-memory eval-baseline data train-cpt train-sft train-dpo

# GPU targets use --no-sync to avoid uv reverting to CPU torch from lock on partial sync
PYTHON ?= uv run --no-sync python
PYTHON_CPU ?= uv run python
PIP ?= uv pip

setup:
	uv venv --python 3.11
	uv sync --extra dev

setup-gpu:
	powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1

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
	bash scripts/10_prepare_data.sh

train-cpt:
	bash scripts/20_train_cpt.sh

train-sft:
	bash scripts/30_train_sft.sh

train-dpo:
	bash scripts/40_train_dpo.sh
