.PHONY: setup lint test test-gpu smoke-test probe-memory eval-baseline data train-cpt train-sft train-dpo

PYTHON ?= uv run python
PIP ?= uv pip

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

lint:
	uv run ruff check src tests

test:
	uv run pytest -m "not gpu"

test-gpu:
	uv run pytest -m gpu

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
