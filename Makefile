.PHONY: test lint eval report colab

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run pyright

eval:          ## run every arm on the held-out suites (needs adapters/ from Colab)
	scripts/run_evals.sh

report:
	uv run python -m evals.report

colab:         ## bundle code + verified data for train/colab_tth.ipynb
	scripts/pack_colab.sh
