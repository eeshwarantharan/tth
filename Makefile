.PHONY: test lint eval report colab demo

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

demo:         ## render demo/tth.gif (+ mp4) from real eval traces; run after `make eval`
	uv run python -m demo.render
