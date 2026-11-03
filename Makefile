.PHONY: test lint eval report site

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run lint-imports

eval:          ## run every arm on the held-out suites (needs adapters/ from Colab)
	scripts/run_evals.sh

report:
	uv run python -m evals.report

site:          ## regenerate site/data.js from committed results
	uv run python scripts/build_site_data.py
