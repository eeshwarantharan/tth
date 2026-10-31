#!/bin/bash
# Bundle exactly what Colab needs for training. Never includes .env, models or traces.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m train.data.filter data/chess/teacher_raw.jsonl data/chess/teacher_verified.jsonl
uv run python -m train.data.filter data/shell/teacher_raw.jsonl data/shell/teacher_verified.jsonl
uv run python -m train.data.filter data/chess/scripted_raw.jsonl data/chess/scripted_verified.jsonl
rm -f tth_colab.zip
zip -qr tth_colab.zip src train experts evals demo scripts pyproject.toml \
  data/chess/tasks_train.jsonl data/chess/teacher_verified.jsonl data/chess/scripted_verified.jsonl \
  data/shell/teacher_verified.jsonl data/table/tasks_train.jsonl data/table/scripted_verified.jsonl \
  -x '*/__pycache__/*' '*.pyc' 'evals/results/*' 'demo/tth.*'
if unzip -l tth_colab.zip | grep -qE '\.env|models/|\.venv'; then echo "refusing: secret or large file in bundle"; rm tth_colab.zip; exit 1; fi
ls -lh tth_colab.zip
