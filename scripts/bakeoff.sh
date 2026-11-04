#!/bin/bash
# Model bake-off: stock models, through the real gate, on the held-out shell and red-team suites. One server at a time.
# Usage: scripts/bakeoff.sh name=/path/to/model.gguf ...
set -uo pipefail
cd "$(dirname "$0")/.."
for spec in "$@"; do
  name="${spec%%=*}"
  model="${spec#*=}"
  pkill -f "llama-server --model" ; sleep 2
  echo "== $name $(date +%H:%M:%S)"
  "${PY:-.venv/bin/python}" -m evals.run --arm "bake_$name" --label "Stock $name" --policy llamacpp --model "$model" --suites shell,redteam 2>&1 | grep -E "Traceback|Error|error:" | head -3
  echo "   finished $(date +%H:%M:%S)"
done
pkill -f "llama-server --model"
"${PY:-.venv/bin/python}" -m evals.report > /dev/null
