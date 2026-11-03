#!/bin/bash
# Evaluate every arm that has an adapter, then build the results table.
set -uo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
run() {  # arm adapter
  if [ -n "$2" ] && [ ! -d "$2" ]; then echo "skip $1 (no adapter at $2)"; return; fi
  if [ -n "$2" ]; then uv run python -m evals.run --arm "$1" --adapter "$2"; else uv run python -m evals.run --arm "$1"; fi
}
run A_base ""
run B_shell adapters/B_shell/mlx
run C_chess_shell adapters/C_chess_shell/mlx
run C0_chess adapters/C0_chess_grpo/mlx
run D_table_shell adapters/D_table_shell/mlx
uv run python -m evals.report
