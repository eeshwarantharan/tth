"""Generate chess training tasks (seeds disjoint from the eval split)."""

from __future__ import annotations

import sys

from train.envs.chess.tasks import generate, save

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 340
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = "data/chess/tasks_train.jsonl"
    tasks = list(generate(n, seed=1_000, prefix="t"))
    tasks += list(generate(n_eval, seed=2_000, mix={"engine_eval": 1}, prefix="h"))
    save(tasks, out)
    print(f"wrote {len(tasks)} chess training tasks to {out}")
