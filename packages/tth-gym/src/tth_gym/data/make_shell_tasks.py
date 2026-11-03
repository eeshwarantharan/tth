"""Generate shell training tasks (fixture seeds disjoint from the eval split)."""

from __future__ import annotations

import sys

from tth_gym.envs.shell.tasks import TRAIN_PHRASES, make_tasks, save

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    out = sys.argv[2] if len(sys.argv) > 2 else "data/shell/tasks_train.jsonl"
    save(make_tasks(n, seed=11, phrases=TRAIN_PHRASES, prefix="t"), out)
    print(f"wrote {n} shell training tasks to {out}")
