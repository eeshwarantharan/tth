"""Build the held-out chess eval split (never imported by train/)."""

from __future__ import annotations

from tth_gym.envs.chess.tasks import generate, load, save

TRAIN_TASKS = "data/chess/tasks_train.jsonl"
OUT = "evals/chess/tasks.jsonl"

if __name__ == "__main__":
    train_fens = {t.fen for t in load(TRAIN_TASKS)}
    tasks = [t for t in generate(130, seed=90_000, prefix="e") if t.fen not in train_fens][:105]
    tasks += [t for t in generate(18, seed=91_000, mix={"engine_eval": 1}, prefix="eh") if t.fen not in train_fens][:15]
    save(tasks, OUT)
    print(f"wrote {len(tasks)} held-out chess tasks to {OUT}")
