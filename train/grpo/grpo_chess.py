"""GRPO in the chess environment (runs on a Colab GPU).

Per round: sample G episodes per task with the current policy (batched), score
each with the verifier reward table, normalise rewards within each group
(advantage = (r - mean) / std), then take one policy-gradient pass where each
trajectory's assistant-token NLL is weighted by its advantage. On-policy with a
single inner step, so the PPO ratio is 1 and no clipping is needed. No KL term.
Rewards come from python-chess / Stockfish verifiers only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tth.agent.loop import run_batched


def _domain(name: str) -> tuple[Any, Any, Any, Any]:
    """(load_tasks, make_env, make_expert, score) for chess or the arm-D table control."""
    if name == "table":
        from train.envs.table import env as tb

        return tb.load, tb.TableEnv, tb.TableExpert, lambda e: tb.reward(e.task, e.st)[:2]
    from train.envs.chess.env import ChessEnv, StockfishExpert, reward
    from train.envs.chess.tasks import load

    return load, ChessEnv, StockfishExpert, lambda e: (lambda rb: (rb.total, rb.correct))(reward(e.task, e.st))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["chess", "table"], default="chess")
    ap.add_argument("--tasks", default=None, help="default: data/<domain>/tasks_train.jsonl")
    ap.add_argument("--adapter", required=True, help="PEFT adapter from chess SFT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tasks-per-round", type=int, default=64)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from train.hf_policy import HFBatchPolicy
    from train.sft.peft_to_mlx import convert
    from train.sft.train_lora import build_model, train

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    load, make_env, make_expert, score = _domain(args.domain)
    tasks = load(args.tasks or f"data/{args.domain}/tasks_train.jsonl")
    model, tok, dtype = build_model(args.base, args.adapter, 16, 32)
    expert = make_expert()
    history = []
    try:
        for rnd in range(args.rounds):
            t0 = time.time()
            batch = rng.sample(tasks, args.tasks_per_round)
            envs = [make_env(t, seed=rng.randrange(1 << 30), expert=expert) for t in batch for _ in range(args.group)]
            model.eval()
            model.config.use_cache = True
            policy = HFBatchPolicy(model, tok, temperature=args.temperature)
            runners = run_batched(envs, policy, budget=args.budget)
            groups: dict[str, list[tuple[float, bool, object]]] = defaultdict(list)
            for env, r in zip(envs, runners, strict=True):
                total, correct = score(env)
                groups[env.task.id].append((total, correct, r))
            rows, all_r, all_c = [], [], []
            for items in groups.values():
                rs = [x[0] for x in items]
                all_r += rs
                all_c += [x[1] for x in items]
                sd = statistics.pstdev(rs)
                if sd < 1e-6:
                    continue
                mu = statistics.mean(rs)
                for rew, _, runner in items:
                    rows.append({"messages": runner.ep.messages, "tools": runner.ep.tools,  # type: ignore[attr-defined]
                                 "weight": (rew - mu) / sd})  # fmt: skip
            sample_s = round(time.time() - t0)
            stats = {"round": rnd, "episodes": len(envs), "mean_reward": round(statistics.mean(all_r), 4),
                     "accuracy": round(sum(all_c) / len(all_c), 4), "train_rows": len(rows), "sample_seconds": sample_s}  # fmt: skip
            print(json.dumps(stats), flush=True)
            if rows:
                model.config.use_cache = False
                tr = train(model, tok, dtype, rows, epochs=1, lr=args.lr, batch_size=2, accum=4, max_len=4096, seed=args.seed + rnd, log_path=None, label=f'{out.name}-round{rnd}')  # fmt: skip
                stats["train_seconds"] = tr["seconds"]
            history.append(stats)
            (out / "grpo_log.json").write_text(json.dumps(history, indent=1))
    finally:
        expert.close()
    model.save_pretrained(out / "peft")
    convert(out / "peft", out / "mlx")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
