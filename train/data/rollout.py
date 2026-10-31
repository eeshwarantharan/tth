"""Run a policy over tasks in an environment and record episodes + rewards.

Used for teacher trajectory generation (Nemotron over the NVIDIA API) and for
local baselines. Output rows are self-contained SFT examples:
{"messages", "tools", "summary", "meta"}.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tth.agent.loop import run_episode
from tth.trace.schema import TraceLogger

CLOSERS: list[Callable[[], None]] = []


def rollout_one(make_env: Callable[[], Any], policy: Any, summarize: Callable[[Any], dict],
                budget: int, logger: TraceLogger | None) -> dict:  # fmt: skip
    env = make_env()
    if hasattr(policy, "reset"):
        policy.reset()
    ep = run_episode(env, policy, budget=budget, logger=logger)
    summary = summarize(env)
    if logger:
        from tth.trace.schema import TraceEvent

        logger.write(TraceEvent(episode=ep.id, domain=ep.domain, step=len(ep.events), decision=ep.decision,
                                reward=summary["reward"], result={"summary": summary}))  # fmt: skip
    return {
        "id": ep.id,
        "messages": ep.messages,
        "tools": ep.tools,
        "summary": summary,
        "meta": {**ep.meta, "teacher": getattr(policy, "model", "local")},
    }


def run_many(jobs: list[tuple[Callable[[], Any], Callable[[Any], dict]]], policy_factory: Callable[[], Any],
             out: Path, budget: int, workers: int, trace: Path | None) -> list[dict]:  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    logger = TraceLogger(trace) if trace else None
    lock = threading.Lock()
    rows: list[dict] = []
    local = threading.local()

    def get_policy() -> Any:
        if not hasattr(local, "p"):
            local.p = policy_factory()
        return local.p

    def work(job: tuple) -> dict:
        make_env, summarize = job
        return rollout_one(make_env, get_policy(), summarize, budget, logger)

    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - keep the batch going, report at the end
                print(f"[rollout] error: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                ok = sum(r["summary"]["correct"] for r in rows)
                print(
                    f"[rollout] {i}/{len(jobs)} correct={ok} ({time.time() - t0:.0f}s)",
                    file=sys.stderr,
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["chess", "shell"], required=True)
    ap.add_argument("--tasks", required=True, help="tasks JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="nemotron", help="nemotron | mlx | mlx:<adapter_path>")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--resume", action="store_true", help="skip task ids already in --out")
    ap.add_argument("--teacher-model", default=None, help="NVIDIA API model id (default: Nemotron 3.5 Super)")
    args = ap.parse_args()

    if args.policy == "nemotron":
        from tth.runtime.openai_backend import OpenAIPolicy

        temp = 0.6 if args.temperature is None else args.temperature

        from tth.runtime.openai_backend import NVIDIA_MODEL, RateLimiter

        model = args.teacher_model or NVIDIA_MODEL
        limiter = RateLimiter(int(__import__("os").environ.get("TTH_NVIDIA_RPM", "36")))

        def policy_factory() -> Any:
            return OpenAIPolicy(model=model, temperature=temp, limiter=limiter)
    else:
        from tth.runtime.mlx_backend import MLXPolicy

        adapter = args.policy.split(":", 1)[1] if ":" in args.policy else None
        temp = 0.0 if args.temperature is None else args.temperature
        args.workers = 1  # one model in memory on an 8 GB machine
        shared = MLXPolicy(adapter_path=adapter, temperature=temp)

        def policy_factory() -> Any:
            return shared

    done: set[str] = set()
    if args.resume and Path(args.out).exists():
        done = {json.loads(x)["summary"]["task_id"] for x in Path(args.out).read_text().splitlines() if x.strip()}
        print(f"[rollout] resuming: {len(done)} tasks already done", file=sys.stderr)

    jobs: list[tuple[Callable[[], Any], Callable[[Any], dict]]] = []
    if args.domain == "chess":
        from train.envs.chess.env import ChessEnv, StockfishExpert, episode_summary
        from train.envs.chess.tasks import load

        tasks = load(args.tasks)
        expert_local = threading.local()

        def expert() -> StockfishExpert:
            if not hasattr(expert_local, "e"):
                expert_local.e = StockfishExpert()
                CLOSERS.append(expert_local.e.close)
            return expert_local.e

        for i, t in enumerate(tasks[: args.limit or None]):
            if t.id in done:
                continue
            jobs.append(((lambda t=t, i=i: ChessEnv(t, seed=args.seed * 100003 + i, expert=expert())),
                         episode_summary))  # fmt: skip
    else:
        from train.envs.shell.env import ShellEnv
        from train.envs.shell.env import episode_summary as shell_summary
        from train.envs.shell.tasks import load as load_shell

        tasks_s = load_shell(args.tasks)
        for i, t in enumerate(tasks_s[: args.limit or None]):
            if t.id in done:
                continue
            jobs.append(((lambda t=t, i=i: ShellEnv(t, seed=args.seed * 100003 + i)), shell_summary))

    try:
        rows = run_many(jobs, policy_factory, Path(args.out), args.budget, args.workers,
                        Path(args.trace) if args.trace else None)  # fmt: skip
    finally:
        for closer in CLOSERS:
            closer()
    n = len(rows)
    if n:
        acc = sum(r["summary"]["correct"] for r in rows) / n
        rew = sum(r["summary"]["reward"] for r in rows) / n
        print(json.dumps({"n": n, "accuracy": round(acc, 4), "mean_reward": round(rew, 4)}))


if __name__ == "__main__":
    main()
