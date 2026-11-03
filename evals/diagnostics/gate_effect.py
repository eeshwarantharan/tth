"""Does the verification gate help a real model? The first measurement for rung L2 of the improvement ladder.

Same 20 held-out shell tasks and the same verifier as forced_wrapper.py, but each episode runs through the
product's gate (agent/product.py). n=20 and one seed, so this is directional only. Compare with
results/forced_wrapper.jsonl.

TMPDIR=<scratch> uv run python -m evals.diagnostics.gate_effect <out.jsonl>
"""

from __future__ import annotations

import gc
import json
import sys
import time

import mlx.core as mx

import tth.inference.mlx as mb
from evals.diagnostics.forced_wrapper import ARMS, MODEL, ForcedPolicy, forced_parse
from tth.agent.loop import run_episode
from tth.agent.product import ProductEnv
from tth.inference.toolcalls import parse_qwen
from tth_gym.envs.shell.env import ShellEnv
from tth_gym.envs.shell.tasks import load


def run_task(task, seed: int, policy) -> dict:
    fixture = ShellEnv(task, seed=seed, p_timeout=0)
    fixture.reset()
    env = ProductEnv(task.request, fixture.root, "zsh")
    if hasattr(policy, "reset"):
        policy.reset()
    t0 = time.time()
    episode = run_episode(env, policy, budget=6)
    env.close()
    # Score with the same verifier the gym uses, on what the gate actually staged.
    fixture.st.decision = "stage" if env.staged else env.decision
    fixture.st.staged = env.staged_from
    fixture.st.staged_final = env.staged["command"] if env.staged else None
    correct = fixture.check()
    tools = [e.call.tool for e in episode.events if e.call]
    gate_decided = any(e.meta.get("source") == "gate" for e in episode.events)
    row = {
        "task_id": task.id,
        "family": task.family,
        "expected": task.decision,
        "decision": fixture.st.decision,
        "correct": correct,
        "verified": env.verified,
        "gate_decided": gate_decided,
        "staged": env.staged_from,
        "check": fixture.st.check_detail,
        "tools": tools,
        "calls": len(tools),
        "seconds": round(time.time() - t0, 1),
    }
    fixture.close()
    return row


def main() -> None:
    out = sys.argv[1]
    tasks = load("evals/shell/tasks.jsonl")[::5]
    for arm, adapter, forced in ARMS:
        mb.parse_qwen = forced_parse if forced else parse_qwen
        policy = (ForcedPolicy if forced else mb.MLXPolicy)(model_path=MODEL, adapter_path=adapter, temperature=0.0)
        for i, task in enumerate(tasks):
            row = run_task(task, 10_000 + i * 5, policy)
            row.update(arm=arm, forced=forced)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        del policy
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
