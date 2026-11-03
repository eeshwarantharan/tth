"""Run one arm (model + optional adapter) on the held-out suites and save raw results.

uv run python -m evals.run --arm A_base
uv run python -m evals.run --arm B_shell --adapter adapters/B_shell/mlx
uv run python -m evals.run --arm frontier --policy nemotron
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from tth.agent.loop import run_episode
from tth.trace.schema import TraceLogger

RESULTS = Path("evals/results")
DEV_MODEL = "models/qwen2.5-coder-1.5b-4bit"


def _policy(args: argparse.Namespace) -> Any:
    if args.policy == "hf":  # GPU: batched greedy decoding over a whole suite (Kaggle/Colab)
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tth_gym.train.hf_policy import HFBatchPolicy

        base = os.environ.get("TTH_BASE", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        tok = AutoTokenizer.from_pretrained(base)
        cuda = torch.cuda.is_available()
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16 if cuda else torch.float32)
        model = model.to("cuda" if cuda else "cpu")  # pyright: ignore[reportCallIssue,reportArgumentType]
        if args.adapter:
            from peft import PeftModel  # pyright: ignore[reportMissingImports]

            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.eval()
        return HFBatchPolicy(model, tok, temperature=0.0, max_batch=8)
    if args.policy == "nemotron":
        from tth.inference.openai_compat import OpenAIPolicy

        return OpenAIPolicy(temperature=0.0)
    from tth.inference.mlx import MLXPolicy

    return MLXPolicy(model_path=os.environ.get("TTH_MODEL", DEV_MODEL), adapter_path=args.adapter, temperature=0.0)


def run_suite(name: str, jobs: list, policy: Any, arm: str, budget: int) -> list[dict]:
    out = RESULTS / f"{arm}__{name}.jsonl"
    out.unlink(missing_ok=True)
    logger = TraceLogger(RESULTS / "traces" / f"{arm}__{name}.jsonl")
    rows = []
    if hasattr(policy, "chat_batch"):
        from tth.agent.loop import run_batched

        t0 = time.time()
        envs = [make_env() for make_env, _ in jobs]

        def progress(done: int, total: int) -> None:
            print(json.dumps({"job": f"eval {arm}/{name}", "step": done, "of": total,
                              "eta_min": None}), flush=True)  # fmt: skip

        runners = run_batched(envs, policy, budget=budget, logger=logger, on_step=progress)
        per = round((time.time() - t0) / max(1, len(envs)), 2)
        with out.open("a") as f:
            for (env, r), (_, summarize) in zip(zip(envs, runners, strict=True), jobs, strict=True):
                s = summarize(env)
                s["seconds"], s["episode"] = per, r.ep.id
                rows.append(s)
                f.write(json.dumps(s) + "\n")
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        print(
            json.dumps({"job": f"eval {arm}/{name}", "step": len(rows), "of": len(rows), "accuracy": round(acc, 3)}),
            flush=True,
        )
        return rows
    for i, (make_env, summarize) in enumerate(jobs):
        env = make_env()
        if hasattr(policy, "reset"):
            policy.reset()
        t0 = time.time()
        ep = run_episode(env, policy, budget=budget, logger=logger)
        s = summarize(env)
        s["seconds"] = round(time.time() - t0, 2)
        s["episode"] = ep.id
        rows.append(s)
        with out.open("a") as f:
            f.write(json.dumps(s) + "\n")
        if (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[{arm}/{name}] {i + 1}/{len(jobs)} correct={acc:.2f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--policy", default="mlx", choices=["mlx", "nemotron", "hf"])
    ap.add_argument("--suites", default="shell,chess,redteam")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=6)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    policy = _policy(args)
    suites = args.suites.split(",")
    lim = args.limit or None

    if "shell" in suites:
        from tth_gym.envs.shell.env import ShellEnv, episode_summary
        from tth_gym.envs.shell.tasks import load as load_shell

        tasks = load_shell("evals/shell/tasks.jsonl")[:lim]
        run_suite("shell", [((lambda t=t, i=i: ShellEnv(t, seed=10_000 + i, p_timeout=0)), episode_summary)
                            for i, t in enumerate(tasks)], policy, args.arm, args.budget)  # fmt: skip
    if "chess" in suites:
        from tth_gym.envs.chess.env import ChessEnv, StockfishExpert
        from tth_gym.envs.chess.env import episode_summary as chess_summary
        from tth_gym.envs.chess.tasks import load

        expert = StockfishExpert()
        tasks_c = load("evals/chess/tasks.jsonl")[:lim]
        try:
            run_suite("chess", [((lambda t=t, i=i: ChessEnv(t, seed=20_000 + i, expert=expert)), chess_summary)
                                for i, t in enumerate(tasks_c)], policy, args.arm, args.budget)  # fmt: skip
        finally:
            expert.close()
    if "redteam" in suites:
        from evals.redteam.env import make_env, summary

        reqs = yaml.safe_load(Path("evals/redteam/requests.yaml").read_text())
        run_suite("redteam", [((lambda r=r, i=i: make_env(r, i)), summary) for i, r in enumerate(reqs)],
                  policy, args.arm, args.budget)  # fmt: skip
    usage = getattr(policy, "usage", None)
    if usage:
        (RESULTS / f"{args.arm}__usage.json").write_text(json.dumps(usage))


if __name__ == "__main__":
    main()
