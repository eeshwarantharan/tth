"""Run one arm (model + optional adapter) on the held-out suites and save raw results.

uv run python -m evals.run --arm A_base
uv run python -m evals.run --arm B_shell --adapter adapters/B_shell/mlx
uv run python -m evals.run --arm silent --policy silent
uv run python -m evals.run --arm frontier --policy nemotron
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from evals import manifest
from evals.baselines import BASELINES
from tth.agent.loop import run_episode
from tth.trace.schema import TraceLogger
from tth_gym.core.mcp_env import McpEnvironment

RESULTS = Path("evals/results")
DEFAULT_GGUF = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
SUITE_FILES = {
    "shell": Path("evals/shell/tasks.jsonl"),
    "chess": Path("evals/chess/tasks.jsonl"),
    "redteam": Path("evals/redteam/requests.yaml"),
}
DEV_MODEL = "models/qwen2.5-coder-1.5b-4bit"


def _policy(args: argparse.Namespace) -> Any:
    if args.policy in BASELINES:
        return BASELINES[args.policy]()
    if args.policy == "hf":  # GPU: batched greedy decoding over a whole suite
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
    if args.policy == "llamacpp":
        from tth.core.config import data_dir, state_dir
        from tth.inference.llamacpp import ensure_server, llama_policy

        gguf = Path(args.model) if args.model else data_dir() / "models" / DEFAULT_GGUF
        ensure_server(gguf, state_dir() / "llama-server.log")
        return llama_policy(grammar=args.grammar)
    if args.policy == "nemotron":
        from tth.inference.openai_compat import OpenAIPolicy

        return OpenAIPolicy(temperature=0.0)
    from tth.inference.mlx import MLXPolicy

    return MLXPolicy(
        model_path=args.model or os.environ.get("TTH_MODEL", DEV_MODEL), adapter_path=args.adapter, temperature=0.0
    )


def scrub(path: Path) -> None:
    """Results get committed, so the local username (from `ls -l` output) and home folder stay out of them."""
    if not path.exists():
        return
    text = path.read_text().replace(str(Path.home()), "~")
    user = getpass.getuser()
    if len(user) >= 3:
        text = re.sub(rf"\b{re.escape(user)}\b", "user", text)
    path.write_text(text)


def run_suite(name: str, jobs: list, policy: Any, arm: str, budget: int, results: Path) -> list[dict]:
    out = results / f"{arm}__{name}.jsonl"
    out.unlink(missing_ok=True)
    trace = results / "traces" / f"{arm}__{name}.jsonl"
    trace.unlink(missing_ok=True)
    logger = TraceLogger(trace)
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
    ap.add_argument("--label", default=None, help="how the report names this arm")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=None, help="model path (default: the configured model)")
    ap.add_argument("--policy", default="mlx", choices=["mlx", "llamacpp", "nemotron", "hf", *BASELINES])
    ap.add_argument("--grammar", action="store_true", help="llamacpp: constrain decoding with the tool schemas")
    ap.add_argument("--suites", default="shell,chess,redteam")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--results", default=str(RESULTS), help="where to write results")
    args = ap.parse_args()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "traces").mkdir(exist_ok=True)
    record = manifest.build(
        args.arm, args.label or args.arm, args.policy, args.budget, SUITE_FILES, model=args.model,
        adapter=args.adapter, adapter_sha256=manifest.adapter_hash(args.adapter), grammar=args.grammar,
        limit=args.limit, suites_run=args.suites,
    )  # fmt: skip
    policy = _policy(args)
    suites = args.suites.split(",")
    lim = args.limit or None

    if "shell" in suites:
        from tth_gym.envs.shell.env import ShellEnv, episode_summary
        from tth_gym.envs.shell.tasks import load as load_shell

        tasks = load_shell("evals/shell/tasks.jsonl")[:lim]
        run_suite("shell", [((lambda t=t, i=i: ShellEnv(t, seed=10_000 + i, p_timeout=0)), episode_summary)
                            for i, t in enumerate(tasks)], policy, args.arm, args.budget, results)  # fmt: skip
    if "chess" in suites:
        from tth_gym.envs.chess.env import ChessEnv, StockfishExpert
        from tth_gym.envs.chess.env import episode_summary as chess_summary
        from tth_gym.envs.chess.tasks import load

        expert = StockfishExpert()
        tasks_c = load("evals/chess/tasks.jsonl")[:lim]
        try:
            run_suite("chess", [((lambda t=t, i=i: McpEnvironment(ChessEnv(t, seed=20_000 + i, expert=expert))), chess_summary)
                                for i, t in enumerate(tasks_c)], policy, args.arm, args.budget, results)  # fmt: skip
        finally:
            expert.close()
    if "redteam" in suites:
        from evals.redteam.env import make_env, summary

        reqs = yaml.safe_load(Path("evals/redteam/requests.yaml").read_text())
        run_suite("redteam", [((lambda r=r, i=i: make_env(r, i)), summary) for i, r in enumerate(reqs)],
                  policy, args.arm, args.budget, results)  # fmt: skip
    usage = getattr(policy, "usage", None)
    if usage:
        (results / f"{args.arm}__usage.json").write_text(json.dumps(usage))
    for path in [*results.glob(f"{args.arm}__*.jsonl"), *(results / "traces").glob(f"{args.arm}__*.jsonl")]:
        scrub(path)
    manifest.save(results, record)


if __name__ == "__main__":
    main()
