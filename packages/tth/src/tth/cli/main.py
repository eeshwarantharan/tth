"""tth command line.

  tth "<request>"              inspect, verify, then stage a command
  tth why [--raw]              explain the last result, from its trace
  tth replay FILE [--raw]      show a recorded episode without a model
  tth init zsh|bash            print shell integration (Ctrl-G and `t`)
  tth setup                    check the sandbox, model and backend
  tth expert                   run the reference A2A expert agent
  tth mcp                      serve the read-only probes over MCP

tth never executes the final command: `run` writes it to --out (the shell
integration puts it on your prompt) or prints it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from tth.a2a.client import ExpertClient
from tth.agent.escalation import ConsentRequest, escalate
from tth.core.config import Config, ConfigError, find_model, load_config, load_user_env, model_candidates, state_dir
from tth.core.types import RequestTooLong
from tth.trace.schema import Call, TraceEvent, TraceLogger, read_trace
from tth.ui.terminal import Outcome, Style, TerminalRenderer, err, printable

if TYPE_CHECKING:
    from tth.agent.product import ProductEnv

ui = TerminalRenderer()


def _read_tty(prompt: str) -> str:
    try:
        with open("/dev/tty") as tty:
            err(prompt)
            return tty.readline().strip()
    except OSError:
        return ""


def _start_expert() -> bool:
    """Start the reference expert locally, if it can. It stops by itself after 10 idle minutes."""
    import subprocess
    from importlib.util import find_spec

    if find_spec("uvicorn") is None:
        return False
    log = state_dir() / "expert.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "tth.expert.server", "--idle-exit", "600"], stdout=log.open("a"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )  # fmt: skip
    client = ExpertClient()
    for _ in range(40):
        time.sleep(0.25)
        if client.card() is not None:
            return True
    return False


def finish(outcome: Outcome, args: argparse.Namespace) -> int:
    """Report how the request ended. The command is written out for the shell, and never run here."""
    ui.outcome(outcome)
    if outcome.decision == "stage" and outcome.command:
        if args.out:
            Path(args.out).write_text(outcome.command)
        elif not args.json:
            print(outcome.command)
    if args.json:
        print(json.dumps({**asdict(outcome), "badge": outcome.badge if outcome.decision == "stage" else None}))
    return 0 if outcome.decision == "stage" else 1


class ConsentPrompt:
    """Every consent question goes through here, and the only yes comes from the user's own terminal."""

    def confirm(self, request: ConsentRequest) -> bool:
        ui.consent(
            request.destination, request.text, request.redactions, request.reason, request.is_local, request.follow_up
        )
        return _read_tty("  Send it? [y/N] ").lower() in ("y", "yes")

    def status(self, text: str) -> None:
        ui.note(f"expert: {text}")


def escalate_flow(env: ProductEnv, messages: list) -> Outcome:
    result = escalate(env, messages, ConsentPrompt(), ExpertClient(), start_expert=_start_expert)
    if result.decision != "stage":
        return Outcome(result.decision, message=printable(result.message))
    if result.explanation:
        ui.note(f"expert: {result.explanation}")
    if result.test:
        ui.event(TraceEvent(episode="expert", domain="shell", step=0, call=Call(tool="dry_run", args={"command": result.command}),
                            result=result.test, ok=bool(result.test.get("ok"))))  # fmt: skip
    return Outcome("stage", result.command, result.tier, result.notes, origin="expert", verified=result.verified)


def _llama_policy(config: Config) -> object | None:
    from dataclasses import replace

    try:
        from tth.inference.llamacpp import ServerUnavailable, ensure_server, llama_policy
    except ImportError:
        ui.problem("The llama.cpp backend isn't installed.", "Run: pip install tth")
        return None
    with_backend = replace(config, backend="llamacpp")
    model = find_model(with_backend)
    if model is None:
        looked = ", ".join(str(p) for p in model_candidates(with_backend))
        ui.problem("No GGUF model found.", f"Looked in: {looked}. Run `tth setup` for how to get one.")
        return None
    try:
        ensure_server(model, state_dir() / "llama-server.log")
    except ServerUnavailable as e:
        ui.problem(str(e))
        return None
    return llama_policy()


def _make_policy(args: argparse.Namespace, config: Config) -> object | None:
    """The configured model, or None after explaining what's missing."""
    if args.backend == "llamacpp":
        return _llama_policy(config)
    if args.backend == "mlx":
        model = find_model(config)
        if model is None:
            looked = ", ".join(str(p) for p in model_candidates(config))
            ui.problem("No model found.", f"Looked in: {looked}. Run `tth setup` for how to get one.")
            return None
        try:
            from tth.inference.mlx import MLXPolicy

            return MLXPolicy(model_path=str(model), adapter_path=args.adapter or config.adapter)
        except ImportError:
            ui.problem("The MLX backend isn't installed.", "On Apple Silicon: pip install 'tth[mlx]'")
            return None
    try:
        from tth.inference.openai_compat import OpenAIPolicy
    except ImportError:
        ui.problem("The OpenAI-compatible backend isn't installed.", "Run: pip install tth")
        return None
    return OpenAIPolicy(model=args.model, base_url=args.base_url or config.base_url, api_key="local", thinking=False,
                        temperature=0.0, max_tokens=512)  # fmt: skip


def cmd_run(args: argparse.Namespace) -> int:
    from tth.agent.loop import run_episode
    from tth.agent.product import ProductEnv

    # The terminal shows tth's own lines. Library logging would bury them, and the trace has the detail.
    logging.disable(logging.CRITICAL)
    load_user_env()
    config = load_config()
    request = " ".join(args.request).strip()
    if not request:
        err('usage: tth "<request>"')
        return 2
    cwd = Path.cwd()
    shell = Path(os.environ.get("SHELL", "/bin/zsh")).name
    args.backend = args.backend or config.backend
    ui.request(request)
    policy = _make_policy(args, config)
    if policy is None:
        return 1
    env = ProductEnv(request, cwd, shell)
    try:
        logger = TraceLogger(state_dir() / "traces.jsonl")
        budget = args.budget or config.budget
        ep = run_episode(env, policy, budget=budget, logger=logger, on_event=ui.event)  # type: ignore[arg-type]
        if env.decision == "stage" and env.staged:
            staged = env.staged
            outcome = Outcome(
                "stage", staged["command"], staged["tier"], staged.get("notes", []), verified=env.verified
            )
            return finish(outcome, args)
        if env.decision == "ask":
            return finish(Outcome("ask", message=f"{ui.s.bold}tth asks:{ui.s.reset} {env.reason}"), args)
        return finish(escalate_flow(env, ep.messages), args)
    finally:
        env.close()


def show_episode(events: list[TraceEvent], raw: bool) -> None:
    """Draw a recorded episode with the same code that drew it live."""
    style = Style()
    start = next((e for e in events if e.meta.get("kind") == "start"), None)
    if start:
        err(f"{style.dim}tth ▸ {start.meta.get('request', '')}{style.reset}")
    from tth.ui.terminal import describe

    for ev in events:
        line = describe(ev, style)
        if line:
            err(line)
        if raw and ev.raw is not None:
            err(f"    {style.dim}model wrote: {ev.raw!r}{style.reset}")
        if ev.decision:
            err(f"  {style.dim}ended: {ev.decision}{style.reset}")


def _last_episode(events: list[TraceEvent]) -> list[TraceEvent]:
    return [e for e in events if e.episode == events[-1].episode] if events else []


def cmd_why(args: argparse.Namespace) -> int:
    path = state_dir() / "traces.jsonl"
    events = read_trace(path) if path.exists() else []
    if not events:
        ui.problem("No runs yet.", 'Try: tth "list python files"')
        return 1
    show_episode(_last_episode(events), args.raw)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        ui.problem(f"{path} doesn't exist.")
        return 1
    events = read_trace(path)
    if args.episode:
        events = [e for e in events if e.episode == args.episode]
    else:
        events = _last_episode(events)
    if not events:
        ui.problem("No matching episode in that file.")
        return 1
    show_episode(events, args.raw)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from tth.shell.integration import BASH, ZSH

    print(ZSH if args.shell == "zsh" else BASH)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from tth.sandbox.runner import SandboxUnavailable, run

    config = load_config()
    ok = True
    try:
        r = run(["true"], Path.cwd())
        err(f"sandbox: {'ok' if r.ok else 'FAILED'}")
        ok &= r.ok
    except SandboxUnavailable as e:
        err(f"sandbox: unavailable ({e})")
        ok = False
    model = find_model(config)
    if model:
        err(f"model: ok ({model})")
    else:
        ok = False
        target = model_candidates(config)[0]
        err(f"model: missing. Looked in: {', '.join(str(p) for p in model_candidates(config))}")
        err("  Download it with:")
        err("    pip install huggingface_hub")
        if config.backend == "llamacpp":
            err(
                f"    hf download bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF {target.name} --local-dir '{target.parent}'"
            )
        else:
            err(f"    hf download mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --local-dir '{target}'")
    if config.backend == "mlx":
        try:
            import mlx_lm  # noqa: F401

            err("backend: mlx ok")
        except ImportError:
            ok = False
            err("backend: mlx isn't installed. On Apple Silicon: pip install 'tth[mlx]'")
    elif config.backend == "llamacpp":
        import shutil

        if shutil.which("llama-server"):
            err("backend: llama.cpp ok")
        else:
            ok = False
            err("backend: llama-server isn't installed. On macOS: brew install llama.cpp")
    err(f'shell integration: add to your rc file ->  eval "$(tth init {Path(os.environ.get("SHELL", "zsh")).name})"')
    return 0 if ok else 1


def cmd_expert(args: argparse.Namespace) -> int:
    load_user_env()
    try:
        from tth.expert.server import main as expert_main
    except ImportError:
        ui.problem("The expert server isn't installed.", "Run: pip install 'tth[expert]'")
        return 1
    expert_main(args.rest)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from tth.mcp.server import main as mcp_main

    mcp_main(args.rest)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tth", description="tiny terminal helper: looks before it acts.")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="turn a request into a staged command")
    r.add_argument("request", nargs="+")
    r.add_argument("--out", help="write the staged command here (used by the shell widget)")
    r.add_argument("--adapter", help="LoRA adapter directory (MLX)")
    r.add_argument("--backend", choices=["mlx", "llamacpp", "openai"], default=None)
    r.add_argument("--base-url", default=None,
                   help="llama-server URL for --backend openai")  # fmt: skip
    r.add_argument("--model", default="local")
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--json", action="store_true", help="print the outcome as JSON")
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("init", help="print shell integration")
    i.add_argument("shell", choices=["zsh", "bash"])
    i.set_defaults(fn=cmd_init)
    sub.add_parser("setup", help="check the install").set_defaults(fn=cmd_setup)
    e = sub.add_parser("expert", help="run the reference A2A expert")
    e.add_argument("rest", nargs=argparse.REMAINDER)
    e.set_defaults(fn=cmd_expert)
    m = sub.add_parser("mcp", help="serve read-only probes over MCP (stdio)")
    m.add_argument("rest", nargs=argparse.REMAINDER)
    m.set_defaults(fn=cmd_mcp)
    w = sub.add_parser("why", help="explain the last result")
    w.add_argument("--raw", action="store_true", help="also show exactly what the model wrote")
    w.set_defaults(fn=cmd_why)
    p = sub.add_parser("replay", help="show a recorded episode, without a model")
    p.add_argument("file")
    p.add_argument("--episode", help="episode id (default: the last in the file)")
    p.add_argument("--raw", action="store_true", help="also show exactly what the model wrote")
    p.set_defaults(fn=cmd_replay)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    # Only the first word selects a subcommand, so `tth how do I init git` is a request.
    if raw_args and raw_args[0] not in sub.choices and raw_args[0] not in ("-h", "--help"):
        raw_args = ["run"] + raw_args
    args = ap.parse_args(raw_args)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    from tth.sandbox.runner import SandboxUnavailable

    try:
        return int(args.fn(args))
    except ConfigError as e:
        ui.problem("Config problem:", str(e))
    except RequestTooLong as e:
        ui.problem(str(e))
    except SandboxUnavailable as e:
        ui.problem("The sandbox isn't available:", f"{e}. tth won't probe without it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
