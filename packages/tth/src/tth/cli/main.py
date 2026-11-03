"""tth command line.

  tth run "<request>" [--out FILE]   inspect, verify, then stage a command
  tth init zsh|bash                  print shell integration (Ctrl-G and `t`)
  tth expert                         run the reference A2A expert agent
  tth mcp                            run the read-only probe tools as an MCP server
  tth setup                          check the install

tth never executes the final command: `run` writes it to --out (the shell
integration puts it on your prompt) or prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tth.core.config import Config, ConfigError, find_model, load_config, load_user_env, model_candidates, state_dir
from tth.trace.schema import TraceEvent, TraceLogger

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)
BADGE = {"read-only": (GREEN, "read-only"), "changes-files": (YELLOW, "changes files"),
         "needs-review": (RED, "needs review")}  # fmt: skip


def err(s: str = "") -> None:
    print(s, file=sys.stderr, flush=True)


def _summarize_result(ev: TraceEvent) -> str:
    r = ev.result
    if isinstance(r, dict):
        if "error" in r:
            return f"{RED}error{RESET} {str(r['error'])[:80]}"
        if "exit_code" in r and "lines" in r:
            return f"exit {r['exit_code']}, {r['lines']} lines" + (f" {DIM}(preview){RESET}" if r.get("note") else "")
        if "columns" in r:
            return "columns: " + ", ".join(r["columns"])[:80]
        if "keys" in r and r.get("keys"):
            return "keys: " + ", ".join(map(str, r["keys"]))[:80]
        if "listing" in r:
            return f"{len(r['listing'])} entries in {r.get('cwd')}"
        if r.get("staged") is False:
            return f"{RED}refused{RESET} {'; '.join(r.get('notes', []))[:80]}"
    return str(r)[:80].replace("\n", " ")


def log_row(ev: TraceEvent) -> None:
    if ev.call is None:
        return
    args = ev.call.args
    arg = args.get("command") or args.get("path") or args.get("reason") or args.get("question") or ""
    mark = f"{GREEN}✓{RESET}" if ev.ok else f"{RED}✗{RESET}"
    err(f"  {mark} {BOLD}{ev.call.tool:<15}{RESET} {DIM}{str(arg)[:60]}{RESET}  → {_summarize_result(ev)}")


def _read_tty(prompt: str) -> str:
    try:
        with open("/dev/tty") as tty:
            err(prompt)
            return tty.readline().strip()
    except OSError:
        return ""


def _start_expert() -> bool:
    """Start the local A2A expert server in the background (it is a server, not the user's command)."""
    import subprocess

    from tth.a2a.client import expert_reachable

    log = state_dir() / "expert.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "tth.expert.server"], stdout=log.open("a"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )  # fmt: skip
    for _ in range(40):
        time.sleep(0.25)
        if expert_reachable():
            return True
    return False


def stage_output(cmd: str, tier: str, notes: list[str], out: str | None, source: str) -> int:
    color, label = BADGE.get(tier, (RED, tier))
    err()
    err(f"  {color}{BOLD}[{label}]{RESET} {BOLD}{cmd}{RESET}  {DIM}({source}){RESET}")
    for n in notes:
        err(f"  {DIM}note: {n}{RESET}")
    if out:
        Path(out).write_text(cmd)  # the shell widget puts this on the prompt; we never run it
    else:
        print(cmd)
    return 0


def escalate_flow(request: str, ep_messages: list, reason: str, cwd: Path, shell: str, out: str | None,
                  assume_yes: bool) -> int:  # fmt: skip
    from tth.a2a.client import ask_expert, expert_reachable
    from tth.a2a.payload import build_payload
    from tth.agent.product import os_label
    from tth.safety.rewriter import rewrite

    p = build_payload(request, ep_messages, reason, os_label(), shell)
    err(f"\n  {YELLOW}{BOLD}can't verify locally → escalate?{RESET}  {DIM}{reason}{RESET}")
    err(
        f"  {DIM}This exact text would be sent to the expert agent (redacted: {', '.join(p.redactions) or 'nothing'}):{RESET}"
    )
    for line in p.text.splitlines():
        err(f"  {DIM}│{RESET} {line}")
    auto_yes = (
        assume_yes
        or os.environ.get("TTH_AUTO_ESCALATE", "").lower() in ("1", "true", "yes")
        or os.environ.get("TTH_YES", "").lower() in ("1", "true", "yes")
    )
    ans = "y" if auto_yes else _read_tty("  Send it? [y/N] ")
    if ans.lower() not in ("y", "yes"):
        err(f"  {DIM}Not sent. Nothing staged (local-only).{RESET}")
        return 1
    if not expert_reachable() and not _start_expert():
        err(f"  {RED}expert agent not reachable; start it with: tth expert{RESET}")
        return 1
    a = ask_expert(p.text)
    if not a.command:
        err(f"  expert: {a.explanation}")
        return 1
    rw = rewrite(a.command, cwd=cwd)
    if rw.staged is None:
        err(f"  {RED}expert's command refused by the safety layer:{RESET} {a.command}")
        for n in rw.notes:
            err(f"  {DIM}{n}{RESET}")
        return 1
    err(f"  {DIM}expert: {a.explanation}{RESET}")
    # Verify the expert's answer the same way as a local one: dry-run its read/selection part.
    from tth.mcp.toolbox import ShellToolbox
    from tth.safety.rewriter import selection_part

    source = "from expert, unverified locally"
    if selection_part(a.command) is not None:
        r = ShellToolbox(cwd, shell=shell).call("dry_run", {"command": a.command})  # noqa: S604
        sample = (r.data.get("sample") or [""])[0]
        err(f"  {GREEN if r.ok else RED}{'✓' if r.ok else '✗'}{RESET} {BOLD}{'dry_run':<15}{RESET} "
            f"{DIM}expert command{RESET}  → exit {r.data.get('exit_code')}, {r.data.get('lines', 0)} lines {DIM}{sample[:60]}{RESET}")  # fmt: skip
        source = "from expert, dry-run ok" if r.ok else "from expert, dry-run FAILED — review it"
    return stage_output(rw.staged, rw.tier.label, rw.notes, out, source)


def _make_policy(args: argparse.Namespace, config: Config) -> object | None:
    """The configured model, or None after explaining what's missing."""
    if args.backend == "mlx":
        model = find_model(config)
        if model is None:
            err(f"{RED}No model found.{RESET} Looked in: {', '.join(str(p) for p in model_candidates(config))}")
            err("Run `tth setup` for how to get one.")
            return None
        try:
            from tth.inference.mlx import MLXPolicy

            return MLXPolicy(model_path=str(model), adapter_path=args.adapter or config.adapter)
        except ImportError:
            err(f"{RED}The MLX backend isn't installed.{RESET} On Apple Silicon: pip install 'tth[mlx]'")
            return None
    try:
        from tth.inference.openai_compat import OpenAIPolicy
    except ImportError:
        err(f"{RED}The OpenAI-compatible backend isn't installed.{RESET} Run: pip install 'tth[openai]'")
        return None
    return OpenAIPolicy(model=args.model, base_url=args.base_url or config.base_url, api_key="local", thinking=False,
                        temperature=0.0, max_tokens=512)  # fmt: skip


def cmd_run(args: argparse.Namespace) -> int:
    from tth.agent.loop import run_episode
    from tth.agent.product import ProductEnv

    load_user_env()
    config = load_config()
    request = " ".join(args.request).strip()
    if not request:
        err('usage: tth run "<request>"')
        return 2
    cwd = Path.cwd()
    shell = Path(os.environ.get("SHELL", "/bin/zsh")).name
    args.backend = args.backend or config.backend
    err(f"{DIM}tth ▸ {request}{RESET}")
    policy = _make_policy(args, config)
    if policy is None:
        return 1
    env = ProductEnv(request, cwd, shell)
    logger = TraceLogger(state_dir() / "traces.jsonl")
    budget = args.budget or config.budget
    ep = run_episode(env, policy, budget=budget, logger=logger, on_event=log_row)  # type: ignore[arg-type]
    if env.decision == "stage" and env.staged:
        source = "verified locally" if env.verified else "not verified, read it carefully"
        return stage_output(env.staged["command"], env.staged["tier"], env.staged.get("notes", []), args.out, source)
    if env.decision == "ask":
        err(f"\n  {BOLD}tth asks:{RESET} {env.reason}")
        return 1
    return escalate_flow(request, ep.messages, env.reason or "could not verify locally", cwd, shell, args.out,
                         args.yes)  # fmt: skip


ZSH = r"""# tth shell integration (zsh). Type a request, press Ctrl-G: the command lands on your prompt.
_tth_widget() {
  local req="$BUFFER" out
  [[ -z "$req" ]] && return
  out=$(mktemp -t tth.XXXXXX) || return
  zle -I
  command tth run --out "$out" -- "$req" </dev/tty >/dev/tty 2>&1
  if [[ -s "$out" ]]; then BUFFER="$(<"$out")"; CURSOR=${#BUFFER}; fi
  rm -f "$out"
  zle reset-prompt
}
zle -N _tth_widget
bindkey '^G' _tth_widget
# Or: t <request>  -> the command appears on your next prompt (print -z), never executed.
t() {
  local out c; out=$(mktemp -t tth.XXXXXX) || return
  command tth run --out "$out" -- "$*"
  c="$(<"$out")"; rm -f "$out"
  [[ -n "$c" ]] && print -z -- "$c"
}
"""

BASH = r"""# tth shell integration (bash). Type a request, press Ctrl-G: the line is replaced by the command.
_tth_readline() {
  local req="$READLINE_LINE" out
  [[ -z "$req" ]] && return
  out=$(mktemp -t tth.XXXXXX) || return
  command tth run --out "$out" -- "$req" </dev/tty >/dev/tty 2>&1
  if [[ -s "$out" ]]; then READLINE_LINE="$(<"$out")"; READLINE_POINT=${#READLINE_LINE}; fi
  rm -f "$out"
}
bind -x '"\C-g": _tth_readline'
"""


def cmd_init(args: argparse.Namespace) -> int:
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
        err(f"    hf download mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --local-dir '{target}'")
    if config.backend == "mlx":
        try:
            import mlx_lm  # noqa: F401

            err("backend: mlx ok")
        except ImportError:
            ok = False
            err("backend: mlx isn't installed. On Apple Silicon: pip install 'tth[mlx]'")
    err(f'shell integration: add to your rc file ->  eval "$(tth init {Path(os.environ.get("SHELL", "zsh")).name})"')
    return 0 if ok else 1


def cmd_expert(args: argparse.Namespace) -> int:
    load_user_env()
    try:
        from tth.expert.server import main as expert_main
    except ImportError:
        err(f"{RED}The expert server isn't installed.{RESET} Run: pip install 'tth[expert]'")
        return 1
    expert_main()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from tth.mcp.server import main as mcp_main

    mcp_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tth", description="tiny terminal helper: looks before it acts.")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="turn a request into a staged command")
    r.add_argument("request", nargs="+")
    r.add_argument("--out", help="write the staged command here (used by the shell widget)")
    r.add_argument("--adapter", help="LoRA adapter directory (MLX)")
    r.add_argument("--backend", choices=["mlx", "openai"], default=None)
    r.add_argument("--base-url", default=None,
                   help="llama-server URL for --backend openai")  # fmt: skip
    r.add_argument("--model", default="local")
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--yes", action="store_true", help="auto-approve escalation (for recorded runs only)")
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("init", help="print shell integration")
    i.add_argument("shell", choices=["zsh", "bash"])
    i.set_defaults(fn=cmd_init)
    sub.add_parser("setup", help="check the install").set_defaults(fn=cmd_setup)
    sub.add_parser("expert", help="run the reference A2A expert").set_defaults(fn=cmd_expert)
    sub.add_parser("mcp", help="serve read-only probes over MCP (stdio)").set_defaults(fn=cmd_mcp)
    KNOWN = {"run", "init", "setup", "expert", "mcp"}
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and not any(a in KNOWN for a in raw_args) and not any(a in ("-h", "--help") for a in raw_args):
        raw_args = ["run"] + raw_args
    args = ap.parse_args(raw_args)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    from tth.sandbox.runner import SandboxUnavailable

    try:
        return int(args.fn(args))
    except ConfigError as e:
        err(f"{RED}Config problem:{RESET} {e}")
    except SandboxUnavailable as e:
        err(f"{RED}The sandbox isn't available:{RESET} {e}. tth won't probe without it.")
    return 1


def _json(x: object) -> str:
    return json.dumps(x)


if __name__ == "__main__":
    sys.exit(main())
