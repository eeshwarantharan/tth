"""What the user sees. Everything here is drawn from trace events, so the terminal, `tth why` and replays agree."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from tth.trace.schema import TraceEvent


class Style:
    """ANSI styling, off when stderr isn't a terminal or NO_COLOR is set."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = sys.stderr.isatty() and "NO_COLOR" not in os.environ
        on = enabled
        self.green, self.yellow, self.red = ("\033[32m", "\033[33m", "\033[31m") if on else ("", "", "")
        self.dim, self.bold, self.reset = ("\033[2m", "\033[1m", "\033[0m") if on else ("", "", "")


def err(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


@dataclass
class Outcome:
    """How a request ended, in terms the user can act on."""

    decision: str  # stage | ask | declined | failed
    command: str | None = None
    tier: str | None = None
    notes: list[str] = field(default_factory=list)
    origin: str = "local"  # local | expert
    verified: bool = False
    message: str = ""

    @property
    def badge(self) -> str:
        checked = "verified" if self.verified else "not verified"
        return f"{self.tier or 'unknown'} · {self.origin} · {checked}"


def _clip(text: object, limit: int = 72) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _probe_phrase(args: dict) -> str:
    path, kind = args.get("path", ""), args.get("kind", "head")
    return {
        "csv_header": f"reading the columns of {path}",
        "json_keys": f"reading the keys of {path}",
        "help": f"checking {path} --help",
        "stat": f"checking {path}",
        "tree": f"listing {path}",
    }.get(kind, f"reading {path}")


def _result_summary(result: object) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("columns"):
        return "columns: " + ", ".join(map(str, result["columns"]))
    if result.get("keys"):
        return "keys: " + ", ".join(map(str, result["keys"]))
    return ""


def describe(ev: TraceEvent, s: Style) -> str | None:
    """One line for one event, or None if there's nothing the user needs to see."""
    if ev.meta.get("kind") == "start":
        return None
    if ev.meta.get("source") == "gate" and ev.decision == "stage":
        return f"  {s.dim}⋯ taking the command it tested{s.reset}"
    if ev.meta.get("source") == "gate" and ev.decision == "escalate":
        return f"  {s.red}✗{s.reset} {s.dim}it repeated a failing command, so it needs help{s.reset}"
    if ev.call is None:
        if isinstance(ev.result, dict):
            return None
        return f"  {s.red}✗{s.reset} {s.dim}the model gave no usable tool call{s.reset}"
    tool, args, result = ev.call.tool, ev.call.args, ev.result
    if tool == "inspect_context":
        return f"  {s.dim}⋯ looking at the folder{s.reset}"
    if tool == "probe":
        summary = _result_summary(result)
        tail = f" ({_clip(summary, 48)})" if summary else ""
        if not ev.ok:
            return f"  {s.red}✗{s.reset} {s.dim}{_probe_phrase(args)}: {_clip(_error(result))}{s.reset}"
        return f"  {s.dim}⋯ {_probe_phrase(args)}{tail}{s.reset}"
    if tool == "dry_run":
        return _describe_dry_run(args, result, ev.ok, s)
    if tool == "stage" and not ev.ok:
        notes = result.get("notes", []) if isinstance(result, dict) else []
        return f"  {s.red}✗{s.reset} {s.dim}the safety layer refused that command: {_clip('; '.join(notes))}{s.reset}"
    if tool not in ("stage", "escalate", "ask_user"):
        return f"  {s.dim}⋯ {tool}{s.reset}"
    return None


def _error(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("error") or result.get("note") or result.get("stderr") or "failed")
    return "failed"


def _describe_dry_run(args: dict, result: object, ok: bool, s: Style) -> str:
    command = _clip(args.get("command", ""), 56)
    if not ok or not isinstance(result, dict):
        return f"  {s.red}✗{s.reset} {s.dim}tested {command}: {_clip(_error(result), 60)}{s.reset}"
    lines, sample = result.get("lines", 0), result.get("sample") or []
    plural = "" if lines == 1 else "s"
    first = f": {_clip(sample[0], 40)}" if sample else ""
    return f"  {s.green}✓{s.reset} {s.dim}tested {command} ({lines} line{plural}{first}){s.reset}"


class TerminalRenderer:
    def __init__(self, style: Style | None = None) -> None:
        self.s = style or Style()

    def request(self, text: str) -> None:
        err(f"{self.s.dim}tth ▸ {text}{self.s.reset}")

    def event(self, ev: TraceEvent) -> None:
        line = describe(ev, self.s)
        if line:
            err(line)

    def outcome(self, o: Outcome) -> None:
        s = self.s
        if o.decision == "stage":
            color = {"read-only": s.green, "changes-files": s.yellow}.get(o.tier or "", s.red)
            err()
            err(f"  {color}{s.bold}[{o.badge}]{s.reset}")
            err(f"  {s.bold}{o.command}{s.reset}")
            for note in o.notes:
                err(f"  {s.dim}note: {note}{s.reset}")
        elif o.message:
            err(f"  {o.message}")

    def problem(self, text: str, fix: str = "") -> None:
        err(f"{self.s.red}{text}{self.s.reset}" + (f" {fix}" if fix else ""))
