"""Read-only shell probes plus the stage/escalate/ask decisions.

Every probe runs inside the sandbox (read-only, offline, 3 s timeout, 8 KB cap)
and refuses secret paths. `stage` never executes anything: it classifies and
rewrites the command and returns what would be placed on the prompt.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tth.core.types import ToolDef
from tth.safety.classifier import Tier, classify
from tth.safety.rewriter import rewrite, selection_part
from tth.safety.secrets import is_secret_path
from tth.sandbox.runner import DEFAULT_CAP, DEFAULT_TIMEOUT, ProbeResult, run, run_shell

SAMPLE_LINES = 15


def _obj(props: dict[str, dict], required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOL_DEFS = [
    ToolDef("inspect_context", "See the machine: OS, shell, a directory listing (default: the working "
            "directory) and git branch/status.",
            _obj({"path": {"type": "string", "description": "Optional directory to list (default '.')"}}, []),
            read_only=True, sandboxed=True, timeout_s=DEFAULT_TIMEOUT, output_cap=DEFAULT_CAP),
    ToolDef("probe", "Look inside a file or binary without changing anything. kind: head (first lines), "
            "csv_header (columns + sample rows), json_keys (keys of a JSON/JSONL file), help (--help of a "
            "command), stat (size/type), tree (directory listing, depth 2).",
            _obj({"path": {"type": "string", "description": "File, directory or command name"},
                  "kind": {"type": "string", "enum": ["head", "csv_header", "json_keys", "help", "stat", "tree"]}},
                 ["path", "kind"]),
            read_only=True, sandboxed=True, timeout_s=DEFAULT_TIMEOUT, output_cap=DEFAULT_CAP),
    ToolDef("dry_run", "Run the read/selection part of a command in a read-only, offline sandbox. Returns exit "
            "code, match count and sample output (or the error). Deletions are previewed as a listing.",
            _obj({"command": {"type": "string"}}, ["command"]),
            read_only=True, sandboxed=True, timeout_s=DEFAULT_TIMEOUT, output_cap=DEFAULT_CAP),
    ToolDef("stage", "Place the final, verified command on the user's prompt (the user runs it, not you). "
            "Ends the task.",
            _obj({"command": {"type": "string"},
                  "explanation": {"type": "string", "description": "One short sentence for the user"}},
                 ["command"]),
            read_only=True, sandboxed=False),
    ToolDef("escalate", "Ask a stronger expert agent when you cannot verify the task locally (e.g. it needs the "
            "network or knowledge you lack). The user must consent first. Ends the task.",
            _obj({"reason": {"type": "string"}}, ["reason"]), read_only=True, sandboxed=False),
    ToolDef("ask_user", "Ask the user a clarifying question when the request is ambiguous or refers to something "
            "that does not exist. Ends the task.",
            _obj({"question": {"type": "string"}}, ["question"]), read_only=True, sandboxed=False),
]  # fmt: skip


@dataclass
class ToolResult:
    data: dict[str, Any]
    ok: bool = True
    decision: str | None = None


@dataclass
class ShellToolbox:
    cwd: Path
    shell: str = field(default_factory=lambda: Path(os.environ.get("SHELL", "/bin/zsh")).name)
    timeout: float = DEFAULT_TIMEOUT
    cap: int = DEFAULT_CAP
    allow_disposable: bool = False  # training/eval envs only; the product never sets this

    def defs(self) -> list[ToolDef]:
        return TOOL_DEFS

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        fn = getattr(self, f"t_{name}", None)
        spec = next((t for t in TOOL_DEFS if t.name == name), None)
        if fn is None or spec is None:
            return ToolResult({"error": f"unknown tool '{name}'"}, ok=False)
        props = spec.parameters["properties"]
        allowed = set(props)
        not_strings = sorted(
            k for k, v in args.items() if props.get(k, {}).get("type") == "string" and not isinstance(v, str)
        )
        if not_strings:
            return ToolResult({"error": f"bad arguments: {not_strings} must be strings"}, ok=False)
        extra = sorted(set(args) - allowed)
        missing = [k for k in spec.parameters["required"] if k not in args]
        if extra or missing:
            return ToolResult({"error": f"bad arguments: unexpected {extra} missing {missing}; "
                                        f"{name} takes {sorted(allowed) or 'no arguments'}"}, ok=False)  # fmt: skip
        try:
            return fn(**args)
        except TypeError as e:
            return ToolResult({"error": f"bad arguments: {e}"}, ok=False)

    # ---- probes -------------------------------------------------------------------
    def _run(self, argv: list[str]) -> ProbeResult:
        return run(argv, self.cwd, timeout=self.timeout, cap=self.cap, allow_disposable=self.allow_disposable)

    def t_inspect_context(self, path: str = ".") -> ToolResult:
        if is_secret_path(path, self.cwd):
            return ToolResult({"error": f"refused: {path} is a secret path and is never probed"}, ok=False)
        ls = self._run(["ls", "-la", "--", path or "."])
        lines = ls.stdout.splitlines()
        data: dict[str, Any] = {
            "os": f"{platform.system()} {platform.release()}",
            "shell": self.shell,
            "cwd": str(self.cwd).replace(str(Path.home()), "~"),
            "listing": lines[:40] + ([f"... {len(lines) - 40} more"] if len(lines) > 40 else []),
        }
        if (self.cwd / ".git").exists():
            g = self._run(["git", "status", "-sb", "--untracked-files=normal"])
            data["git"] = g.stdout.splitlines()[:15]
        return ToolResult(data)

    def t_probe(self, path: str, kind: str = "head") -> ToolResult:
        if kind != "help" and is_secret_path(path, self.cwd):
            return ToolResult({"error": f"refused: {path} is a secret path and is never probed"}, ok=False)
        if kind == "head":
            r = self._run(["head", "-n", "20", "--", path])
        elif kind == "csv_header":
            r = self._run(["head", "-n", "4", "--", path])
            if r.ok:
                rows = r.stdout.splitlines()
                cols = rows[0].split(",") if rows else []
                total = self._run(["wc", "-l", "--", path]).stdout.split()
                return ToolResult({"path": path, "columns": cols, "sample_rows": rows[1:],
                                   "line_count": int(total[0]) if total else None})  # fmt: skip
        elif kind == "json_keys":
            r = self._run(["head", "-c", "4000", "--", path])
            if r.ok:
                return ToolResult(_json_keys(path, r.stdout, self))
        elif kind == "help":
            if not path.replace("-", "").replace("_", "").isalnum():
                return ToolResult({"error": "help takes a bare command name"}, ok=False)
            r = self._run([path, "--help"])
            if not r.stdout.strip():
                r = ProbeResult(r.exit_code, r.stderr, "", r.truncated, r.timed_out, r.seconds)
            return ToolResult({"command": path, "help": r.stdout.splitlines()[:60], "exit_code": r.exit_code})
        elif kind == "stat":
            r = self._run(["ls", "-ld", "--", path])
            if r.ok:
                f = self._run(["file", "-b", "--", path])
                return ToolResult({"path": path, "ls": r.stdout.strip(), "type": f.stdout.strip()})
        elif kind == "tree":
            r = self._run(["find", path, "-maxdepth", "2", "-not", "-path", "*/.git/*"])
            if r.ok:
                lines = sorted(r.stdout.splitlines())
                return ToolResult({"path": path, "entries": lines[:80], "count": len(lines)})
        else:
            return ToolResult({"error": f"unknown probe kind '{kind}'"}, ok=False)
        if not r.ok:
            return ToolResult({"error": (r.stderr or "failed").strip()[:500], "exit_code": r.exit_code,
                               "timed_out": r.timed_out}, ok=False)  # fmt: skip
        return ToolResult({"path": path, "lines": r.stdout.splitlines(), "truncated": r.truncated})

    def t_dry_run(self, command: str) -> ToolResult:
        c = classify(command)
        sel = selection_part(command)
        if sel is None:
            note = (
                f"not dry-runnable: command modifies files/state (tier: {c.tier.label}). "
                "If this matches the user request, call stage directly with it."
                if c.tier.label == "changes-files"
                else "not dry-runnable: only read-only commands (or deletion previews) run"
            )
            return ToolResult(
                {"ran": False, "tier": c.tier.label, "reasons": c.reasons, "note": note},
                ok=False,
            )
        r = run_shell(sel, self.cwd, timeout=self.timeout, cap=self.cap, allow_disposable=self.allow_disposable)
        out = r.stdout.splitlines()
        data: dict[str, Any] = {
            "ran": sel, "exit_code": r.exit_code, "lines": len(out), "sample": out[:SAMPLE_LINES],
            "tier": c.tier.label,
        }  # fmt: skip
        if sel != command:
            data["note"] = "ran the selection/preview part only"
        if r.stderr.strip():
            data["stderr"] = r.stderr.strip()[:600]
        ok = r.ok
        if r.timed_out:
            data["error"] = f"timed out after {self.timeout}s"
            ok = False
        elif (
            not out
            and r.stderr.strip()
            and not any(r.stderr.strip().startswith(s) for s in ("stat: ./.env", "stat: ./"))
        ):
            data["error"] = r.stderr.strip()[:600]
            ok = False
        return ToolResult(data, ok=ok)

    # ---- decisions ----------------------------------------------------------------
    def t_stage(self, command: str, explanation: str = "") -> ToolResult:
        rw = rewrite(command, cwd=self.cwd)
        if rw.staged is None:
            return ToolResult({"staged": False, "tier": rw.tier.label, "notes": rw.notes,
                               "hint": "choose a safer command, ask the user, or escalate"}, ok=False)  # fmt: skip
        return ToolResult({"staged": True, "command": rw.staged, "tier": rw.tier.label,
                           "rewritten": rw.changed, "notes": rw.notes, "explanation": explanation},
                          decision="stage")  # fmt: skip

    def t_escalate(self, reason: str) -> ToolResult:
        return ToolResult({"escalated": True, "reason": reason}, decision="escalate")

    def t_ask_user(self, question: str) -> ToolResult:
        return ToolResult({"asked": True, "question": question}, decision="ask")


def _json_keys(path: str, head: str, box: ShellToolbox) -> dict[str, Any]:
    first = head.lstrip()[:1]
    if path.endswith(".jsonl") or (first == "{" and "\n{" in head):
        line = head.splitlines()[0] if head else ""
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return {"path": path, "error": "first line is not valid JSON", "first_line": line[:300]}
        n = box._run(["wc", "-l", "--", path]).stdout.split()
        return {"path": path, "format": "jsonl", "keys": list(rec) if isinstance(rec, dict) else None,
                "first_record": rec, "records": int(n[0]) if n else None}  # fmt: skip
    r = box._run(["jq", "-c", "if type==\"array\" then {type:\"array\",length:length,item_keys:(.[0]|keys?)} "
                  "else {type:type,keys:keys?} end", "--", path])  # fmt: skip
    if r.ok and r.stdout.strip():
        try:
            return {"path": path, "format": "json", **json.loads(r.stdout)}
        except json.JSONDecodeError:
            pass
    return {"path": path, "format": "unknown", "head": head[:500]}


def describe_staged(data: dict[str, Any]) -> str:
    return shlex.quote(data.get("command", ""))


TIER_BADGE = {Tier.READ_ONLY.label: "read-only", Tier.CHANGES_FILES.label: "changes files",
              Tier.NEEDS_REVIEW.label: "needs review"}  # fmt: skip
