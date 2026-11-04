"""What staging a command means. It classifies and rewrites, and never executes anything."""

from __future__ import annotations

import shutil
from difflib import SequenceMatcher
from pathlib import Path

from tth.core.types import ToolResult
from tth.safety.classifier import classify
from tth.safety.parser import parse
from tth.safety.rewriter import rewrite

# Shell builtins aren't on PATH, and they're real commands.
_BUILTINS = {
    "cd", "echo", "printf", "print", "test", "[", "[[", "type", "true", "false", "pwd", "export", "unset", "set", "alias",
    "unalias", "source", ".", "eval", "exec", "command", "builtin", "read", "wait", "kill", "jobs", "fg", "bg", "ulimit",
    "umask", "time", "history", "trap", "shift", "return", ":", "hash", "getopts", "pushd", "popd", "dirs", "whence",
    "where", "which", "disown", "let", "local", "declare", "typeset", "readonly", "rehash", "exit", "logout",
}  # fmt: skip
_ECHO_LIMIT = 0.85


def has_control_characters(text: str) -> bool:
    return any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in text)


def echoes_request(command: str, request: str) -> bool:
    """Small models sometimes return the request itself as the command."""
    if not request:
        return False
    a, b = " ".join(command.lower().split()), " ".join(request.lower().split())
    return a == b or SequenceMatcher(None, a, b).ratio() >= _ECHO_LIMIT


def missing_programs(command: str) -> list[str]:
    """Programs the command names that this machine doesn't have. Computed names can't be checked, so they're skipped."""
    missing = []
    for sc in parse(command).commands:
        if not sc.argv or 0 in sc.dynamic_args:
            continue
        name = sc.argv[0]
        if name in _BUILTINS or shutil.which(name):
            continue
        if "/" in name:
            continue  # classified separately as running a local script
        missing.append(name)
    return missing


def refuse(command: str, request: str = "") -> str | None:
    """Why this text can't be offered as a command, or None."""
    if has_control_characters(command):
        return "refused: the command has control characters, which could hide what it does on screen"
    if echoes_request(command, request):
        return "refused: that's the request itself, not a shell command"
    missing = missing_programs(command)
    if missing:
        return f"refused: {', '.join(missing)} isn't on this machine, so this can't be a working command"
    return None


def stage(command: str, explanation: str, cwd: Path, request: str = "") -> ToolResult:
    why_not = refuse(command, request)
    if why_not:
        return ToolResult({"staged": False, "tier": classify(command).tier.label, "notes": [why_not],
                           "hint": "write a real shell command, ask the user, or escalate"}, ok=False)  # fmt: skip
    rw = rewrite(command, cwd=cwd)
    if rw.staged is None:
        return ToolResult({"staged": False, "tier": rw.tier.label, "notes": rw.notes,
                           "hint": "choose a safer command, ask the user, or escalate"}, ok=False)  # fmt: skip
    return ToolResult({"staged": True, "command": rw.staged, "tier": rw.tier.label,
                       "rewritten": rw.changed, "notes": rw.notes, "explanation": explanation},
                      decision="stage")  # fmt: skip
