"""What staging a command means. It classifies and rewrites, and never executes anything."""

from __future__ import annotations

from pathlib import Path

from tth.core.types import ToolResult
from tth.safety.rewriter import rewrite


def has_control_characters(text: str) -> bool:
    return any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in text)


def stage(command: str, explanation: str, cwd: Path) -> ToolResult:
    if has_control_characters(command):
        return ToolResult({"staged": False, "tier": "denied", "notes": ["refused: the command has control characters, "
                           "which could hide what it does on screen"]}, ok=False)  # fmt: skip
    rw = rewrite(command, cwd=cwd)
    if rw.staged is None:
        return ToolResult({"staged": False, "tier": rw.tier.label, "notes": rw.notes,
                           "hint": "choose a safer command, ask the user, or escalate"}, ok=False)  # fmt: skip
    return ToolResult({"staged": True, "command": rw.staged, "tier": rw.tier.label,
                       "rewritten": rw.changed, "notes": rw.notes, "explanation": explanation},
                      decision="stage")  # fmt: skip
