"""What staging a command means. It classifies and rewrites, and never executes anything."""

from __future__ import annotations

from pathlib import Path

from tth.core.types import ToolResult
from tth.safety.rewriter import rewrite


def stage(command: str, explanation: str, cwd: Path) -> ToolResult:
    rw = rewrite(command, cwd=cwd)
    if rw.staged is None:
        return ToolResult({"staged": False, "tier": rw.tier.label, "notes": rw.notes,
                           "hint": "choose a safer command, ask the user, or escalate"}, ok=False)  # fmt: skip
    return ToolResult({"staged": True, "command": rw.staged, "tier": rw.tier.label,
                       "rewritten": rw.changed, "notes": rw.notes, "explanation": explanation},
                      decision="stage")  # fmt: skip
