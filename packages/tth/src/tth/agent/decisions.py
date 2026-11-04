"""The tools that end an episode. They run in the client, never on an MCP server.

Staging writes to the user's own prompt, and escalating needs the user's consent. Neither may be reachable
remotely, so they aren't served over MCP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tth.core.types import ToolDef, ToolResult
from tth.safety.staging import stage


def _obj(props: dict[str, dict], required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


DECISION_DEFS = [
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
DECISION_NAMES = {d.name for d in DECISION_DEFS}


def decide(name: str, args: dict[str, Any], cwd: Path, request: str = "") -> ToolResult:
    spec = next(d for d in DECISION_DEFS if d.name == name)
    missing = [k for k in spec.parameters["required"] if k not in args]
    extra = sorted(set(args) - set(spec.parameters["properties"]))
    not_strings = sorted(k for k, v in args.items() if not isinstance(v, str))
    if not_strings:
        return ToolResult({"error": f"bad arguments: {not_strings} must be strings"}, ok=False)
    if missing or extra:
        return ToolResult({"error": f"bad arguments: unexpected {extra} missing {missing}"}, ok=False)
    if name == "stage":
        return stage(args["command"], args.get("explanation", ""), cwd, request)
    if name == "escalate":
        return ToolResult({"escalated": True, "reason": args["reason"]}, decision="escalate")
    return ToolResult({"asked": True, "question": args["question"]}, decision="ask")
