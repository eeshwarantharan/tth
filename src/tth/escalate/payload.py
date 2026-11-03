"""Build the exact escalation payload the user sees and approves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tth.escalate.redact import redact

MAX_RESULT_CHARS = 1200


@dataclass
class Payload:
    text: str  # exactly what is sent
    redactions: list[str]


def _short(x: Any, hits: list[str]) -> str:
    """Redact first: truncating a secret can cut off what the patterns need to recognise it."""
    s, found = redact(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False))
    hits.extend(found)
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + " …[truncated]"


def build_payload(request: str, messages: list[dict[str, Any]], reason: str, os_name: str, shell: str) -> Payload:
    """Summarise what the local agent saw. Only tool results it already obtained are included."""
    lines = [
        f"User request: {request}",
        f"Environment: {os_name}, shell {shell}",
        f"Local agent's reason for escalating: {reason}",
        "What the local agent observed (read-only probes):",
    ]
    hits: list[str] = []
    calls: dict[str, tuple[str, dict]] = {}
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                calls[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
        elif m.get("role") == "tool":
            name, args = calls.get(m.get("tool_call_id", ""), (m.get("name", "?"), {}))
            if name in ("escalate", "ask_user", "stage"):
                continue
            lines.append(f"- {name}({json.dumps(args, ensure_ascii=False)}) -> {_short(m.get('content', ''), hits)}")
    if len(lines) == 4:
        lines.append("- (nothing)")
    lines.append("Reply with JSON: {\"command\": \"<one shell command>\", \"explanation\": \"<one sentence>\"}. "
                 "If it cannot be done safely, set command to \"\" and explain.")  # fmt: skip
    text, found = redact("\n".join(lines))
    return Payload(text=text, redactions=sorted(set(hits + found)))
