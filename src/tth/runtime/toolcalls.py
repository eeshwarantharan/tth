"""Parse Hermes/Qwen-style tool calls and convert message formats."""

from __future__ import annotations

import json
import re
from typing import Any

from tth.agent.protocol import ModelTurn, ToolCall

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


_JUNK_TAGS = re.compile(r"<\|im_start\|>(?:assistant)?|<\|im_end\|>|<\|endoftext\|>")


def _bare_calls(text: str) -> list[ToolCall]:
    """JSON objects shaped like {"name": str, "arguments": {...}} outside <tool_call> tags."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            out.append(ToolCall(name=obj["name"], args=obj["arguments"]))
        i = end
    return out


def parse_qwen(text: str) -> ModelTurn:
    calls: list[ToolCall] = []
    for m in _TOOL_CALL.finditer(text):
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError("name must be a string and arguments an object")
            calls.append(ToolCall(name=name, args=args))
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            calls.append(ToolCall(name="", args={}, malformed=f"{type(e).__name__}: {e}"))
    content = _TOOL_CALL.sub("", text).strip()
    if not calls:
        # A correct tool-call body with a missing/wrong wrapper tag still counts (applies to every arm alike).
        calls = _bare_calls(_JUNK_TAGS.sub("", text))
        if calls:
            content = ""
    return ModelTurn(content=_JUNK_TAGS.sub("", content).strip(), tool_calls=calls, raw=text)


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our messages keep tool arguments as dicts; the OpenAI API wants JSON strings."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {**tc, "function": {**tc["function"],
                                    "arguments": json.dumps(tc["function"]["arguments"])}}
                for tc in m["tool_calls"]
            ]  # fmt: skip
        if m["role"] == "tool":
            m.pop("name", None)
        out.append(m)
    return out
