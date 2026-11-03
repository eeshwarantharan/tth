"""Parse Hermes/Qwen-style tool calls and convert message formats."""

from __future__ import annotations

import json
import re
from typing import Any

from tth.core.types import ModelTurn, ToolCall

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


_FN_CALL = re.compile(
    r"<function\s+name=[\"'](?P<name>[^\"']+)[\"'][^>]*>(?P<body>.*?)</function>",
    re.DOTALL,
)
_ARGS_BLOCK = re.compile(r"<arguments>(?P<args>.*?)</arguments>", re.DOTALL)
_TAG_PARAM = re.compile(r"<(?P<key>[a-zA-Z0-9_]+)>(?P<val>.*?)</(?P=key)>", re.DOTALL)


def _xml_calls(text: str) -> list[ToolCall]:
    out: list[ToolCall] = []
    for m in _FN_CALL.finditer(text):
        name = m.group("name")
        body = m.group("body")
        m_args = _ARGS_BLOCK.search(body)
        args: dict[str, Any] = {}
        if m_args:
            raw_args = m_args.group("args").strip()
            if raw_args.startswith("{") and raw_args.endswith("}"):
                try:
                    loaded = json.loads(raw_args)
                    if isinstance(loaded, dict):
                        args = loaded
                except json.JSONDecodeError:
                    pass
            if not args:
                for p in _TAG_PARAM.finditer(raw_args):
                    args[p.group("key")] = p.group("val").strip()
        if args:
            out.append(ToolCall(name=name, args=args))
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
    if not calls:
        calls = _xml_calls(text)
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
