"""Serve any gym environment as a real MCP server, so the student reaches its tools the way the product does.

The server lists the environment's tools for this episode, including randomized names and schemas, and every call
crosses the protocol. Whether a call ended the episode travels in the result's metadata, so the text the model reads
is exactly what the environment produced.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, TextContent, Tool

from tth.core.types import Environment, EpisodeStart, StepOutcome, ToolCall, ToolDef
from tth.mcp.client import ToolSession

META_KEY = "tth/step"


def _to_text(result: Any) -> tuple[str, bool]:
    if isinstance(result, str):
        return result, False
    return json.dumps(result, ensure_ascii=False), True


def build_env_server(env: Environment, tools: list[ToolDef]) -> Server:
    """One server for one episode. It deliberately doesn't validate arguments, because bad ones are part of training."""
    listed = [Tool(name=t.name, description=t.description, input_schema=t.parameters) for t in tools]

    async def on_list_tools(ctx: Any, params: Any) -> ListToolsResult:
        return ListToolsResult(tools=listed)

    async def on_call_tool(ctx: Any, params: CallToolRequestParams) -> CallToolResult:
        out = env.step(ToolCall(name=params.name, args=dict(params.arguments or {})))
        text, is_json = _to_text(out.result)
        result = CallToolResult(content=[TextContent(type="text", text=text)], is_error=not out.ok)
        result.meta = {META_KEY: {"done": out.done, "decision": out.decision, "source": out.source, "json": is_json}}
        return result

    return Server(f"tth-gym-{env.domain}", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


class McpEnvironment:
    """An Environment whose tools live behind MCP. Anything else, such as a verifier reading .st, reaches the inner env."""

    def __init__(self, env: Any) -> None:
        self.inner = env
        self.domain = env.domain
        self.session: ToolSession | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def reset(self) -> EpisodeStart:
        start = self.inner.reset()
        self._close_session()
        self.session = ToolSession(build_env_server(self.inner, start.tools))
        return EpisodeStart(system=start.system, user=start.user, tools=self.session.tools(), meta=start.meta)

    def step(self, call: ToolCall) -> StepOutcome:
        if call.malformed:  # it never parsed into a call, so there's nothing to send
            return self.inner.step(call)
        assert self.session is not None, "reset() first"
        text, is_error, meta = self.session.call_raw(call.name, call.args)
        step = meta.get(META_KEY, {})
        result = json.loads(text) if step.get("json") else text
        out = StepOutcome(result, ok=not is_error, done=bool(step.get("done")), decision=step.get("decision"),
                          source=step.get("source", "model"))  # fmt: skip
        if out.done:
            self._close_session()
        return out

    def on_text(self, text: str) -> StepOutcome:
        return self.inner.on_text(text)

    def finish(self, reason: str) -> StepOutcome:
        self._close_session()
        return self.inner.finish(reason)

    def _close_session(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def close(self) -> None:
        self._close_session()
        if hasattr(self.inner, "close"):
            self.inner.close()
