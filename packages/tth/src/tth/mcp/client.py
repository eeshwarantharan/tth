"""The agent's only route to tools: a synchronous session over a real MCP client.

Tools are discovered with tools/list and called with tools/call. The agent loop is synchronous and the SDK is
async, so a background thread owns the connection. The default in-memory transport speaks the same protocol as
stdio at a fraction of the startup cost (see evals/diagnostics/results/mcp_overhead.json).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.client.client import Client
from mcp.server.lowlevel import Server
from mcp.server.mcpserver import MCPServer

from tth.core.types import ToolDef, ToolResult
from tth.mcp.server import build_server

CONNECT_TIMEOUT = 15.0
CALL_TIMEOUT = 30.0  # the sandbox's own timeout is much shorter, so this only catches a broken connection


class ToolSessionError(Exception):
    """The connection to a tool server failed. The message says what happened."""


def _clean_schema(schema: Any) -> Any:
    """Drop pydantic's generated titles and the defaults, which cost prompt tokens and tell the model nothing."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k not in ("title", "default")}
    return schema


class ToolSession:
    def __init__(self, target: MCPServer | Server | StdioServerParameters) -> None:
        # Bad model arguments are normal here, and they go back to the model. They shouldn't reach the user's terminal.
        logging.getLogger("mcp").setLevel(logging.CRITICAL)
        self._target = target
        self._loop = asyncio.new_event_loop()
        self._client: Client | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._defs: dict[str, ToolDef] = {}
        self._thread = threading.Thread(target=self._loop.run_until_complete, args=(self._main(),), daemon=True)
        self._thread.start()
        if not self._ready.wait(CONNECT_TIMEOUT):
            raise ToolSessionError(f"couldn't connect to the tool server within {CONNECT_TIMEOUT:.0f}s")
        if self._failure:
            raise ToolSessionError(f"couldn't connect to the tool server: {self._failure}") from self._failure
        listed = self._run(self._require_client().list_tools())
        for tool in listed.tools:
            self._defs[tool.name] = ToolDef(
                name=tool.name, description=tool.description or "", parameters=_clean_schema(tool.input_schema),
                read_only=bool(tool.annotations and tool.annotations.read_only_hint), sandboxed=True,
            )  # fmt: skip

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with Client(self._target) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - reported to the caller on the other thread
            self._failure = e
            self._ready.set()

    def _require_client(self) -> Client:
        if self._client is None:
            raise ToolSessionError("the tool server connection is closed")
        return self._client

    def _run(self, coroutine: Any) -> Any:
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(CALL_TIMEOUT)

    def tools(self, only: set[str] | None = None) -> list[ToolDef]:
        return [d for name, d in self._defs.items() if only is None or name in only]

    def call_raw(self, name: str, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
        """The result as sent: its text, whether it's an error, and its metadata. Any name may be sent."""
        result = self._run(self._require_client().call_tool(name, args))
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        return text, bool(result.is_error), dict(result.meta or {})

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self._defs:
            return ToolResult({"error": f"unknown tool '{name}'"}, ok=False)
        result = self._run(self._require_client().call_tool(name, args))
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        if result.is_error:
            first_line = text.splitlines()[0] if text else "the tool failed"
            return ToolResult({"error": f"bad arguments: {first_line}"}, ok=False)
        data = result.structured_content or json.loads(text)
        ok = bool(data.pop("ok", True))
        return ToolResult(data, ok=ok)

    def close(self) -> None:
        if self._stop is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=5)
        self._client = None

    def __enter__(self) -> ToolSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def connect(cwd: Path, shell: str | None = None, transport: str = "memory") -> ToolSession:
    """Connect to tth's probes for one directory, in-process or as a separate server process."""
    if transport == "memory":
        return ToolSession(build_server(cwd, shell=shell))
    if transport == "stdio":
        args = ["-m", "tth.mcp.server", "--cwd", str(cwd)] + (["--shell", shell] if shell else [])
        return ToolSession(StdioServerParameters(command=sys.executable, args=args))
    raise ValueError(f"unknown MCP transport: {transport}")
