"""The MCP server that gives any agent, including tth's own, safe access to a directory.

The sandbox and the secret checks live here, so no client can skip them. Only read-only tools are exposed.
Staging is deliberately absent: it writes to the user's prompt, and must never be reachable remotely.

Run for another agent: tth mcp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from tth.core.types import ToolResult
from tth.mcp.toolbox import TOOL_DEFS, ShellToolbox
from tth.packs.data import CompileError, Query
from tth.safety.rewriter import rewrite
from tth.sandbox.runner import SandboxUnavailable

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
_DESCRIPTIONS = {t.name: t.description for t in TOOL_DEFS}
AGGREGATE_DESCRIPTION = (
    "Answer a total, average, count, min, max or filtered list over a CSV, JSON or JSONL file. Say what you want, and "
    "this writes and tests the command. It returns the command and its result. Then call stage with that command."
)
_PROBE_KINDS = Literal["head", "csv_header", "json_keys", "help", "stat", "tree"]


def build_server(cwd: Path, shell: str | None = None, allow_disposable: bool = False) -> MCPServer:
    """A server for one directory. allow_disposable is for training and evals on throwaway machines."""
    box = ShellToolbox(Path(cwd), allow_disposable=allow_disposable)
    if shell:
        box.shell = shell
    server = MCPServer(
        name="tth-probes",
        instructions="Read-only, sandboxed shell probes. Nothing here changes files or reaches the network.",
    )

    def respond(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            r: ToolResult = box.call(tool, args)
        except SandboxUnavailable as e:
            return {"ok": False, "error": f"the sandbox isn't available: {e}"}
        return {"ok": r.ok, **r.data}

    @server.tool(description=_DESCRIPTIONS["inspect_context"], annotations=READ_ONLY)
    def inspect_context(
        path: Annotated[str, Field(description="Optional directory to list (default '.')")] = ".",
    ) -> dict[str, Any]:
        return respond("inspect_context", {"path": path})

    @server.tool(description=_DESCRIPTIONS["probe"], annotations=READ_ONLY)
    def probe(
        path: Annotated[str, Field(description="File, directory or command name")], kind: _PROBE_KINDS
    ) -> dict[str, Any]:
        return respond("probe", {"path": path, "kind": kind})

    @server.tool(description=_DESCRIPTIONS["dry_run"], annotations=READ_ONLY)
    def dry_run(command: str) -> dict[str, Any]:
        return respond("dry_run", {"command": command})

    @server.tool(description=AGGREGATE_DESCRIPTION, annotations=READ_ONLY)
    def aggregate(
        file: Annotated[str, Field(description="A .csv, .json or .jsonl file")],
        op: Literal["sum", "avg", "count", "min", "max", "list"],
        column: Annotated[str, Field(description="Column to use, or arithmetic such as 'units * price'")] = "",
        where_column: Annotated[str, Field(description="Only use rows where this column matches")] = "",
        where_op: Literal["eq", "ne", "gt", "ge", "lt", "le"] = "eq",
        where_value: str = "",
    ) -> dict[str, Any]:
        query = Query(file, op, column, where_column, where_op, where_value)
        try:
            r = box.aggregate(query)
        except CompileError as e:
            return {"ok": False, "error": str(e)}
        except SandboxUnavailable as e:
            return {"ok": False, "error": f"the sandbox isn't available: {e}"}
        return {"ok": r.ok, **r.data}

    @server.tool(
        description="Classify a command's safety tier and show tth's safe rewrite. Never executes it.",
        annotations=READ_ONLY,
    )
    def classify(command: str) -> dict[str, Any]:
        rw = rewrite(command, cwd=box.cwd)
        return {"ok": True, "tier": rw.tier.label, "staged": rw.staged, "preview": rw.preview, "notes": rw.notes,
                "reasons": rw.reasons}  # fmt: skip

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tth mcp", description="Serve tth's read-only probes over MCP (stdio).")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--shell", default=None)
    args = ap.parse_args(argv)
    build_server(Path(args.cwd).resolve(), shell=args.shell).run("stdio")


if __name__ == "__main__":
    main()
