"""Expose tth's read-only probes over MCP (stdio), e.g. for other agents.

Only read-only, sandboxed tools are exposed, plus `classify` which runs the
safety classifier/rewriter on a command without staging or executing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from tth.mcp.toolbox import ShellToolbox
from tth.safety.rewriter import rewrite

server = MCPServer(name="tth-probes", instructions="Read-only, sandboxed shell probes. Nothing here executes "
                   "a command outside the sandbox or changes files.")  # fmt: skip


def _box() -> ShellToolbox:
    return ShellToolbox(Path.cwd())


@server.tool(description="OS, shell, directory listing and git status (read-only, sandboxed, 3s, 8KB).")
def inspect_context(path: str = ".") -> dict[str, Any]:
    return _box().call("inspect_context", {"path": path}).data


@server.tool(description="Look inside a file: kind is head|csv_header|json_keys|help|stat|tree (read-only, sandboxed).")
def probe(path: str, kind: str = "head") -> dict[str, Any]:
    return _box().call("probe", {"path": path, "kind": kind}).data


@server.tool(description="Run the read/selection part of a command in a read-only offline sandbox.")
def dry_run(command: str) -> dict[str, Any]:
    return _box().call("dry_run", {"command": command}).data


@server.tool(description="Classify a command's safety tier and show tth's safe rewrite. Never executes it.")
def classify(command: str) -> dict[str, Any]:
    rw = rewrite(command, cwd=Path.cwd())
    return {"tier": rw.tier.label, "staged": rw.staged, "preview": rw.preview, "notes": rw.notes,
            "reasons": rw.reasons}  # fmt: skip


def main() -> None:
    server.run("stdio")
