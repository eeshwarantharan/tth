"""What does routing every tool call through MCP cost, compared with calling the code directly?

Measures connection time and per-call time for a real sandboxed dry_run, directly, over in-memory MCP and over
stdio MCP. Results are on one machine, so treat them as a rough scale, not a benchmark.

uv run python -m evals.diagnostics.mcp_overhead evals/diagnostics/results/mcp_overhead.json
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

from tth.mcp.client import connect
from tth.mcp.toolbox import ShellToolbox

CALLS = 30
CONNECTS = 5
ARGS = {"command": "wc -l a.txt"}


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _summary(values: list[float]) -> dict[str, float]:
    return {"median_ms": round(statistics.median(values), 2), "max_ms": round(max(values), 2)}


def measure(transport: str, cwd: Path) -> dict[str, dict[str, float]]:
    connects = []
    for _ in range(CONNECTS):
        t = time.perf_counter()
        session = connect(cwd, shell="zsh", transport=transport)
        connects.append(_ms(t))
        session.close()
    calls = []
    with connect(cwd, shell="zsh", transport=transport) as session:
        for _ in range(CALLS):
            t = time.perf_counter()
            assert session.call("dry_run", ARGS).ok
            calls.append(_ms(t))
    return {"connect": _summary(connects), "dry_run": _summary(calls)}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp).resolve()
        (cwd / "a.txt").write_text("hello\n")
        box = ShellToolbox(cwd, shell="zsh")
        direct = []
        for _ in range(CALLS):
            t = time.perf_counter()
            assert box.call("dry_run", ARGS).ok
            direct.append(_ms(t))
        results = {
            "machine": f"{platform.system()} {platform.machine()}",
            "calls": CALLS,
            "connects": CONNECTS,
            "direct": {"dry_run": _summary(direct)},
            "memory": measure("memory", cwd),
            "stdio": measure("stdio", cwd),
        }
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
