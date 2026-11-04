"""MCP is the agent's only route to tools, so the contract is tested over a real MCP client."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tth.agent.decisions import DECISION_NAMES
from tth.agent.loop import run_episode
from tth.agent.product import ProductEnv
from tth.core.types import ModelTurn, ToolDef
from tth.inference.toolcalls import parse_qwen
from tth.mcp.client import ToolSession, connect
from tth.mcp.toolbox import TOOL_DEFS

ROOT = Path(__file__).resolve().parents[1]
PROBES = {"inspect_context", "probe", "dry_run"}
SERVED = PROBES | {"aggregate", "classify"}


@pytest.fixture(params=["memory", "stdio"])
def session(request: pytest.FixtureRequest, tmp_path: Path):
    """Every contract test runs over both transports, so they can't drift apart."""
    (tmp_path / "a.txt").write_text("hello\nworld\n")
    (tmp_path / ".env").write_text("SECRET=hunter2\n")
    with connect(tmp_path, shell="zsh", transport=request.param) as s:
        yield s


def test_tools_are_discovered_over_the_protocol(session: ToolSession) -> None:
    assert {t.name for t in session.tools()} == SERVED


def test_every_tool_is_declared_read_only(session: ToolSession) -> None:
    assert all(t.read_only for t in session.tools())


def test_staging_is_never_reachable_over_mcp(session: ToolSession) -> None:
    """Invariant 1. Staging writes to the user's prompt."""
    assert not DECISION_NAMES & {t.name for t in session.tools()}
    assert session.call("stage", {"command": "ls"}).ok is False


def test_schemas_match_what_the_model_is_trained_on(session: ToolSession) -> None:
    trained = {t.name: t.parameters for t in TOOL_DEFS if t.name in PROBES}
    served = {t.name: t.parameters for t in session.tools(only=PROBES)}
    for name, schema in trained.items():
        assert served[name]["properties"] == schema["properties"], name
        assert sorted(served[name].get("required", [])) == sorted(schema["required"]), name


def test_descriptions_match_what_the_model_is_trained_on(session: ToolSession) -> None:
    trained = {t.name: t.description for t in TOOL_DEFS if t.name in PROBES}
    assert {t.name: t.description for t in session.tools(only=PROBES)} == trained


def test_a_tool_call_returns_real_data(session: ToolSession) -> None:
    r = session.call("probe", {"path": "a.txt", "kind": "head"})
    assert r.ok
    assert r.data["lines"] == ["hello", "world"]


def test_secrets_are_refused_by_the_server(session: ToolSession) -> None:
    r = session.call("probe", {"path": ".env", "kind": "head"})
    assert not r.ok
    assert "hunter2" not in json.dumps(r.data)


def test_writes_are_refused_by_the_server(session: ToolSession, tmp_path: Path) -> None:
    r = session.call("dry_run", {"command": "touch pwned"})
    assert not r.ok
    assert not (tmp_path / "pwned").exists()


def test_a_dry_run_passes_through_the_sandbox(session: ToolSession) -> None:
    r = session.call("dry_run", {"command": "wc -l a.txt"})
    assert r.ok
    assert r.data["exit_code"] == 0


def test_bad_arguments_come_back_as_a_clean_error(session: ToolSession) -> None:
    r = session.call("probe", {"path": 5, "kind": "nope"})
    assert not r.ok
    assert r.data["error"].startswith("bad arguments")
    assert len(r.data["error"]) < 200


def test_unknown_tools_come_back_as_a_clean_error(session: ToolSession) -> None:
    r = session.call("delete_everything", {})
    assert not r.ok
    assert "unknown tool" in r.data["error"]


class Scripted:
    def __init__(self, *calls: tuple[str, dict]) -> None:
        self.calls = list(calls)

    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        name, args = self.calls.pop(0)
        return parse_qwen("<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>")


def test_the_model_sees_probes_and_decisions_only(tmp_path: Path) -> None:
    env = ProductEnv("x", tmp_path, "zsh")
    start = env.reset()
    env.close()
    assert {t.name for t in start.tools} == PROBES | {"aggregate"} | DECISION_NAMES


def test_a_whole_episode_runs_over_mcp(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x\n")
    env = ProductEnv("list python files", tmp_path, "zsh")
    policy = Scripted(("inspect_context", {}), ("dry_run", {"command": "ls *.py"}),
                      ("stage", {"command": "ls *.py", "explanation": "x"}))  # fmt: skip
    run_episode(env, policy, budget=6)
    env.close()
    assert env.decision == "stage"
    assert env.verified


def test_the_agent_reaches_tools_only_through_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the agent called the toolbox directly, replacing the session's tools wouldn't change what it sees."""

    class Fake:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def tools(self, only: set[str] | None = None) -> list[ToolDef]:
            return [ToolDef("dry_run", "d", {"type": "object", "properties": {"command": {"type": "string"}}})]

        def call(self, name: str, args: dict) -> object:
            from tth.core.types import ToolResult

            self.seen.append(name)
            return ToolResult({"ran": args["command"], "exit_code": 0, "lines": 1, "sample": ["faked"]})

        def close(self) -> None:
            pass

    fake = Fake()
    env = ProductEnv("x", tmp_path, "zsh", tools=fake)  # type: ignore[arg-type]
    start = env.reset()
    assert [t.name for t in start.tools if t.name not in DECISION_NAMES] == ["dry_run"]
    out = env.step(parse_qwen('{"name": "dry_run", "arguments": {"command": "ls"}}').tool_calls[0])
    assert out.result["sample"] == ["faked"]
    assert fake.seen == ["dry_run"]


def test_agent_and_cli_source_never_import_the_toolbox() -> None:
    for folder in ("agent", "cli", "ui"):
        for f in (ROOT / "packages" / "tth" / "src" / "tth" / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(f.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "tth.mcp.toolbox", f
