"""The product must not claim more than happened."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from tth.agent.product import ProductEnv
from tth.core.types import ToolCall
from tth.mcp.toolbox import ShellToolbox
from tth.trace.schema import TraceLogger


@pytest.fixture
def env(tmp_path: Path):
    e = ProductEnv("list files", tmp_path, "zsh")
    e.reset()
    yield e
    e.close()


def test_a_command_that_cannot_be_tested_is_not_labelled_verified(env: ProductEnv) -> None:
    env.step(ToolCall(name="inspect_context", args={}))
    env.step(ToolCall(name="stage", args={"command": "mkdir out", "explanation": "x"}))
    assert env.decision == "stage"
    assert not env.verified


def test_a_command_that_passed_its_test_is_verified(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    e = ProductEnv("list files", tmp_path, "zsh")
    e.reset()
    e.step(ToolCall(name="stage", args={"command": "ls", "explanation": "x"}))
    e.close()
    assert e.decision == "stage"
    assert e.verified


def test_non_string_command_is_a_clean_error(tmp_path: Path) -> None:
    box = ShellToolbox(tmp_path, shell="zsh")
    for tool in ("stage", "dry_run"):
        args = {"command": 5, "explanation": "x"} if tool == "stage" else {"command": 5}
        r = box.call(tool, args)
        assert not r.ok
        assert "must be strings" in r.data["error"]


def test_traces_are_readable_only_by_the_user(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("")
    path.chmod(0o644)
    TraceLogger(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
