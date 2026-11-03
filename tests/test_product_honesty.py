"""The product must not claim more than happened."""

from __future__ import annotations

import stat
from pathlib import Path

from tth.agent.product import ProductEnv
from tth.agent.protocol import ToolCall
from tth.tools.shell_tools import ShellToolbox
from tth.trace.schema import TraceLogger


def _env(tmp_path: Path) -> ProductEnv:
    env = ProductEnv("list files", tmp_path, "zsh")
    env.reset()
    return env


def test_unverified_command_is_not_labelled_verified(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.step(ToolCall(name="stage", args={"command": "ls -la", "explanation": "x"}))
    assert env.decision == "stage"
    assert not env.verified


def test_command_that_passed_a_dry_run_is_verified(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.last_verified = "ls -la"
    env.step(ToolCall(name="stage", args={"command": "ls -la", "explanation": "x"}))
    assert env.verified


def test_a_different_command_than_the_one_verified_is_not_verified(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.last_verified = "ls"
    env.step(ToolCall(name="stage", args={"command": "ls -la", "explanation": "x"}))
    assert not env.verified


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
