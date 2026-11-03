"""A failed run must explain itself. Missing raw output is what hid the v0.1 training bug."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tth.agent.loop import run_episode
from tth.agent.product import ProductEnv
from tth.cli import main as cli
from tth.core.types import ModelTurn, ToolCall, ToolDef
from tth.inference.toolcalls import parse_qwen
from tth.trace.schema import TraceEvent, TraceLogger, read_trace
from tth.ui.terminal import Outcome, Style, describe


class Scripted:
    """A policy that replays fixed model output, with no model."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        return parse_qwen(self.texts.pop(0))


def call(name: str, **args: object) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


@pytest.fixture
def traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return tmp_path / "state" / "tth" / "traces.jsonl"


def run(tmp_path: Path, *texts: str) -> list[TraceEvent]:
    path = tmp_path / "t.jsonl"
    env = ProductEnv("count files", tmp_path, "zsh")
    run_episode(env, Scripted(*texts), budget=4, logger=TraceLogger(path))
    return read_trace(path)


def test_raw_model_output_is_traced_for_a_call(tmp_path: Path) -> None:
    text = call("stage", command="ls", explanation="x")
    events = run(tmp_path, text)
    assert any(e.raw == text for e in events)


def test_raw_model_output_is_traced_when_nothing_parses(tmp_path: Path) -> None:
    """The exact failure of the v0.1 trained arms: an end-of-turn token and nothing else."""
    events = run(tmp_path, "<|im_end|>", "<|im_end|>")
    assert [e.raw for e in events if e.raw] == ["<|im_end|>", "<|im_end|>"]


def test_episode_starts_with_its_request_and_tools(tmp_path: Path) -> None:
    start = run(tmp_path, call("stage", command="ls"))[0]
    assert start.meta["kind"] == "start"
    assert start.meta["request"] == "count files"
    assert "stage" in start.meta["tools"]
    assert start.meta["system_hash"]


def test_badge_never_claims_verification_that_did_not_happen() -> None:
    assert Outcome("stage", "ls", "read-only").badge == "read-only · local · not verified"
    assert Outcome("stage", "ls", "read-only", verified=True).badge == "read-only · local · verified"
    assert Outcome("stage", "x", "changes-files", origin="expert").badge == "changes-files · expert · not verified"


def _ev(tool: str | None, args: dict | None = None, result: object = None, ok: bool = True) -> TraceEvent:
    from tth.trace.schema import Call

    return TraceEvent(episode="e", domain="shell", step=0, call=Call(tool=tool, args=args or {}) if tool else None,
                      result=result, ok=ok)  # fmt: skip


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (_ev("inspect_context"), "looking at the folder"),
        (_ev("probe", {"path": "a.csv", "kind": "csv_header"}, {"columns": ["x", "y"]}), "columns: x, y"),
        (_ev("probe", {"path": "a.log"}, {"error": "no such file"}, ok=False), "no such file"),
        (_ev("dry_run", {"command": "wc -l f"}, {"lines": 1, "sample": ["3"]}), "tested wc -l f (1 line: 3)"),
        (_ev("dry_run", {"command": "bad"}, {"error": "boom"}, ok=False), "tested bad: boom"),
        (_ev("stage", {"command": "rm x"}, {"notes": ["refused: no"]}, ok=False), "safety layer refused"),
        (_ev(None, result=""), "no usable tool call"),
    ],
)
def test_events_read_as_plain_sentences(event: TraceEvent, expected: str) -> None:
    line = describe(event, Style(enabled=False))
    assert line is not None and expected in line


def test_style_can_be_disabled_for_pipes() -> None:
    line = describe(_ev("inspect_context"), Style(enabled=False))
    assert line is not None and "\033" not in line


def _patch_policy(monkeypatch: pytest.MonkeyPatch, *texts: str) -> None:
    monkeypatch.setattr(cli, "_make_policy", lambda args, config: Scripted(*texts))


def test_run_writes_the_command_and_never_runs_it(
    tmp_path: Path, traces: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.txt"
    marker = tmp_path / "was_run"
    _patch_policy(monkeypatch, call("inspect_context"), call("stage", command=f"touch {marker}", explanation="x"))
    assert cli.main(["run", "--out", str(out), "--", "make", "a", "file"]) == 0
    assert out.read_text() == f"touch {marker}"
    assert not marker.exists()
    assert "changes-files · local · not verified" in capsys.readouterr().err


def test_json_mode_reports_the_outcome(
    tmp_path: Path, traces: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x\n")
    _patch_policy(monkeypatch, call("stage", command="ls", explanation="x"))
    assert cli.main(["run", "--json", "--", "list"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "stage"
    assert result["command"] == "ls"
    assert result["verified"] is True


def test_why_explains_the_last_run(
    tmp_path: Path, traces: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_policy(monkeypatch, "<|im_end|>", call("stage", command="ls"))
    cli.main(["run", "--json", "--", "list", "files"])
    capsys.readouterr()
    assert cli.main(["why"]) == 0
    text = capsys.readouterr().err
    assert "list files" in text
    assert "no usable tool call" in text


def test_why_with_raw_shows_exact_model_output(
    tmp_path: Path, traces: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_policy(monkeypatch, "<|im_end|>", call("stage", command="ls"))
    cli.main(["run", "--json", "--", "list"])
    capsys.readouterr()
    assert cli.main(["why", "--raw"]) == 0
    assert "<|im_end|>" in capsys.readouterr().err


def test_why_with_no_history_says_so(traces: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["why"]) == 1
    assert "No runs yet" in capsys.readouterr().err


def test_a_request_can_start_with_a_subcommand_word(
    tmp_path: Path, traces: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`tth how do I init git` must mean a request, not the init subcommand."""
    monkeypatch.chdir(tmp_path)
    _patch_policy(monkeypatch, call("inspect_context"), call("stage", command="git init"))
    assert cli.main(["--json", "how", "do", "I", "init", "git"]) == 0
    assert json.loads(capsys.readouterr().out)["command"] == "git init"


def test_scripted_tool_call_helper_matches_the_parser() -> None:
    turn = parse_qwen(call("dry_run", command="ls"))
    assert turn.tool_calls == [ToolCall(name="dry_run", args={"command": "ls"})]


def test_library_logging_stays_out_of_the_users_terminal(
    tmp_path: Path, traces: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import logging

    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x\n")
    _patch_policy(monkeypatch, call("stage", command="ls"))
    try:
        cli.main(["--json", "--", "list"])
        logging.getLogger("httpx").warning("HTTP Request: noise")
        assert "noise" not in capsys.readouterr().err
    finally:
        logging.disable(logging.NOTSET)
