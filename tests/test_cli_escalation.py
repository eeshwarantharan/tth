"""Invariant 5: nothing leaves the machine without consent, and an expert's answer is never trusted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tth.a2a import client as a2a
from tth.a2a.client import ExpertAnswer
from tth.cli import main as cli
from tth.core.types import ModelTurn, ToolDef
from tth.inference.toolcalls import parse_qwen


class GivesUp:
    """A model that produces nothing, so the run reaches the escalation prompt."""

    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        return parse_qwen("")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for variable, folder in (("XDG_STATE_HOME", "s"), ("XDG_CONFIG_HOME", "c"), ("XDG_DATA_HOME", "d")):
        monkeypatch.setenv(variable, str(tmp_path / folder))
    monkeypatch.delenv("TTH_YES", raising=False)
    monkeypatch.delenv("TTH_AUTO_ESCALATE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_policy", lambda args, config: GivesUp())
    (tmp_path / "a.txt").write_text("hello\n")
    return tmp_path


def _expert_says(monkeypatch: pytest.MonkeyPatch, command: str) -> list[str]:
    sent: list[str] = []

    def ask(text: str) -> ExpertAnswer:
        sent.append(text)
        return ExpertAnswer(command, "because", command)

    monkeypatch.setattr(a2a, "ask_expert", ask)
    monkeypatch.setattr(a2a, "expert_reachable", lambda: True)
    return sent


def test_declining_sends_nothing(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent = _expert_says(monkeypatch, "ls")
    monkeypatch.setattr(cli, "_read_tty", lambda prompt: "n")
    assert cli.main(["--json", "do", "something"]) == 1
    assert sent == []
    assert json.loads(capsys.readouterr().out)["decision"] == "declined"


def test_no_terminal_means_no(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _expert_says(monkeypatch, "ls")
    monkeypatch.setattr(cli, "_read_tty", lambda prompt: "")
    assert cli.main(["--json", "do", "something"]) == 1
    assert sent == []


def test_the_payload_is_shown_before_it_is_sent(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent = _expert_says(monkeypatch, "wc -l a.txt")
    monkeypatch.setattr(cli, "_read_tty", lambda prompt: "y")
    assert cli.main(["--json", "count", "lines", "in", "a.txt"]) == 0
    assert sent and sent[0].splitlines()[0] in capsys.readouterr().err


def test_a_good_expert_answer_is_tested_locally_and_badged(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _expert_says(monkeypatch, "wc -l a.txt")
    monkeypatch.setattr(cli, "_read_tty", lambda prompt: "y")
    assert cli.main(["--json", "count", "lines"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["origin"] == "expert"
    assert result["verified"] is True
    assert result["badge"] == "read-only · expert · verified"


def test_a_dangerous_expert_answer_is_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _expert_says(monkeypatch, "rm -rf ~")
    monkeypatch.setattr(cli, "_read_tty", lambda prompt: "y")
    assert cli.main(["--json", "clean", "up"]) == 1
    assert json.loads(capsys.readouterr().out)["command"] is None
    assert (project / "a.txt").exists()
