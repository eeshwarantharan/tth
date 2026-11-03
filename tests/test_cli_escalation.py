"""Consent as the user meets it. There's no switch that skips it for a real expert."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tth.a2a.client import EvidenceRequest, ExpertAnswer, ExpertCard
from tth.cli import main as cli
from tth.core.types import ModelTurn, ToolDef
from tth.inference.toolcalls import parse_qwen


class GivesUp:
    """A model that produces nothing, so the run reaches the escalation prompt."""

    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        return parse_qwen("")


class FakeExpert:
    def __init__(self, command: str = "wc -l a.txt", forwards_to: str = "https://api.example.com/v1") -> None:
        self.command = command
        self.forwards_to = forwards_to
        self.sent: list[str] = []

    def card(self) -> ExpertCard:
        return ExpertCard("tth-expert", "http://127.0.0.1:8765/", forwards_to=self.forwards_to)

    def ask(self, text: str, on_evidence: object, on_status: object = None) -> ExpertAnswer:
        self.sent.append(text)
        return ExpertAnswer(self.command, "because")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for variable, folder in (("XDG_STATE_HOME", "s"), ("XDG_CONFIG_HOME", "c"), ("XDG_DATA_HOME", "d")):
        monkeypatch.setenv(variable, str(tmp_path / folder))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_policy", lambda args, config: GivesUp())
    (tmp_path / "a.txt").write_text("hello\n")
    return tmp_path


def _use(monkeypatch: pytest.MonkeyPatch, expert: FakeExpert, answer: str) -> None:
    monkeypatch.setattr(cli, "ExpertClient", lambda: expert)
    monkeypatch.setattr(cli, "_read_tty", lambda prompt: answer)


def test_declining_sends_nothing(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expert = FakeExpert()
    _use(monkeypatch, expert, "n")
    assert cli.main(["--json", "do", "something"]) == 1
    assert expert.sent == []
    assert json.loads(capsys.readouterr().out)["decision"] == "declined"


def test_no_terminal_means_no(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expert = FakeExpert()
    _use(monkeypatch, expert, "")
    assert cli.main(["--json", "do", "something"]) == 1
    assert expert.sent == []


@pytest.mark.parametrize("variable", ["TTH_YES", "TTH_AUTO_ESCALATE"])
def test_old_environment_switches_no_longer_skip_consent(
    project: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "1")
    expert = FakeExpert()
    _use(monkeypatch, expert, "")
    assert cli.main(["--json", "do", "something"]) == 1
    assert expert.sent == []


def test_yes_flag_does_not_exist(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--yes", "--", "do", "something"])


def test_the_prompt_shows_the_exact_text_and_the_destination(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expert = FakeExpert()
    _use(monkeypatch, expert, "y")
    assert cli.main(["--json", "count", "lines", "in", "a.txt"]) == 0
    shown = capsys.readouterr().err
    assert "https://api.example.com/v1" in shown
    for line in expert.sent[0].splitlines():
        assert line in shown


def test_a_good_answer_gets_an_honest_badge(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use(monkeypatch, FakeExpert(), "y")
    assert cli.main(["--json", "count", "lines"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["origin"] == "expert"
    assert result["badge"] == "read-only · expert · verified"


def test_a_dangerous_answer_is_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use(monkeypatch, FakeExpert("rm -rf ~"), "y")
    assert cli.main(["--json", "clean", "up"]) == 1
    assert json.loads(capsys.readouterr().out)["command"] is None
    assert (project / "a.txt").exists()


def test_terminal_control_characters_from_an_expert_are_neutralised(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hostile expert could otherwise rewrite the screen, or hide what it's asking for."""
    expert = FakeExpert()
    monkeypatch.setattr(expert, "ask", lambda *a, **k: ExpertAnswer("wc -l a.txt", "\x1b[2Jgotcha"))
    _use(monkeypatch, expert, "y")
    cli.main(["--json", "count", "lines"])
    assert "\x1b[2J" not in capsys.readouterr().err


def test_a_staged_command_with_control_characters_is_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use(monkeypatch, FakeExpert("wc -l a.txt\x1b[2K"), "y")
    assert cli.main(["--json", "count", "lines"]) == 1
    assert json.loads(capsys.readouterr().out)["command"] is None


def test_evidence_request_type_is_importable() -> None:
    assert EvidenceRequest("probe").tool == "probe"
