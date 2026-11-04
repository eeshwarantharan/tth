"""The llama.cpp backend. The live tests need llama-server and a GGUF, so they skip where those are missing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tth.core.config import data_dir
from tth.core.types import ToolDef
from tth.inference import llamacpp
from tth.inference.openai_compat import OpenAIPolicy

GGUF = data_dir() / "models" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
needs_model = pytest.mark.skipif(
    not (GGUF.exists() and llamacpp.shutil.which("llama-server")), reason="needs llama-server and the GGUF model"
)
TOOLS = [
    ToolDef(
        "dry_run",
        "Test a command.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    ),
    ToolDef(
        "stage",
        "Stage a command.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    ),
]


def test_the_server_only_listens_on_this_machine() -> None:
    args = llamacpp.server_args(Path("m.gguf"), 1234)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--jinja" in args
    assert "--no-webui" in args


def test_missing_llama_server_explains_how_to_get_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "is_healthy", lambda port: False)
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    with pytest.raises(llamacpp.ServerUnavailable, match="brew install llama.cpp"):
        llamacpp.ensure_server(tmp_path / "m.gguf", tmp_path / "log")


def test_a_server_running_another_model_is_never_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise a model comparison could silently test the same model every time."""
    monkeypatch.setattr(llamacpp, "is_healthy", lambda port: True)
    monkeypatch.setattr(llamacpp, "served_model", lambda port: str(tmp_path / "other.gguf"))
    with pytest.raises(llamacpp.ServerUnavailable, match="other.gguf"):
        llamacpp.ensure_server(tmp_path / "m.gguf", tmp_path / "log")


def test_a_healthy_server_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "is_healthy", lambda port: True)
    monkeypatch.setattr(llamacpp, "served_model", lambda port: str(tmp_path / "m.gguf"))
    monkeypatch.setattr(llamacpp.subprocess, "Popen", lambda *a, **k: pytest.fail("started a second server"))
    llamacpp.ensure_server(tmp_path / "m.gguf", tmp_path / "log")


def _fake_reply(content: str = "", calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))], usage=None
    )


class _Client:
    def __init__(self, reply: SimpleNamespace) -> None:
        self.kwargs: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reply = reply

    def _create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return self.reply


def _policy(reply: SimpleNamespace, **kw: object) -> tuple[OpenAIPolicy, _Client]:
    p = OpenAIPolicy(model="local", base_url="http://127.0.0.1:1/v1", api_key="x", limiter=None, **kw)  # type: ignore[arg-type]
    client = _Client(reply)
    p.client = client  # type: ignore[assignment]
    return p, client


def test_tool_schemas_become_the_grammar() -> None:
    p, client = _policy(_fake_reply(), tool_choice="required")
    p.chat([{"role": "user", "content": "x"}], TOOLS)
    assert client.kwargs["tool_choice"] == "required"
    assert [t["function"]["name"] for t in client.kwargs["tools"]] == ["dry_run", "stage"]


def test_without_a_grammar_tool_choice_is_not_sent() -> None:
    p, client = _policy(_fake_reply())
    p.chat([{"role": "user", "content": "x"}], TOOLS)
    assert "tool_choice" not in client.kwargs


def test_the_shipped_policy_has_no_grammar() -> None:
    """The bake-off ran without one, and it was ten times slower. The product has to match what was measured."""
    assert llamacpp.llama_policy().tool_choice is None
    assert llamacpp.llama_policy(grammar=True).tool_choice == "required"


def test_raw_output_is_kept_for_traces() -> None:
    call = SimpleNamespace(id="1", function=SimpleNamespace(name="stage", arguments='{"command": "ls"}'))
    p, _ = _policy(_fake_reply(calls=[call]))
    turn = p.chat([{"role": "user", "content": "x"}], TOOLS)
    assert turn.tool_calls[0].name == "stage"
    assert "stage" in turn.raw


def test_a_tool_call_written_as_text_is_still_understood() -> None:
    """Servers without a grammar can return the call as plain text."""
    text = '<tool_call>\n{"name": "stage", "arguments": {"command": "ls"}}\n</tool_call>'
    p, _ = _policy(_fake_reply(content=text))
    turn = p.chat([{"role": "user", "content": "x"}], TOOLS)
    assert turn.tool_calls[0].args == {"command": "ls"}
    assert turn.raw == text


@needs_model
def test_a_real_model_with_a_grammar_always_calls_a_real_tool(tmp_path: Path) -> None:
    llamacpp.ensure_server(GGUF, tmp_path / "log")
    policy = llamacpp.llama_policy(grammar=True)
    for request in ("count python files", "hello", "what is the meaning of life?"):
        turn = policy.chat([{"role": "user", "content": request}], TOOLS)
        assert len(turn.tool_calls) >= 1, turn.raw
        assert turn.tool_calls[0].name in {"dry_run", "stage"}
        assert isinstance(turn.tool_calls[0].args, dict)
        json.dumps(turn.tool_calls[0].args)
