"""The llama.cpp backend. The live tests need llama-server and a GGUF, so they skip where those are missing."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tth.core.config import DEFAULT_GGUF, data_dir
from tth.core.types import RequestTooLong, ToolDef
from tth.inference import llamacpp
from tth.inference.openai_compat import OpenAIPolicy

GGUF = data_dir() / "models" / DEFAULT_GGUF
TEST_PORT = 8796  # never the user's warm server
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


def _tool(content: str, call_id: str = "1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": "dry_run", "content": content}


def test_trimming_removes_the_oldest_result_first_and_leaves_the_original_alone() -> None:
    messages = [{"role": "user", "content": "x"}, _tool("old" * 200, "1"), _tool("new" * 200, "2")]
    trimmed = llamacpp.trim_results(messages)
    assert trimmed is not None
    assert "removed" in trimmed[1]["content"]
    assert trimmed[2]["content"] == "new" * 200
    assert messages[1]["content"] == "old" * 200


def test_trimming_cuts_the_newest_result_last_and_then_stops() -> None:
    messages = [{"role": "user", "content": "x"}, _tool("a" * 1000)]
    seen = []
    while (messages := llamacpp.trim_results(messages)) is not None:
        seen.append(messages[1]["content"])
    assert seen[0].startswith("a" * 500) and "cut to fit" in seen[0]
    assert len(seen[-1]) <= 300


class _CountingPolicy(llamacpp.LlamaPolicy):
    def prompt_tokens(self, messages: list[dict], tools: object) -> int:
        return sum(len(m["content"]) for m in messages)


def test_a_prompt_that_fits_is_left_exactly_as_it_was() -> None:
    policy = _CountingPolicy(1, 2048, model="local", api_key="x", limiter=None)
    messages = [{"role": "user", "content": "x"}, _tool("short")]
    assert policy.fit(messages, None) is messages


def test_a_huge_tool_result_is_cut_to_leave_room_for_the_reply() -> None:
    policy = _CountingPolicy(1, 2048, model="local", api_key="x", limiter=None)
    fitted = policy.fit([{"role": "user", "content": "x"}, _tool("a" * 8000)], None)
    assert sum(len(m["content"]) for m in fitted) <= 2048 - llamacpp.REPLY_TOKENS


def test_a_request_too_long_for_any_prompt_says_so() -> None:
    policy = _CountingPolicy(1, 2048, model="local", api_key="x", limiter=None)
    with pytest.raises(RequestTooLong):
        policy.fit([{"role": "user", "content": "x" * 5000}], None)


@pytest.fixture
def real_server(tmp_path: Path) -> Iterator[None]:
    """Its own server on its own port, so tests can't touch the warm one or leave one running."""
    with (tmp_path / "log").open("w") as log:
        process = subprocess.Popen(llamacpp.server_args(GGUF, TEST_PORT), stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + llamacpp.START_TIMEOUT
        while not llamacpp.is_healthy(TEST_PORT) and time.monotonic() < deadline:
            time.sleep(0.25)
        yield
    finally:
        process.terminate()
        process.wait()


@needs_model
def test_a_real_server_survives_a_tool_result_larger_than_its_context(real_server: None) -> None:
    policy = llamacpp.llama_policy(port=TEST_PORT)
    messages = [{"role": "user", "content": "count python files"}, _tool("x y z\n" * 1400)]
    assert policy.prompt_tokens(messages, TOOLS) > llamacpp.CONTEXT_TOKENS
    policy.chat(messages, TOOLS)


@needs_model
@pytest.mark.xfail(reason="llama-server doesn't enforce the tool schemas on this build, so rung L1 needs a rerun")
def test_a_real_model_with_a_grammar_always_calls_a_real_tool(real_server: None) -> None:
    policy = llamacpp.llama_policy(port=TEST_PORT, grammar=True)
    for request in ("count python files", "hello", "what is the meaning of life?"):
        turn = policy.chat([{"role": "user", "content": request}], TOOLS)
        assert len(turn.tool_calls) >= 1, turn.raw
        assert turn.tool_calls[0].name in {"dry_run", "stage"}
        assert isinstance(turn.tool_calls[0].args, dict)
        json.dumps(turn.tool_calls[0].args)
