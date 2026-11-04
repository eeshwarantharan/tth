"""Run a GGUF model through a warm local llama-server.

The server stays loaded between requests, so `t` doesn't reload the model every time, and it unloads the model after
10 idle minutes. The tool schemas from MCP can become llama.cpp's decoding grammar through `tool_choice: required`.
That's off by default: we measured the same accuracy at ten times the time (docs/ROADMAP.md, rung L1).

The process started here is a model server. It never receives or runs the user's command.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tth.core.types import RequestTooLong, ToolDef
from tth.inference.openai_compat import OpenAIPolicy, to_openai_messages

DEFAULT_PORT = 8791
# Sized for the 1,000 MB target: evals/diagnostics/results/memory_candidates.jsonl
CONTEXT_TOKENS = 2048
UBATCH_TOKENS = 128
CHECKPOINTS = 0
REPLY_TOKENS = 256  # room kept free for the model's next reply
_REMOVED = "[removed to fit the model's memory]"
IDLE_SECONDS = 600
START_TIMEOUT = 90.0


class ServerUnavailable(Exception):
    """llama-server is missing or wouldn't start. The message says what to do."""


def trim_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """One step smaller: remove the oldest tool result, then keep halving the newest. None once nothing can go.

    This only changes what the model reads. The verification gate works from the real results, so it can't be fooled.
    """
    results = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in results[:-1]:
        if messages[i]["content"] != _REMOVED:
            return [*messages[:i], {**messages[i], "content": _REMOVED}, *messages[i + 1 :]]
    if not results or len(messages[results[-1]]["content"]) <= 200:
        return None
    i = results[-1]
    kept = messages[i]["content"][: len(messages[i]["content"]) // 2]
    return [*messages[:i], {**messages[i], "content": f"{kept} [cut to fit the model's memory]"}, *messages[i + 1 :]]


class LlamaPolicy(OpenAIPolicy):
    """A local model has a small context, and one large file would overflow it. So the prompt is measured first."""

    def __init__(self, port: int, context: int, **kwargs: Any) -> None:
        super().__init__(base_url=f"http://127.0.0.1:{port}/v1", **kwargs)
        self.port = port
        self.context = context

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as r:  # noqa: S310 - fixed local URL
            return json.load(r)

    def prompt_tokens(self, messages: list[dict[str, Any]], tools: list[ToolDef] | None) -> int:
        body: dict[str, Any] = {"messages": to_openai_messages(messages)}
        if tools:
            body["tools"] = [t.openai() for t in tools]
        return len(self._post("/tokenize", {"content": self._post("/apply-template", body)["prompt"]})["tokens"])

    def fit(self, messages: list[dict[str, Any]], tools: list[ToolDef] | None) -> list[dict[str, Any]]:
        room = self.context - REPLY_TOKENS
        while self.prompt_tokens(messages, tools) > room:
            smaller = trim_results(messages)
            if smaller is None:
                raise RequestTooLong("That request is too long for the local model. Try a shorter one.")
            messages = smaller
        return messages

    def complete(self, messages: list[dict[str, Any]], tools: list[ToolDef] | None = None) -> Any:
        return super().complete(self.fit(messages, tools), tools)


def server_args(
    model: Path,
    port: int,
    ctx: int = CONTEXT_TOKENS,
    ubatch: int = UBATCH_TOKENS,
    checkpoints: int = CHECKPOINTS,
    template: Path | None = None,
) -> list[str]:
    override = ["--chat-template-file", str(template)] if template else []
    return [
        "llama-server",
        "--model", str(model),
        "--host", "127.0.0.1",  # never reachable from another machine
        "--port", str(port),
        "--ctx-size", str(ctx),
        "--ubatch-size", str(ubatch),
        "--ctx-checkpoints", str(checkpoints),
        "--parallel", "1",
        "--cache-ram", "0",  # one user, one slot: the default keeps up to 8 GB of old prompts in memory
        "--jinja",
        "--no-webui",
        "--sleep-idle-seconds", str(IDLE_SECONDS),
        *override,
    ]  # fmt: skip


def is_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:  # noqa: S310 - fixed local URL
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def served_model(port: int) -> str | None:
    """The model file the server on this port has loaded."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=2) as r:  # noqa: S310 - fixed local URL
            return json.load(r).get("model_path")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def ensure_server(model: Path, log: Path, port: int = DEFAULT_PORT, **settings: Any) -> None:
    """Use the warm server if it has this model. Otherwise start one and wait until the model has loaded."""
    if is_healthy(port):
        running = served_model(port)
        if running and Path(running).resolve() != model.resolve():
            # Answering from the wrong model would be silently wrong, and that would invalidate any comparison.
            raise ServerUnavailable(f"llama-server on port {port} is running {running}. Stop it to use {model}.")
        return
    if shutil.which("llama-server") is None:
        raise ServerUnavailable("llama-server isn't installed. On macOS: brew install llama.cpp")
    log.parent.mkdir(parents=True, exist_ok=True)
    args = server_args(model, port, **settings)
    subprocess.Popen(args, stdout=log.open("a"), stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if is_healthy(port):
            return
        time.sleep(0.25)
    raise ServerUnavailable(f"llama-server didn't become ready within {START_TIMEOUT:.0f}s. See {log}")


def llama_policy(port: int = DEFAULT_PORT, grammar: bool = False, context: int = CONTEXT_TOKENS) -> LlamaPolicy:
    return LlamaPolicy(
        port, context, model="local", api_key="local", temperature=0.0, max_tokens=512, thinking=False, limiter=None,
        tool_choice="required" if grammar else None, attempts=1,
    )  # fmt: skip
