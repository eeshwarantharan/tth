"""Run a GGUF model through a warm local llama-server.

The server stays loaded between requests, so `t` doesn't reload the model every time, and it unloads the model after
10 idle minutes. The tool schemas from MCP become llama.cpp's decoding grammar through `tool_choice: required`, so
every reply is a valid call to a real tool.

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

from tth.inference.openai_compat import OpenAIPolicy

DEFAULT_PORT = 8791
CONTEXT_TOKENS = 8192
IDLE_SECONDS = 600
START_TIMEOUT = 90.0


class ServerUnavailable(Exception):
    """llama-server is missing or wouldn't start. The message says what to do."""


def server_args(model: Path, port: int) -> list[str]:
    return [
        "llama-server",
        "--model", str(model),
        "--host", "127.0.0.1",  # never reachable from another machine
        "--port", str(port),
        "--ctx-size", str(CONTEXT_TOKENS),
        "--parallel", "1",
        "--jinja",
        "--no-webui",
        "--sleep-idle-seconds", str(IDLE_SECONDS),
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


def ensure_server(model: Path, log: Path, port: int = DEFAULT_PORT) -> None:
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
    subprocess.Popen(server_args(model, port), stdout=log.open("a"), stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if is_healthy(port):
            return
        time.sleep(0.25)
    raise ServerUnavailable(f"llama-server didn't become ready within {START_TIMEOUT:.0f}s. See {log}")


def llama_policy(port: int = DEFAULT_PORT, grammar: bool = True) -> OpenAIPolicy:
    return OpenAIPolicy(
        model="local", base_url=f"http://127.0.0.1:{port}/v1", api_key="local", temperature=0.0, max_tokens=512,
        thinking=False, limiter=None, tool_choice="required" if grammar else None, attempts=1,
    )  # fmt: skip
