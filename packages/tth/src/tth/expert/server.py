"""The reference A2A expert: a task-based agent with a pluggable model.

It receives text the user has already approved, and either asks for one more read-only result or returns a typed
answer as an artifact. tth checks that answer before staging it, and never executes it.

Works with any OpenAI-compatible endpoint: Ollama, llama-server, vLLM or a hosted API. Its Agent Card names where
the text goes next, so the consent prompt can tell the user. Listening beyond this machine requires a token.

Run: tth expert
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from a2a.helpers.proto_helpers import get_message_text, new_data_part, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill

from tth.a2a.client import HOME_URL, REFERENCE_ORG
from tth.core.config import load_config, load_user_env

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
SYSTEM = (
    "You are an expert shell assistant helping a small local agent. You get a user request and what the local agent "
    "observed. Reply with JSON only, in one of two forms.\n"
    'To answer: {"command": "<one command for the user\'s OS and shell>", "explanation": "<one sentence>"}\n'
    'To see one more read-only result first: {"need": {"tool": "probe", "args": {"path": "<file>", "kind": "head"}, '
    '"why": "<one sentence>"}}\n'
    'If it can\'t be done safely, set command to "" and explain. Prefer read-only commands. Never use sudo. Never '
    "delete anything the user didn't ask to delete."
)


class Brain(Protocol):
    def think(self, conversation: list[str]) -> dict[str, Any]: ...


class LLMBrain:
    """Any OpenAI-compatible model."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        from tth.inference.openai_compat import OpenAIPolicy

        self.policy = OpenAIPolicy(model=model, base_url=base_url, api_key=api_key or "none", temperature=0.2)

    def think(self, conversation: list[str]) -> dict[str, Any]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(conversation)}]
        text = (self.policy.complete(messages).choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {"command": "", "explanation": text[:300]}
        except json.JSONDecodeError:
            return {"command": "", "explanation": text[:300]}


class ExpertExecutor(AgentExecutor):
    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.conversations: dict[str, list[str]] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if context.current_task is None and context.message is not None:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        await updater.start_work(updater.new_agent_message([new_text_part("Working on it.")]))
        conversation = self.conversations.setdefault(context.task_id, [])
        conversation.append(get_message_text(context.message) if context.message else "")
        try:
            result = await asyncio.to_thread(self.brain.think, conversation)
        except Exception as e:  # noqa: BLE001 - a failed task is how A2A reports this to the client
            await updater.failed(updater.new_agent_message([new_text_part(f"The expert's model failed: {e}")]))
            self.conversations.pop(context.task_id, None)
            return
        if isinstance(result.get("need"), dict):
            await updater.requires_input(updater.new_agent_message([new_data_part(result)]))
            return
        await updater.add_artifact([new_data_part(result)], name="shell-command")
        await updater.complete()
        self.conversations.pop(context.task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.task_id:
            self.conversations.pop(context.task_id, None)
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.cancel()


def build_card(url: str, forwards_to: str = "", name: str = "tth-expert") -> AgentCard:
    return AgentCard(
        name=name,
        description="Expert shell agent for tth escalations.",
        version="0.2.0",
        provider=AgentProvider(organization=REFERENCE_ORG, url=forwards_to or HOME_URL),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json", "text/plain"],
        skills=[
            AgentSkill(id="shell-command", name="Shell command", description="Turn a request into one checked command",
                       tags=["shell"]),
        ],
    )  # fmt: skip


def build_app(url: str, brain: Brain, forwards_to: str = "", token: str | None = None, name: str = "tth-expert"):
    from starlette.applications import Starlette
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    card = build_card(url, forwards_to, name)
    handler = DefaultRequestHandler(
        agent_executor=ExpertExecutor(brain), task_store=InMemoryTaskStore(), agent_card=card
    )
    app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/"))
    app.state.last_request = time.monotonic()

    async def guard(request: Request, call_next: Any) -> Any:
        app.state.last_request = time.monotonic()
        is_card = request.url.path.startswith("/.well-known/")
        supplied = request.headers.get("authorization", "")
        if token and not is_card and not secrets.compare_digest(supplied, f"Bearer {token}"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    app.add_middleware(BaseHTTPMiddleware, dispatch=guard)
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    load_user_env()
    config = load_config()
    parsed = urlparse(config.expert_url)
    ap = argparse.ArgumentParser(prog="tth expert", description="Run the reference A2A expert agent.")
    ap.add_argument("--host", default=parsed.hostname or "127.0.0.1")
    ap.add_argument("--port", type=int, default=parsed.port or 8765)
    ap.add_argument("--idle-exit", type=int, default=0, help="stop after this many idle seconds (0: keep running)")
    args = ap.parse_args(argv)

    token = os.environ.get("TTH_EXPERT_TOKEN")
    if args.host not in LOOPBACK and not token:
        raise SystemExit("Listening beyond this machine needs a token. Set TTH_EXPERT_TOKEN and try again.")
    api_key = os.environ.get("TTH_EXPERT_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
    brain = LLMBrain(config.expert_base_url, config.expert_model, api_key)
    url = f"http://{args.host}:{args.port}/"
    forwards_to = "" if urlparse(config.expert_base_url).hostname in LOOPBACK else config.expert_base_url
    app = build_app(url, brain, forwards_to, token)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning"))

    if args.idle_exit:

        async def watch() -> None:
            while not server.should_exit:
                await asyncio.sleep(5)
                if time.monotonic() - app.state.last_request > args.idle_exit:
                    server.should_exit = True

        async def serve() -> None:
            await asyncio.gather(server.serve(), watch())

        asyncio.run(serve())
    else:
        server.run()


if __name__ == "__main__":
    main()
