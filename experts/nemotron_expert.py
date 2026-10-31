"""A2A expert agent backed by Nemotron on the NVIDIA API.

Receives an already-redacted, user-approved payload and answers with one shell
command as JSON. tth classifies and rewrites the answer before staging it;
the expert's command is never executed by tth.

Run: uv run python -m experts.nemotron_expert  (serves on 127.0.0.1:8765)
"""

from __future__ import annotations

import asyncio
import os

from a2a.helpers.proto_helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

from tth.runtime.openai_backend import OpenAIPolicy

SYSTEM = (
    "You are an expert shell assistant helping a small local agent. You get a user request and what the local "
    'agent observed. Answer with JSON only: {"command": "...", "explanation": "..."}. One command for the '
    "user's OS and shell. Prefer read-only commands; never use sudo; never delete without the user asking."
)


class NemotronExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.policy = OpenAIPolicy(temperature=0.2, thinking=True)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = get_message_text(context.message) if context.message else ""
        r = await asyncio.to_thread(
            self.policy.complete,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        )
        answer = (r.choices[0].message.content or "").strip()
        await event_queue.enqueue_event(new_text_message(answer, context_id=context.context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None


def build_app(url: str):
    from starlette.applications import Starlette

    card = AgentCard(
        name="tth-expert-nemotron",
        description="Expert shell agent (Nemotron) for tth escalations.",
        version="0.1.0",
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="shell-command",
                name="Shell command",
                description="Turn a request into one command",
                tags=["shell"],
            )
        ],  # fmt: skip
    )
    handler = DefaultRequestHandler(agent_executor=NemotronExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/")
    return Starlette(routes=routes)


def main() -> None:
    import uvicorn

    host, port = "127.0.0.1", int(os.environ.get("TTH_EXPERT_PORT", "8765"))
    uvicorn.run(build_app(f"http://{host}:{port}/"), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
