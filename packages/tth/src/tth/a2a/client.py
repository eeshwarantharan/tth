"""A2A client: send an approved payload to the expert agent and parse its answer."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from tth.core.config import load_config

DEFAULT_EXPERT_URL = load_config().expert_url


@dataclass
class ExpertAnswer:
    command: str
    explanation: str
    raw: str


def parse_answer(text: str) -> ExpertAnswer:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return ExpertAnswer(str(obj.get("command", "")).strip(), str(obj.get("explanation", "")).strip(), text)
        except json.JSONDecodeError:
            pass
    fence = re.search(r"```(?:\w+)?\n(.*?)```", text, re.S)
    cmd = fence.group(1).strip().splitlines()[0] if fence else ""
    return ExpertAnswer(cmd, text.strip()[:300], text)


async def _send(url: str, text: str, timeout: float) -> str:
    import httpx
    from a2a.client import ClientConfig, create_client
    from a2a.helpers.proto_helpers import get_message_text, new_text_message
    from a2a.types.a2a_pb2 import Role, SendMessageRequest

    async with httpx.AsyncClient(timeout=timeout) as http:
        client = await create_client(url, client_config=ClientConfig(streaming=False, httpx_client=http))
        req = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))
        out: list[str] = []
        async for resp in client.send_message(req):
            if resp.HasField("message"):
                out.append(get_message_text(resp.message))
            elif resp.HasField("task"):
                for a in resp.task.artifacts:
                    out.extend(p.text for p in a.parts if p.text)
                if resp.task.status.HasField("message"):
                    out.append(get_message_text(resp.task.status.message))
        return "\n".join(x for x in out if x)


def ask_expert(text: str, url: str = DEFAULT_EXPERT_URL, timeout: float = 120.0) -> ExpertAnswer:
    return parse_answer(asyncio.run(_send(url, text, timeout)))


def expert_reachable(url: str = DEFAULT_EXPERT_URL) -> bool:
    import httpx

    try:
        return httpx.get(url.rstrip("/") + "/.well-known/agent-card.json", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False
