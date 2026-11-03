"""A2A client: find an expert by its Agent Card, send it a task, and follow the task to its result.

Any A2A agent can be an expert. Typed answers come back as a data artifact, and plain text from other agents is
parsed as a fallback. Only tth.agent.escalation may use this, because that's where consent is enforced.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from tth.core.config import load_config

MAX_FOLLOW_UPS = 2
REFERENCE_ORG = "tth"
HOME_URL = "https://github.com/eeshwarantharan/tth"
DEFAULT_TIMEOUT = 120.0


@dataclass
class ExpertCard:
    """Who would receive the data. Shown to the user before anything is sent."""

    name: str
    url: str
    forwards_to: str = ""
    reference: bool = False  # tth's own expert, whose card says where it forwards to

    @property
    def is_local(self) -> bool:
        """We can only vouch for our own expert. Any other agent might pass the text on."""
        loopback = urlparse(self.url).hostname in ("127.0.0.1", "localhost", "::1")
        return loopback and self.reference and not self.forwards_to

    def destination(self) -> str:
        where = f"{self.name} at {self.url}"
        return f"{where}, which sends it on to {self.forwards_to}" if self.forwards_to else where


@dataclass
class EvidenceRequest:
    """The expert asks to see one more read-only result before it answers."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""


@dataclass
class ExpertAnswer:
    command: str
    explanation: str
    raw: str = ""


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


def _evidence_request(data: Any) -> EvidenceRequest | None:
    need = data.get("need") if isinstance(data, dict) else None
    if not isinstance(need, dict) or not isinstance(need.get("tool"), str):
        return None
    args = need.get("args")
    return EvidenceRequest(need["tool"], args if isinstance(args, dict) else {}, str(need.get("why", "")))


class ExpertClient:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.url = url or load_config().expert_url
        self.token = token if token is not None else os.environ.get("TTH_EXPERT_TOKEN")
        self.timeout = timeout

    def _http(self, timeout: float):
        import httpx

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        return httpx.AsyncClient(timeout=timeout, headers=headers)

    def card(self) -> ExpertCard | None:
        """The expert's Agent Card, or None if nothing is answering. Sends no user data."""
        import httpx

        try:
            return asyncio.run(self._card())
        except (httpx.HTTPError, OSError, ValueError):
            return None
        except Exception as e:  # noqa: BLE001 - the SDK wraps card failures in its own error types
            if type(e).__module__.startswith("a2a"):
                return None
            raise

    async def _card(self) -> ExpertCard:
        from a2a.client import A2ACardResolver

        async with self._http(3.0) as http:
            card = await A2ACardResolver(http, self.url).get_agent_card()
        reference = card.provider.organization == REFERENCE_ORG
        forwards_to = card.provider.url if reference and card.provider.url != HOME_URL else ""
        return ExpertCard(name=card.name, url=self.url, forwards_to=forwards_to, reference=reference)

    def ask(
        self,
        text: str,
        on_evidence: Callable[[EvidenceRequest], str | None],
        on_status: Callable[[str], None] | None = None,
    ) -> ExpertAnswer:
        """Send a task and see it through. on_evidence returns text to send, or None to stop."""
        return asyncio.run(self._ask(text, on_evidence, on_status or (lambda _: None)))

    async def _ask(
        self, text: str, on_evidence: Callable[[EvidenceRequest], str | None], on_status: Callable[[str], None]
    ) -> ExpertAnswer:
        from a2a.client import ClientConfig, create_client
        from a2a.helpers.proto_helpers import new_text_message
        from a2a.types.a2a_pb2 import Role, SendMessageRequest

        async with self._http(self.timeout) as http:
            client = await create_client(self.url, client_config=ClientConfig(streaming=True, httpx_client=http))
            task_id, context_id = "", ""
            for _ in range(MAX_FOLLOW_UPS + 1):
                message = new_text_message(text, role=Role.ROLE_USER)
                if task_id:
                    message.task_id, message.context_id = task_id, context_id
                turn = _Turn()
                async for response in client.send_message(SendMessageRequest(message=message)):
                    turn.read(response, on_status)
                task_id, context_id = turn.task_id or task_id, turn.context_id or context_id
                if turn.evidence is None:
                    return turn.answer()
                follow_up = on_evidence(turn.evidence)
                if follow_up is None:
                    return ExpertAnswer("", "Stopped, because you declined to share that.")
                text = follow_up
        return ExpertAnswer("", f"The expert kept asking for more after {MAX_FOLLOW_UPS} follow-ups.")


class _Turn:
    """Accumulates one send_message stream."""

    def __init__(self) -> None:
        self.task_id, self.context_id = "", ""
        self.data: dict[str, Any] | None = None
        self.text: list[str] = []
        self.evidence: EvidenceRequest | None = None
        self.failed = ""

    def read(self, response: Any, on_status: Callable[[str], None]) -> None:
        from a2a.helpers.proto_helpers import get_message_text

        if response.HasField("message"):
            self.text.append(get_message_text(response.message))
        if response.HasField("task"):
            task = response.task
            self.task_id, self.context_id = task.id, task.context_id
            for artifact in task.artifacts:
                self._artifact(artifact)
            self._status(task.status, on_status)
        if response.HasField("status_update"):
            update = response.status_update
            self.task_id, self.context_id = update.task_id, update.context_id
            self._status(update.status, on_status)
        if response.HasField("artifact_update"):
            self._artifact(response.artifact_update.artifact)

    def _artifact(self, artifact: Any) -> None:
        from a2a.helpers.proto_helpers import get_data_parts

        for data in get_data_parts(artifact.parts):
            if isinstance(data, dict):
                self.data = data
        self.text.extend(p.text for p in artifact.parts if p.text)

    def _status(self, status: Any, on_status: Callable[[str], None]) -> None:
        from a2a.helpers.proto_helpers import get_data_parts, get_message_text
        from a2a.types.a2a_pb2 import TaskState

        message_text = get_message_text(status.message) if status.HasField("message") else ""
        if status.state == TaskState.TASK_STATE_INPUT_REQUIRED and status.HasField("message"):
            for data in get_data_parts(status.message.parts):
                self.evidence = _evidence_request(data) or self.evidence
        elif status.state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_REJECTED):
            self.failed = message_text or "the expert couldn't complete the task"
        elif status.state == TaskState.TASK_STATE_WORKING and message_text:
            on_status(message_text)
        elif status.state == TaskState.TASK_STATE_COMPLETED and message_text:
            self.text.append(message_text)

    def answer(self) -> ExpertAnswer:
        if self.failed:
            return ExpertAnswer("", self.failed)
        if self.data is not None:
            return ExpertAnswer(str(self.data.get("command", "")).strip(), str(self.data.get("explanation", "")).strip(),
                                json.dumps(self.data))  # fmt: skip
        return parse_answer("\n".join(x for x in self.text if x))
