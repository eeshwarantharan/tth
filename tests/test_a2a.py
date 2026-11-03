"""The full escalation path over real A2A: an HTTP server, an Agent Card, task states and typed artifacts."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from a2a.helpers.proto_helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from tth.a2a.client import ExpertCard, ExpertClient, parse_answer
from tth.agent.escalation import ConsentRequest, escalate
from tth.agent.product import ProductEnv
from tth.expert.server import build_app, build_card


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app: Any, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("test server didn't start")


class FakeBrain:
    def __init__(self, *replies: dict) -> None:
        self.replies = list(replies)
        self.seen: list[list[str]] = []

    def think(self, conversation: list[str]) -> dict:
        self.seen.append(list(conversation))
        return self.replies.pop(0)


class Listener:
    def __init__(self, *answers: bool) -> None:
        self.answers = list(answers)
        self.requests: list[ConsentRequest] = []
        self.statuses: list[str] = []

    def confirm(self, request: ConsentRequest) -> bool:
        self.requests.append(request)
        return self.answers.pop(0)

    def status(self, text: str) -> None:
        self.statuses.append(text)


@pytest.fixture
def project(tmp_path: Path) -> Iterator[ProductEnv]:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    (tmp_path / ".env").write_text("API_KEY=hunter2\n")
    env = ProductEnv("count the lines", tmp_path, "zsh")
    env.reset()
    yield env
    env.close()


def expert(brain: FakeBrain, forwards_to: str = "", token: str | None = None) -> tuple[str, uvicorn.Server]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    return url, _serve(build_app(url, brain, forwards_to, token), port)


def test_the_consent_prompt_names_where_the_text_really_goes(project: ProductEnv) -> None:
    url, server = expert(FakeBrain(), forwards_to="https://api.example.com/v1")
    listener = Listener(False)
    result = escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "declined"
    request = listener.requests[0]
    assert "tth-expert" in request.destination
    assert "https://api.example.com/v1" in request.destination
    assert request.is_local is False


def test_declining_means_the_expert_receives_nothing(project: ProductEnv) -> None:
    brain = FakeBrain()
    url, server = expert(brain)
    escalate(project, [], Listener(False), ExpertClient(url, token=""))
    server.should_exit = True
    assert brain.seen == []


def test_a_typed_answer_is_tested_locally(project: ProductEnv) -> None:
    url, server = expert(FakeBrain({"command": "wc -l a.txt", "explanation": "counts lines"}))
    listener = Listener(True)
    result = escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "stage"
    assert result.command == "wc -l a.txt"
    assert result.verified is True
    assert result.explanation == "counts lines"
    assert "Working on it." in listener.statuses


def test_a_dangerous_answer_is_refused(project: ProductEnv) -> None:
    url, server = expert(FakeBrain({"command": "rm -rf ~", "explanation": "cleans up"}))
    result = escalate(project, [], Listener(True), ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "failed"
    assert result.command is None
    assert "refused" in result.message


def test_a_broken_answer_is_not_called_verified(project: ProductEnv) -> None:
    url, server = expert(FakeBrain({"command": "cat missing.txt", "explanation": "x"}))
    result = escalate(project, [], Listener(True), ExpertClient(url, token=""))
    server.should_exit = True
    assert result.verified is False


def test_an_expert_request_for_evidence_needs_fresh_consent(project: ProductEnv) -> None:
    brain = FakeBrain(
        {"need": {"tool": "probe", "args": {"path": "a.txt", "kind": "head"}, "why": "to see the format"}},
        {"command": "wc -l a.txt", "explanation": "x"},
    )
    url, server = expert(brain)
    listener = Listener(True, True)
    result = escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "stage"
    assert len(listener.requests) == 2
    assert listener.requests[1].follow_up is True
    assert listener.requests[1].reason == "to see the format"
    assert "one" in listener.requests[1].text
    assert any("one" in part for part in brain.seen[1])


def test_declining_a_follow_up_stops_the_task(project: ProductEnv) -> None:
    brain = FakeBrain({"need": {"tool": "probe", "args": {"path": "a.txt", "kind": "head"}}})
    url, server = expert(brain)
    result = escalate(project, [], Listener(True, False), ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "failed"
    assert len(brain.seen) == 1


def test_an_expert_cannot_get_a_secret_through_a_follow_up(project: ProductEnv) -> None:
    brain = FakeBrain(
        {"need": {"tool": "probe", "args": {"path": ".env", "kind": "head"}}},
        {"command": "", "explanation": "no"},
    )
    url, server = expert(brain)
    listener = Listener(True, True)
    escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert "hunter2" not in listener.requests[1].text
    assert "hunter2" not in " ".join(brain.seen[1])


def test_an_expert_cannot_ask_to_run_something_else(project: ProductEnv) -> None:
    brain = FakeBrain(
        {"need": {"tool": "stage", "args": {"command": "rm -rf ~"}}},
        {"command": "", "explanation": "no"},
    )
    url, server = expert(brain)
    listener = Listener(True, True)
    escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert "isn't available" in listener.requests[1].text
    assert project.staged is None


def test_an_expert_that_keeps_asking_is_stopped(project: ProductEnv) -> None:
    need = {"need": {"tool": "inspect_context", "args": {}}}
    brain = FakeBrain(need, need, need, need, need)
    url, server = expert(brain)
    result = escalate(project, [], Listener(True, True, True, True, True), ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "failed"
    assert "kept asking" in result.message


def test_a_model_failure_is_reported_not_raised(project: ProductEnv) -> None:
    class Broken:
        def think(self, conversation: list[str]) -> dict:
            raise RuntimeError("the model is down")

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    server = _serve(build_app(url, Broken()), port)
    result = escalate(project, [], Listener(True), ExpertClient(url, token=""))
    server.should_exit = True
    assert result.decision == "failed"
    assert "the model is down" in result.message


def test_no_expert_running_is_a_clear_message(project: ProductEnv) -> None:
    client = ExpertClient(f"http://127.0.0.1:{_free_port()}/", token="")
    result = escalate(project, [], Listener(), client)
    assert result.decision == "failed"
    assert "isn't reachable" in result.message


def test_the_expert_can_be_started_on_demand(project: ProductEnv) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    servers: list[uvicorn.Server] = []

    def start() -> bool:
        servers.append(_serve(build_app(url, FakeBrain({"command": "wc -l a.txt"})), port))
        return True

    result = escalate(project, [], Listener(True), ExpertClient(url, token=""), start_expert=start)
    servers[0].should_exit = True
    assert result.decision == "stage"


def test_a_token_protects_everything_but_the_card() -> None:
    url, server = expert(FakeBrain(), token="s3cret")
    card = httpx.get(url + ".well-known/agent-card.json", timeout=5)
    denied = httpx.post(url, json={}, timeout=5)
    wrong = httpx.post(url, json={}, headers={"Authorization": "Bearer nope"}, timeout=5)
    allowed = httpx.post(url, json={}, headers={"Authorization": "Bearer s3cret"}, timeout=5)
    server.should_exit = True
    assert card.status_code == 200
    assert denied.status_code == 401
    assert wrong.status_code == 401
    assert allowed.status_code != 401


def test_the_client_sends_its_token(project: ProductEnv) -> None:
    url, server = expert(FakeBrain({"command": "wc -l a.txt"}), token="s3cret")
    result = escalate(project, [], Listener(True), ExpertClient(url, token="s3cret"))
    server.should_exit = True
    assert result.decision == "stage"


def test_listening_beyond_this_machine_requires_a_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tth.expert import server as expert_server

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("TTH_EXPERT_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="needs a token"):
        expert_server.main(["--host", "0.0.0.0"])  # noqa: S104


# ---- Interop: an A2A agent that wasn't built for tth ----------------------------------------------


class PlainTextAgent(AgentExecutor):
    """Replies with one plain message: no tasks, no artifacts. Any A2A agent could do this."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        reply = 'Here you go: {"command": "wc -l a.txt", "explanation": "counts lines"}'
        await event_queue.enqueue_event(new_text_message(reply, context_id=context.context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None


def test_any_a2a_agent_can_be_an_expert(project: ProductEnv) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    card = build_card(url, name="someone-elses-agent")
    handler = DefaultRequestHandler(agent_executor=PlainTextAgent(), task_store=InMemoryTaskStore(), agent_card=card)
    server = _serve(
        Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/")), port
    )
    listener = Listener(True)
    result = escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert "someone-elses-agent" in listener.requests[0].destination
    assert result.command == "wc -l a.txt"
    assert result.verified is True


def test_only_our_own_local_expert_can_claim_to_stay_on_this_machine() -> None:
    assert ExpertCard("x", "http://127.0.0.1:1/", reference=True).is_local
    assert not ExpertCard("x", "http://127.0.0.1:1/", forwards_to="https://hosted", reference=True).is_local
    assert not ExpertCard("x", "https://expert.example.com/", reference=True).is_local
    assert not ExpertCard("someone-elses-agent", "http://127.0.0.1:1/").is_local


def test_a_local_reference_expert_says_so(project: ProductEnv) -> None:
    url, server = expert(FakeBrain())
    listener = Listener(False)
    escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert listener.requests[0].is_local is True


def test_another_agent_is_never_called_local(project: ProductEnv) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    card = build_card(url, name="someone-elses-agent")
    card.provider.organization = "someone else"
    handler = DefaultRequestHandler(agent_executor=PlainTextAgent(), task_store=InMemoryTaskStore(), agent_card=card)
    server = _serve(
        Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/")), port
    )
    listener = Listener(False)
    escalate(project, [], listener, ExpertClient(url, token=""))
    server.should_exit = True
    assert listener.requests[0].is_local is False


def test_plain_text_answers_with_a_code_fence_parse() -> None:
    assert parse_answer("Try:\n```bash\nwc -l f\n```").command == "wc -l f"


def test_only_the_escalation_path_sends_anything_to_an_expert() -> None:
    """Consent is enforced in one function, so no other code may call the client's ask()."""
    import ast

    package = Path(__file__).resolve().parents[1] / "packages" / "tth" / "src" / "tth"
    callers = []
    for f in package.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ask":
                callers.append(f.relative_to(package).as_posix())
    assert callers == ["agent/escalation.py"]
