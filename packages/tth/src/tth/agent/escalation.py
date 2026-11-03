"""The one path to an expert. Invariant 5 lives here: nothing leaves the machine without consent.

Every piece of text bound for an expert is redacted and shown to the user through `confirm` first, including any
evidence the expert asks for mid-task. The answer is never trusted: it goes through the safety layer and a local
test, the same as the model's own commands. Import rules stop the CLI from reaching the A2A client directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from tth.a2a.client import EvidenceRequest, ExpertCard, ExpertClient
from tth.a2a.payload import build_payload
from tth.a2a.redact import redact
from tth.agent.product import ProductEnv, os_label
from tth.safety.rewriter import rewrite, selection_part
from tth.safety.staging import has_control_characters

# What an expert may ask to see: read-only tools on the MCP server, which refuses secret paths.
EVIDENCE_TOOLS = {"inspect_context", "probe", "dry_run"}


@dataclass
class ConsentRequest:
    destination: str
    text: str
    redactions: list[str]
    reason: str
    is_local: bool
    follow_up: bool = False


@dataclass
class Escalation:
    decision: str  # stage | declined | failed
    command: str | None = None
    tier: str | None = None
    notes: list[str] = field(default_factory=list)
    verified: bool = False
    explanation: str = ""
    message: str = ""
    test: dict | None = None  # the local dry-run of the expert's command


class Listener(Protocol):
    def confirm(self, request: ConsentRequest) -> bool: ...

    def status(self, text: str) -> None: ...


def _share_evidence(env: ProductEnv, card: ExpertCard, need: EvidenceRequest, listener: Listener) -> str | None:
    if need.tool not in EVIDENCE_TOOLS:
        result = {"error": f"{need.tool} isn't available. You can ask for: {', '.join(sorted(EVIDENCE_TOOLS))}"}
    else:
        r = env.tools.call(need.tool, need.args)
        result = r.data
    text, hits = redact(json.dumps({"tool": need.tool, "args": need.args, "result": result}, ensure_ascii=False))
    request = ConsentRequest(card.destination(), text, sorted(set(hits)), need.why or "the expert asked to see this",
                             card.is_local, follow_up=True)  # fmt: skip
    return text if listener.confirm(request) else None


def escalate(
    env: ProductEnv,
    messages: list,
    listener: Listener,
    client: ExpertClient | None = None,
    start_expert: Callable[[], bool] | None = None,
) -> Escalation:
    client = client or ExpertClient()
    card = client.card()
    if card is None and start_expert is not None and start_expert():
        card = client.card()
    if card is None:
        return Escalation("failed", message="The expert agent isn't reachable. Start one with: tth expert")

    reason = env.reason or "could not verify locally"
    payload = build_payload(env.request, messages, reason, os_label(), env.shell)
    consent = ConsentRequest(card.destination(), payload.text, payload.redactions, reason, card.is_local)
    if not listener.confirm(consent):
        return Escalation("declined", message="Not sent. Nothing staged.")

    answer = client.ask(payload.text, lambda need: _share_evidence(env, card, need, listener), listener.status)
    if not answer.command:
        return Escalation("failed", message=f"The expert didn't give a command: {answer.explanation}")

    if has_control_characters(answer.command):
        return Escalation("failed", message="The safety layer refused the expert's command: it has control characters.")
    rw = rewrite(answer.command, cwd=env.cwd)
    if rw.staged is None:
        notes = "; ".join(rw.notes)
        return Escalation(
            "failed", message=f"The safety layer refused the expert's command: {answer.command} ({notes})"
        )
    verified, test = False, None
    if selection_part(answer.command) is not None:
        r = env.tools.call("dry_run", {"command": answer.command})
        has_output = any(str(line).strip() for line in r.data.get("sample") or [])
        verified, test = r.ok and has_output, {"ok": r.ok, **r.data}
    return Escalation("stage", rw.staged, rw.tier.label, rw.notes, verified, answer.explanation, test=test)
