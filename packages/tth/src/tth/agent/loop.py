"""The model <-> tools loop. Same code for product runs and training rollouts.

`EpisodeRunner` advances one episode by one model turn; `run_episode` drives a
single episode, `run_batched` drives many in lockstep (one batched generation
per step) for fast rollouts on a GPU.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tth.core.types import Environment, ModelTurn, Policy, StepOutcome, ToolDef
from tth.trace.schema import Call, TraceEvent, TraceLogger

DEFAULT_BUDGET = 6
MAX_TEXT_STREAK = 2  # a model that twice answers without a tool call is not going to recover


@dataclass
class Episode:
    id: str
    domain: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    events: list[TraceEvent] = field(default_factory=list)
    decision: str | None = None
    final: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


class EpisodeRunner:
    def __init__(self, env: Environment, budget: int = DEFAULT_BUDGET, logger: TraceLogger | None = None,
                 on_event: Callable[[TraceEvent], None] | None = None, episode_id: str | None = None) -> None:  # fmt: skip
        self.env = env
        self.budget = budget
        self.logger = logger
        self.on_event = on_event
        start = env.reset()
        self.tool_defs: list[ToolDef] = start.tools
        self.ep = Episode(
            id=episode_id or uuid.uuid4().hex[:12],
            domain=env.domain,
            messages=[
                {"role": "system", "content": start.system},
                {"role": "user", "content": start.user},
            ],
            tools=[t.openai() for t in start.tools],
            meta=dict(start.meta),
        )
        system_hash = hashlib.sha256((start.system + json.dumps(self.ep.tools, sort_keys=True)).encode()).hexdigest()[
            :12
        ]
        self._emit(TraceEvent(episode=self.ep.id, domain=env.domain, step=0, meta={
            "kind": "start", "request": start.user, "tools": [t.name for t in start.tools], "system_hash": system_hash,
        }))  # fmt: skip
        self.step = 0
        self.calls = 0
        self.turns = 0
        self.text_streak = 0  # consecutive turns without a tool call
        self.done = False
        self.outcome: StepOutcome | None = None

    def _emit(self, ev: TraceEvent) -> None:
        self.ep.events.append(ev)
        if self.logger:
            self.logger.write(ev)
        if self.on_event:
            self.on_event(ev)

    def _end(self, outcome: StepOutcome) -> None:
        self.done = True
        self.outcome = outcome
        self.ep.decision = outcome.decision
        self.ep.final = outcome.result

    def _budget_end(self) -> None:
        out = self.env.finish("budget")
        self._emit(TraceEvent(episode=self.ep.id, domain=self.env.domain, step=self.step, result=out.result,
                              ok=out.ok, decision=out.decision))  # fmt: skip
        self._end(out)

    def advance(self, turn: ModelTurn) -> None:
        """Apply one model turn. A few extra turns are allowed to recover from malformed output."""
        ep, env = self.ep, self.env
        self.turns += 1
        if not turn.tool_calls:
            self.text_streak += 1
            ep.messages.append({"role": "assistant", "content": turn.content})
            out = env.on_text(turn.content)
            self._emit(TraceEvent(episode=ep.id, domain=env.domain, step=self.step, result=turn.content[:2000],
                                  ok=out.ok, decision=out.decision, raw=turn.raw))  # fmt: skip
            self.step += 1
            if out.done:
                self._end(out)
                return
            if self.text_streak >= MAX_TEXT_STREAK:
                self._budget_end()
                return
            ep.messages.append({"role": "user", "content": _result_text(out.result)})
        else:
            self.text_streak = 0
            call = turn.tool_calls[0]  # one call per step keeps trajectories simple
            call_id = call.id or f"call_{self.step}"
            ep.messages.append({
                "role": "assistant", "content": turn.content or "",
                "tool_calls": [{"id": call_id, "type": "function",
                                "function": {"name": call.name, "arguments": call.args}}],
            })  # fmt: skip
            out = env.step(call)
            self.calls += 1
            ep.messages.append({"role": "tool", "tool_call_id": call_id, "name": call.name,
                                "content": _result_text(out.result)})  # fmt: skip
            self._emit(TraceEvent(episode=ep.id, domain=env.domain, step=self.step,
                                  call=Call(tool=call.name, args=call.args), result=out.result, ok=out.ok,
                                  decision=out.decision, raw=turn.raw))  # fmt: skip
            self.step += 1
            if out.done:
                self._end(out)
                return
            if self.calls >= self.budget:
                self._budget_end()
                return
        if self.turns >= self.budget + 3:
            self._budget_end()


def run_episode(env: Environment, policy: Policy, budget: int = DEFAULT_BUDGET, logger: TraceLogger | None = None,
                on_event: Callable[[TraceEvent], None] | None = None, episode_id: str | None = None) -> Episode:  # fmt: skip
    r = EpisodeRunner(env, budget, logger, on_event, episode_id)
    while not r.done:
        r.advance(policy.chat(r.ep.messages, r.tool_defs))
    return r.ep


class BatchPolicy(Protocol):
    def chat_batch(self, batch: list[tuple[list[dict[str, Any]], list[ToolDef]]]) -> list[ModelTurn]: ...


def run_batched(
    envs: Sequence[Environment], policy: BatchPolicy, budget: int = DEFAULT_BUDGET,
                logger: TraceLogger | None = None,
                on_step: Callable[[int, int], None] | None = None) -> list[EpisodeRunner]:  # fmt: skip
    runners = [EpisodeRunner(e, budget, logger) for e in envs]
    while True:
        active = [r for r in runners if not r.done]
        if not active:
            return runners
        if on_step:
            on_step(len(runners) - len(active), len(runners))
        turns = policy.chat_batch([(r.ep.messages, r.tool_defs) for r in active])
        for r, t in zip(active, turns, strict=True):
            r.advance(t)
