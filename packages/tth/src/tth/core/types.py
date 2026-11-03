"""Domain-agnostic contracts between the agent loop, environments and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    # Product tools must declare these (see CLAUDE.md "Working conventions").
    read_only: bool = True
    sandboxed: bool = False
    timeout_s: float | None = None
    output_cap: int | None = None

    def openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""
    malformed: str | None = None  # parse error text if the model emitted a broken call


@dataclass
class ModelTurn:
    content: str
    tool_calls: list[ToolCall]
    raw: str = ""  # exact text the model produced (local models)


@dataclass
class ToolResult:
    data: dict[str, Any]
    ok: bool = True
    decision: str | None = None


@dataclass
class StepOutcome:
    result: Any
    ok: bool = True
    done: bool = False
    decision: str | None = None  # set when the episode ends


@dataclass
class EpisodeStart:
    system: str
    user: str
    tools: list[ToolDef]
    meta: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    domain: str

    def reset(self) -> EpisodeStart: ...

    def step(self, call: ToolCall) -> StepOutcome: ...

    def on_text(self, text: str) -> StepOutcome:
        """Called when the model answers with plain text and no tool call."""
        ...

    def finish(self, reason: str) -> StepOutcome:
        """Called when the step budget runs out."""
        ...


class Policy(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn: ...
