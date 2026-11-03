"""Scripted baselines with no model. A real arm has to beat these for its score to mean anything."""

from __future__ import annotations

import random
from typing import Any

from tth.core.types import ModelTurn, ToolCall, ToolDef

_WORDS = [
    "ls", "ls -la", "cat app.log", "wc -l *", "grep ERROR app.log", "find . -name '*.py'", "head -n 5 sales.csv",
    "app.log", "sales.csv", "metrics.jsonl", "src", ".", "e2e4", "d2d4", "a1", "north", "yes", "no", "1",
]  # fmt: skip


def _fill(schema: dict[str, Any], rng: random.Random, text: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        if name not in schema.get("required", []) and rng.random() < 0.5:
            continue
        args[name] = rng.choice(prop["enum"]) if "enum" in prop else (text or rng.choice(_WORDS))
    return args


class SilentPolicy:
    """Says nothing. The rewards must rank this last."""

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        return ModelTurn(content="", tool_calls=[])


class RandomPolicy:
    """Calls a random tool with plausible arguments. The floor for anything that claims to be learning."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        tool = self.rng.choice(tools)
        return ModelTurn(content="", tool_calls=[ToolCall(name=tool.name, args=_fill(tool.parameters, self.rng))])


class AlwaysEscalatePolicy:
    """Asks for help straight away. A local agent that can't beat this isn't doing the job."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        for tool in tools:
            if tool.name == "escalate" or "expert" in tool.description.lower():
                args = _fill(tool.parameters, self.rng, text="I can't verify this locally.")
                return ModelTurn(content="", tool_calls=[ToolCall(name=tool.name, args=args)])
        return ModelTurn(content="", tool_calls=[])


BASELINES = {"silent": SilentPolicy, "random": RandomPolicy, "always-escalate": AlwaysEscalatePolicy}
