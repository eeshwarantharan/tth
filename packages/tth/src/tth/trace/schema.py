"""Unified trajectory schema shared by product runs, training rollouts and evals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Call(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    episode: str
    domain: str  # "chess" | "shell"
    step: int
    call: Call | None = None
    result: Any = None
    ok: bool = True
    decision: str | None = None  # stage | escalate | ask | answer | impossible | budget
    reward: float | None = None
    ts: float = Field(default_factory=time.time)
    meta: dict[str, Any] = Field(default_factory=dict)


class TraceLogger:
    """Append-only JSONL writer. One file can hold many episodes."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Traces hold probe output from the user's files, so only the user may read them.
            self.path.touch(mode=0o600, exist_ok=True)
            self.path.chmod(0o600)

    def write(self, event: TraceEvent) -> None:
        if not self.path:
            return
        with self.path.open("a") as f:
            f.write(event.model_dump_json() + "\n")


def read_trace(path: str | Path) -> list[TraceEvent]:
    return [TraceEvent.model_validate(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
