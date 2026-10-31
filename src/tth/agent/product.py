"""The product environment: the user's real working directory, read-only probes only."""

from __future__ import annotations

import platform
from pathlib import Path

from tth.agent.prompts import SHELL_SYSTEM
from tth.agent.protocol import EpisodeStart, StepOutcome, ToolCall
from tth.tools.shell_tools import ShellToolbox


def os_label() -> str:
    return "macOS" if platform.system() == "Darwin" else platform.system()


class ProductEnv:
    domain = "shell"

    def __init__(self, request: str, cwd: Path, shell: str) -> None:
        self.request = request
        self.box = ShellToolbox(cwd, shell=shell)
        self.staged: dict | None = None
        self.decision: str | None = None
        self.reason: str = ""
        self.last_verified: str | None = None

    def reset(self) -> EpisodeStart:
        return EpisodeStart(system=SHELL_SYSTEM.format(os=os_label(), shell=self.box.shell), user=self.request,
                            tools=self.box.defs())  # fmt: skip

    def step(self, call: ToolCall) -> StepOutcome:
        if call.malformed:
            return StepOutcome({"error": f"could not parse tool call: {call.malformed}"}, ok=False)
        r = self.box.call(call.name, call.args)
        if call.name == "dry_run" and r.ok and r.data.get("exit_code") == 0:
            cmd = call.args.get("command") or r.data.get("ran")
            if cmd:
                self.last_verified = cmd
        if r.decision:
            self.decision = r.decision
            if r.decision == "stage":
                self.staged = r.data
            self.reason = str(call.args.get("reason") or call.args.get("question") or "")
            return StepOutcome(r.data, ok=True, done=True, decision=r.decision)
        return StepOutcome(r.data, ok=r.ok)

    def _auto_stage_verified(self) -> StepOutcome | None:
        if self.last_verified:
            r = self.box.call("stage", {"command": self.last_verified, "explanation": "Verified locally via dry_run."})
            if r.decision == "stage":
                self.decision = "stage"
                self.staged = r.data
                return StepOutcome(r.data, ok=True, done=True, decision="stage")
        return None

    def on_text(self, text: str) -> StepOutcome:
        staged = self._auto_stage_verified()
        if staged:
            return staged
        return StepOutcome("Reply with exactly one tool call (stage, escalate or ask_user to finish).", ok=False)

    def finish(self, reason: str) -> StepOutcome:
        staged = self._auto_stage_verified()
        if staged:
            return staged
        # Step budget exhausted: forced escalate-or-ask.
        self.decision = "budget"
        self.reason = "local step budget exhausted without a verified command"
        return StepOutcome({"ended": reason}, done=True, decision="budget")
