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
        self.last_candidate: str | None = None
        self.staged_from: str | None = None  # the command as proposed, before any rewrite
        self._failed_commands: dict[str, str] = {}  # command -> error message

    @property
    def verified(self) -> bool:
        """True only when this exact command passed a dry-run."""
        return self.staged_from is not None and self.staged_from.strip() == (self.last_verified or "").strip()

    def reset(self) -> EpisodeStart:
        return EpisodeStart(system=SHELL_SYSTEM.format(os=os_label(), shell=self.box.shell), user=self.request,
                            tools=self.box.defs())  # fmt: skip

    def step(self, call: ToolCall) -> StepOutcome:
        if call.malformed:
            staged = self._auto_stage_verified()
            if staged:
                return staged
            return StepOutcome({"error": f"could not parse tool call: {call.malformed}"}, ok=False)

        # Detect duplicate dry_run
        if call.name == "dry_run":
            cmd = (call.args.get("command") or "").strip()
            # If the model repeats a modifying command that wasn't dry-runnable, stage it!
            if cmd and cmd == self.last_candidate:
                staged = self._auto_stage_verified()
                if staged:
                    return staged
            # If the model repeats a command that already failed, stop the loop immediately.
            if cmd and cmd in self._failed_commands:
                staged = self._auto_stage_verified()
                if staged:
                    return staged
                self.decision = "escalate"
                self.reason = f"command failed ({self._failed_commands[cmd]}) and local model cannot self-correct"
                return StepOutcome(
                    {"error": f"repeated failing command: {self._failed_commands[cmd]}", "escalated": True},
                    ok=False,
                    done=True,
                    decision="escalate",
                )

        if call.name == "stage":
            staged_cmd = call.args.get("command", "")
            if self.last_verified and any(
                v in staged_cmd for v in ("$lines", "$count", "$output", "$result", "$sample", "$data")
            ):
                call.args["command"] = self.last_verified

        r = self.box.call(call.name, call.args)

        # Track successful, candidate, and failed dry_runs.
        if call.name == "dry_run":
            cmd = call.args.get("command") or r.data.get("ran") or ""
            if r.ok and r.data.get("exit_code") == 0:
                self.last_verified = cmd
                self._failed_commands.pop(cmd, None)
            elif not r.ok and cmd:
                # If command modifies files (changes-files tier), it is not a broken command;
                # remember it as a valid candidate to stage!
                if r.data.get("tier") in ("changes-files", "read-only") and "not dry-runnable" in str(
                    r.data.get("note", "")
                ):
                    self.last_candidate = cmd
                else:
                    err_msg = r.data.get("error") or r.data.get("stderr", "failed")
                    self._failed_commands[cmd] = str(err_msg)[:120]

        if r.decision:
            self.decision = r.decision
            if r.decision == "stage":
                self.staged = r.data
                self.staged_from = call.args.get("command")
            self.reason = str(call.args.get("reason") or call.args.get("question") or "")
            return StepOutcome(r.data, ok=True, done=True, decision=r.decision)
        return StepOutcome(r.data, ok=r.ok)

    def _auto_stage_verified(self) -> StepOutcome | None:
        target = self.last_verified or self.last_candidate
        if target:
            explanation = (
                "Verified locally via dry_run." if target == self.last_verified else "Selected based on context."
            )
            r = self.box.call("stage", {"command": target, "explanation": explanation})
            if r.decision == "stage":
                self.decision = "stage"
                self.staged = r.data
                self.staged_from = target
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
