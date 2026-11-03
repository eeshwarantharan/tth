"""The product environment: the user's real directory, and a gate between the model and the prompt.

The gate holds the rules that make "look before you act" structural, so the model doesn't have to learn them:

- Staging a read-only command tests it first, without spending a model turn.
- Staging a command that changes files requires having looked at something first.
- Repeating a passing test commits it. Repeating a failure stops the loop.
- A reply saying it can't do the task stages nothing.
- Nothing is swapped for a different command, and "verified" means this exact command passed with output.
"""

from __future__ import annotations

import platform
from pathlib import Path

from tth.agent.decisions import DECISION_DEFS, DECISION_NAMES, decide
from tth.agent.prompts import SHELL_SYSTEM
from tth.core.types import EpisodeStart, StepOutcome, ToolCall, ToolResult
from tth.mcp.client import ToolSession, connect
from tth.safety.rewriter import selection_part

PROBES = {"inspect_context", "probe", "dry_run"}
OBSERVATIONS = {"inspect_context", "probe"}


def os_label() -> str:
    return "macOS" if platform.system() == "Darwin" else platform.system()


def _norm(command: object) -> str:
    return " ".join(str(command).split())


def _clean_exit(r: ToolResult) -> bool:
    """Exit 1 with nothing on stderr is how grep and friends say "no match". It isn't a broken command."""
    return r.data.get("exit_code") in (0, 1) and not r.data.get("stderr") and not r.data.get("error")


def _has_output(r: ToolResult) -> bool:
    """Blank lines don't count: an awk summing nothing prints one, and that isn't an answer."""
    return any(str(line).strip() for line in r.data.get("sample") or [])


def _passed_with_output(r: ToolResult) -> bool:
    return _clean_exit(r) and _has_output(r)


def _ran_with_no_output(r: ToolResult) -> bool:
    return _clean_exit(r) and not _has_output(r)


class ProductEnv:
    domain = "shell"

    def __init__(self, request: str, cwd: Path, shell: str, tools: ToolSession | None = None) -> None:
        self.request = request
        self.cwd = cwd
        self.shell = shell
        self.tools = tools or connect(cwd, shell=shell)
        self.staged: dict | None = None
        self.decision: str | None = None
        self.reason: str = ""
        self.verified: bool = False
        self.staged_from: str | None = None  # the command as the model wrote it, before any rewrite
        self.last_passed: str | None = None  # the last command that passed with output, as the model wrote it
        self._looked = False
        self._warned_empty: set[str] = set()
        self._failed: dict[str, str] = {}  # command -> error
        self._untestable: set[str] = set()  # commands that change files, which dry_run can't run
        self._said_no = False

    def reset(self) -> EpisodeStart:
        return EpisodeStart(system=SHELL_SYSTEM.format(os=os_label(), shell=self.shell), user=self.request,
                            tools=self.tools.tools(only=PROBES) + DECISION_DEFS)  # fmt: skip

    def close(self) -> None:
        self.tools.close()

    # ---- the gate -----------------------------------------------------------------------------

    def step(self, call: ToolCall) -> StepOutcome:
        if call.malformed:
            return StepOutcome({"error": f"could not parse tool call: {call.malformed}"}, ok=False)
        self._said_no = False
        if call.name == "stage":
            return self._stage(call.args)
        if call.name in DECISION_NAMES:
            r = decide(call.name, call.args, self.cwd)
            if r.decision:
                self.decision = r.decision
                self.reason = str(call.args.get("reason") or call.args.get("question") or "")
                return StepOutcome(r.data, ok=True, done=True, decision=r.decision)
            return StepOutcome(r.data, ok=r.ok)
        if call.name == "dry_run":
            return self._dry_run(call.args)
        r = self.tools.call(call.name, call.args)
        if r.ok and call.name in OBSERVATIONS:
            self._looked = True
        return StepOutcome(r.data, ok=r.ok)

    def _dry_run(self, args: dict) -> StepOutcome:
        command = _norm(args.get("command", ""))
        if command and command == self.last_passed:
            return self._commit(command, verified=True, source="gate")
        if command in self._untestable:
            return self._commit(command, verified=False, source="gate")
        if command in self._failed:
            self.decision = "escalate"
            self.reason = f"the command failed ({self._failed[command]}) and the local model couldn't fix it"
            return StepOutcome({"error": f"repeated a failing command: {self._failed[command]}", "escalated": True},
                               ok=False, done=True, decision="escalate", source="gate")  # fmt: skip
        r = self.tools.call("dry_run", args)
        self._looked = self._looked or r.ok
        self._remember_test(command, r)
        return StepOutcome(r.data, ok=r.ok)

    def _remember_test(self, command: str, r: ToolResult) -> None:
        if _passed_with_output(r):
            self.last_passed = command
            self._failed.pop(command, None)
        elif r.data.get("ran") is False and r.data.get("tier") == "changes-files":
            self._untestable.add(command)
        elif not r.ok and command and not _ran_with_no_output(r):
            self._failed[command] = str(r.data.get("error") or r.data.get("stderr") or "failed")[:120]

    def _stage(self, args: dict) -> StepOutcome:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return StepOutcome({"error": "bad arguments: stage needs a command"}, ok=False)
        norm = _norm(command)
        if selection_part(command) is None:
            # dry_run can't run this, so the evidence is whatever the model looked at first.
            if not self._looked:
                self._looked = True  # one reminder is enough
                return StepOutcome({"error": "look first: inspect the folder or probe a file before changing files"},
                                   ok=False)  # fmt: skip
            return self._commit(command, verified=False, explanation=args.get("explanation", ""))
        if norm == self.last_passed:
            return self._commit(command, verified=True, explanation=args.get("explanation", ""))
        r = self.tools.call("dry_run", {"command": command})
        self._remember_test(norm, r)
        if _passed_with_output(r):
            return self._commit(command, verified=True, explanation=args.get("explanation", ""))
        if _ran_with_no_output(r):
            if norm in self._warned_empty:
                return self._commit(command, verified=False, explanation=args.get("explanation", ""))
            self._warned_empty.add(norm)
            return StepOutcome({"error": "the command ran but gave no output, so it may be wrong. Fix it, or stage "
                                         "it again if no output is right.", "exit_code": r.data.get("exit_code")},
                               ok=False)  # fmt: skip
        return StepOutcome({"error": f"not staged, because testing it failed: {r.data.get('error') or r.data.get('stderr') or 'failed'}"},
                           ok=False)  # fmt: skip

    def _commit(self, command: str, verified: bool, explanation: str = "", source: str = "model") -> StepOutcome:
        r = decide("stage", {"command": command, "explanation": explanation}, self.cwd)
        if r.decision != "stage":
            return StepOutcome(r.data, ok=False, source=source)
        self.decision = "stage"
        self.staged = r.data
        self.staged_from = command
        self.verified = verified
        return StepOutcome(r.data, ok=True, done=True, decision="stage", source=source)

    # ---- when the model stops calling tools ------------------------------------------------------

    def on_text(self, text: str) -> StepOutcome:
        self._said_no = bool(text.strip())
        return StepOutcome("Reply with exactly one tool call (stage, escalate or ask_user to finish).", ok=False)

    def finish(self, reason: str) -> StepOutcome:
        # Out of attempts with a tested command: offer it, since it's only placed on the prompt. Not if the model's
        # last words were that it couldn't do this.
        if self.last_passed and not self._said_no:
            out = self._commit(self.last_passed, verified=True, source="gate")
            if out.done:
                return out
        self.decision = "budget"
        self.reason = "ran out of attempts without a verified command"
        return StepOutcome({"ended": reason}, done=True, decision="budget")
