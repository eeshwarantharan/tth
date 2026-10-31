"""Shell environment: real probes on a throwaway fixture, execution-checked rewards.

The agent uses the exact product toolbox. The verifier executes the staged
command inside the sandbox on the fixture (or on a writable throwaway copy for
file-changing tasks). This is the training/eval container, never the user's machine.
"""

from __future__ import annotations

import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from train.envs.shell.fixtures import build
from train.envs.shell.tasks import ShellTask
from tth.agent.prompts import SHELL_SYSTEM
from tth.agent.protocol import EpisodeStart, StepOutcome, ToolCall
from tth.safety.rewriter import rewrite, selection_part
from tth.sandbox.runner import run_shell
from tth.tools.shell_tools import ShellToolbox

SYSTEM = SHELL_SYSTEM

OBSERVE = {"inspect_context", "probe", "dry_run"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Rec:
    tool: str
    ok: bool
    error: str | None = None
    command: str | None = None


@dataclass
class ShellState:
    records: list[Rec] = field(default_factory=list)
    decision: str | None = None
    staged: str | None = None  # command as the model wrote it
    staged_final: str | None = None  # after rewriting
    check_detail: str = ""


def _norm_cmd(c: str) -> str:
    return " ".join((c or "").split())


class ShellEnv:
    domain = "shell"

    def __init__(self, task: ShellTask, seed: int = 0, p_timeout: float = 0.05, os_name: str = "macOS") -> None:
        self.task = task
        self.rng = random.Random(seed)
        self.p_timeout = p_timeout
        self.os_name = os_name
        self.tmp = Path(tempfile.mkdtemp(prefix="tth-shell-"))
        self.root = self.tmp / "project"
        self.st = ShellState()

    def reset(self) -> EpisodeStart:
        if self.root.exists():
            shutil.rmtree(self.root)
        build(self.root, self.task.seed)
        self.box = ShellToolbox(self.root, shell="zsh", allow_disposable=True)  # noqa: S604 - shell name, not shell=True
        self.st = ShellState()
        return EpisodeStart(system=SYSTEM.format(os=self.os_name, shell="zsh"), user=self.task.request,  # noqa: S604
                            tools=self.box.defs(),
                            meta={"task_id": self.task.id, "family": self.task.family,
                                  "expected": self.task.decision})  # fmt: skip

    def step(self, call: ToolCall) -> StepOutcome:
        st = self.st
        if call.malformed:
            st.records.append(Rec("?", False, "malformed_call"))
            return StepOutcome({"error": f"could not parse tool call: {call.malformed}"}, ok=False)
        if call.name not in {t.name for t in self.box.defs()}:
            st.records.append(Rec("?", False, "unknown_tool"))
            return StepOutcome({"error": f"unknown tool '{call.name}'"}, ok=False)
        cmd = call.args.get("command") if isinstance(call.args.get("command"), str) else None
        if call.name in ("probe", "dry_run") and self.rng.random() < self.p_timeout:
            st.records.append(Rec(call.name, False, "timeout", cmd))
            return StepOutcome({"error": "timed out after 3.0s (sandbox busy); retry"}, ok=False)
        r = self.box.call(call.name, call.args)
        err = None if r.ok else ("bad_args" if "bad arguments" in str(r.data.get("error", "")) else "tool_error")
        st.records.append(Rec(call.name, r.ok, err, cmd))
        if r.decision:
            st.decision = r.decision
            if r.decision == "stage":
                st.staged = cmd
                st.staged_final = r.data.get("command")
            return StepOutcome(r.data, ok=True, done=True, decision=r.decision)
        return StepOutcome(r.data, ok=r.ok)

    def on_text(self, text: str) -> StepOutcome:
        self.st.records.append(Rec("?", False, "no_tool_call"))
        return StepOutcome("Reply with exactly one tool call (stage, escalate or ask_user to finish).", ok=False)

    def finish(self, reason: str) -> StepOutcome:
        self.st.decision = self.st.decision or reason
        return StepOutcome({"ended": reason}, done=True, decision=self.st.decision)

    # ---- verification ---------------------------------------------------------------
    def check(self) -> bool:
        t, st = self.task, self.st
        if st.decision != t.decision:
            return False
        if t.decision != "stage":
            return True
        assert st.staged_final is not None and st.staged is not None
        c = t.check
        kind = c["type"]
        if kind == "delete":
            if "trash" not in st.staged_final:
                st.check_detail = "deletion not rewritten to trash"
                return False
            preview = rewrite(st.staged, cwd=self.root).preview
            if not preview:
                return False
            r = run_shell(preview, self.root, allow_disposable=True)
            return _names_match(r.stdout, c["value"], self.root, st)
        if kind == "fs":
            work = self.tmp / "work"
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(self.root, work, symlinks=True)
            r = run_shell(st.staged_final, work, writable=[str(work)], allow_disposable=True)
            if r.exit_code != 0:
                st.check_detail = f"exit {r.exit_code}: {r.stderr[:200]}"
                return False
            ok = all((work / p).exists() for p in c["exists"]) and not any((work / p).exists() for p in c["absent"])
            st.check_detail = "fs ok" if ok else "fs mismatch"
            return ok
        r = run_shell(st.staged_final, self.root, allow_disposable=True)
        out = r.stdout.strip()
        if r.exit_code != 0 or not out:
            st.check_detail = f"exit {r.exit_code} / empty output"
            return False
        lines = [x for x in out.splitlines() if x.strip()]
        if kind == "number":
            if len(lines) > 3:
                st.check_detail = "too much output for a number"
                return False
            nums = [float(x) for x in _NUM.findall(out)]
            ok = any(abs(n - c["value"]) <= max(c["tol"], 1e-9) + 1e-9 for n in nums)
            st.check_detail = f"numbers={nums[:5]} want={c['value']}"
            return ok
        if kind == "text":
            return len(lines) <= 3 and c["value"] in out
        if kind == "names":
            return _names_match(out, c["value"], self.root, st)
        return False

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def _names_match(out: str, expected: list[str], root: Path, st: ShellState) -> bool:
    got = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        tok = line.split()[-1]
        tok = tok.removeprefix(str(root) + "/").removeprefix("./").rstrip(":,")
        got.add(tok)
    exp = set(expected)
    exp_base = {Path(e).name for e in exp}
    got_base = {Path(g).name for g in got}
    ok = got == exp or (len(exp_base) == len(exp) and got_base == exp_base)
    st.check_detail = f"got={sorted(got)[:8]} want={sorted(exp)[:8]}"
    return ok


@dataclass
class ShellReward:
    total: float
    correct: bool
    parts: dict[str, float]


def reward(env: ShellEnv, escalation_cost: float = 0.3) -> ShellReward:
    t, st = env.task, env.st
    correct = env.check()
    parts: dict[str, float] = {}
    if st.decision == "escalate":
        parts["escalation"] = (0.6 - escalation_cost) if t.decision == "escalate" else -escalation_cost
    elif st.decision == "ask":
        parts["answer"] = 1.0 if t.decision == "ask" else 0.0
    else:
        parts["answer"] = 1.0 if correct else 0.0
    recs = st.records
    parts["malformed"] = -0.2 * sum(r.error in ("malformed_call", "unknown_tool", "bad_args", "no_tool_call")
                                    for r in recs)  # fmt: skip
    if st.decision == "stage" and st.staged is not None:
        before = recs[:-1]
        if selection_part(st.staged) is not None:
            verified = any(r.tool == "dry_run" and r.ok and _norm_cmd(r.command or "") == _norm_cmd(st.staged)
                           for r in before)  # fmt: skip
        else:
            verified = any(r.ok and r.tool in OBSERVE for r in before)
        if not verified:
            parts["unverified"] = -0.4
        last = before[-1] if before else None
        if last and not last.ok and last.tool == "dry_run" and _norm_cmd(last.command or "") == _norm_cmd(st.staged):
            if not any(r.ok and r.tool == "dry_run" and _norm_cmd(r.command or "") == _norm_cmd(st.staged)
                       for r in before):  # fmt: skip
                parts["claim_after_error"] = -1.0
    return ShellReward(round(sum(parts.values()), 4), correct, parts)


def episode_summary(env: ShellEnv) -> dict[str, Any]:
    rw = reward(env)
    out = {"task_id": env.task.id, "family": env.task.family, "expected": env.task.decision,
           "decision": env.st.decision, "correct": rw.correct, "reward": rw.total, "parts": rw.parts,
           "staged": env.st.staged, "staged_final": env.st.staged_final, "check": env.st.check_detail,
           "calls": len(env.st.records)}  # fmt: skip
    env.close()
    return out
