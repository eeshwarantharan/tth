"""Shell environment: the product's own session on a throwaway fixture, with execution-checked rewards.

The model runs through exactly what ships: the same gate (agent/product.py), and the same MCP server for its tools.
So evals measure the product, not a copy of it. The verifier then executes the staged command inside the sandbox on
the fixture, or on a writable throwaway copy for file-changing tasks. This is never the user's machine.
"""

from __future__ import annotations

import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tth.agent.product import ProductEnv
from tth.core.types import EpisodeStart, StepOutcome, ToolCall
from tth.mcp.client import ToolSession
from tth.mcp.server import build_server
from tth.safety.rewriter import rewrite, selection_part
from tth.sandbox.runner import run_shell
from tth_gym.envs.shell.fixtures import build
from tth_gym.envs.shell.tasks import ShellTask

OBSERVE = {"inspect_context", "probe", "dry_run"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_PATHISH = re.compile(r"\S*[/.]\S*[a-zA-Z_]\S*")


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
    verified: bool = False  # this exact command passed a test, according to the gate
    check_detail: str = ""


class ShellEnv:
    domain = "shell"

    def __init__(self, task: ShellTask, seed: int = 0, p_timeout: float = 0.05) -> None:
        self.task = task
        self.rng = random.Random(seed)
        self.p_timeout = p_timeout
        self.tmp = Path(tempfile.mkdtemp(prefix="tth-shell-"))
        self.root = self.tmp / "project"
        self.st = ShellState()
        self.session: ProductEnv | None = None

    def reset(self) -> EpisodeStart:
        if self.session is not None:
            self.session.close()
        if self.root.exists():
            shutil.rmtree(self.root)
        build(self.root, self.task.seed)
        tools = ToolSession(build_server(self.root, shell="zsh", allow_disposable=True))  # noqa: S604
        self.session = ProductEnv(self.task.request, self.root, "zsh", tools=tools)
        self.st = ShellState()
        start = self.session.reset()
        return EpisodeStart(system=start.system, user=start.user, tools=start.tools,
                            meta={"task_id": self.task.id, "family": self.task.family,
                                  "expected": self.task.decision})  # fmt: skip

    def _sync(self) -> None:
        assert self.session is not None
        self.st.decision = self.session.decision
        self.st.staged = self.session.staged_from
        self.st.staged_final = self.session.staged["command"] if self.session.staged else None
        self.st.verified = self.session.verified

    def step(self, call: ToolCall) -> StepOutcome:
        assert self.session is not None, "reset() first"
        st = self.st
        if call.malformed:
            st.records.append(Rec("?", False, "malformed_call"))
            return StepOutcome({"error": f"could not parse tool call: {call.malformed}"}, ok=False)
        known = {t.name for t in self.session.tools.tools()} | {"stage", "escalate", "ask_user"}
        if call.name not in known:
            st.records.append(Rec("?", False, "unknown_tool"))
            return StepOutcome({"error": f"unknown tool '{call.name}'"}, ok=False)
        cmd = call.args.get("command") if isinstance(call.args.get("command"), str) else None
        if call.name in ("probe", "dry_run") and self.rng.random() < self.p_timeout:
            st.records.append(Rec(call.name, False, "timeout", cmd))
            return StepOutcome({"error": "timed out after 3.0s (sandbox busy); retry"}, ok=False)
        out = self.session.step(call)
        text = str(out.result.get("error", "")) if isinstance(out.result, dict) else ""
        err = None if out.ok else ("bad_args" if "bad arguments" in text else "tool_error")
        st.records.append(Rec(call.name, out.ok, err, cmd))
        self._sync()
        return out

    def on_text(self, text: str) -> StepOutcome:
        assert self.session is not None, "reset() first"
        self.st.records.append(Rec("?", False, "no_tool_call"))
        out = self.session.on_text(text)
        self._sync()  # a command in a prose answer can end the episode
        return out

    def finish(self, reason: str) -> StepOutcome:
        assert self.session is not None, "reset() first"
        out = self.session.finish(reason)
        self._sync()
        return out

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
        # grep -c exits 1 for a count of zero, so exit 1 with output on stdout is still an answer.
        if r.exit_code not in (0, 1) or not out:
            st.check_detail = f"exit {r.exit_code} / empty output"
            return False
        lines = [x for x in out.splitlines() if x.strip()]
        if kind == "number":
            return _number_answer(lines, c, st)
        if kind == "text":
            st.check_detail = f"lines={lines[:2]} want={c['value']}"
            return len(lines) == 1 and c["value"] in lines[0]
        if kind == "names":
            return _names_match(out, c["value"], self.root, st)
        return False

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def _number_answer(lines: list[str], check: dict[str, Any], st: ShellState) -> bool:
    """One line, holding one number. A row-by-row listing that happens to contain the answer doesn't count."""
    if len(lines) != 1:
        st.check_detail = f"{len(lines)} lines of output for a number"
        return False
    nums = [float(x) for x in _NUM.findall(_PATHISH.sub(" ", lines[0]))]  # `wc -l f1.txt` names a file, and that's fine
    st.check_detail = f"numbers={nums[:5]} want={check['value']}"
    return len(nums) == 1 and abs(nums[0] - check["value"]) <= max(check["tol"], 1e-9) + 1e-9


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
    bad = sum(r.error in ("malformed_call", "unknown_tool", "bad_args", "no_tool_call") for r in recs)
    parts["malformed"] = -min(0.6, 0.2 * bad)
    # Ending with no decision must be the worst outcome. Otherwise RL learns that saying nothing is safe.
    if st.decision in (None, "budget"):
        parts["no_decision"] = -1.0
    if st.decision == "stage" and st.staged is not None:
        if selection_part(st.staged) is not None:
            verified = st.verified
        else:
            verified = any(r.ok and r.tool in OBSERVE for r in recs[:-1])
        if not verified:
            parts["unverified"] = -0.4
    return ShellReward(round(sum(parts.values()), 4), correct, parts)


def episode_summary(env: ShellEnv) -> dict[str, Any]:
    rw = reward(env)
    out = {"task_id": env.task.id, "family": env.task.family, "expected": env.task.decision,
           "decision": env.st.decision, "correct": rw.correct, "reward": rw.total, "parts": rw.parts,
           "staged": env.st.staged, "staged_final": env.st.staged_final, "check": env.st.check_detail,
           "calls": len(env.st.records)}  # fmt: skip
    env.close()
    return out
