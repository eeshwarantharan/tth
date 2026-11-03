"""Scripted teacher for the arm-D table domain (mirrors train/data/scripted_chess.py).

uv run python -m tth_gym.data.scripted_table 760 data/table/scripted_raw.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from tth.agent.loop import run_episode
from tth.core.types import ModelTurn, ToolCall, ToolDef
from tth_gym.envs.table.env import TableEnv, episode_summary, generate, save

TEACHER = "scripted-oracle (table)"


class ScriptedTablePolicy:
    def __init__(self, env: TableEnv) -> None:
        self.env, self.plan, self.last, self.started = env, [], None, False

    def _call(self, canon: str, args: dict[str, Any]) -> ModelTurn:
        st = self.env.st
        named = {st.args[canon][k]: v for k, v in args.items()}
        if canon in ("compute", "submit"):
            named[st.args[canon]["state"]] = st.token
        self.last = (canon, args)
        return ModelTurn("", [ToolCall(st.names[canon], named)])

    def _plan(self) -> list[tuple[str, dict[str, Any]]]:
        t, view = self.env.task, ("view", {"format": self.env.st.fmt})
        p = t.params
        if t.kind == "count":
            return [view, ("query", {"field": "dept", "value": p["value"]}), ("submit", {"answer": t.answer})]
        if t.kind == "sum":
            return [view, ("compute", {"op": f"sum:{p['dept']}", "field": "salary"}), ("submit", {"answer": t.answer})]
        if t.kind == "max_name":
            return [view, ("compute", {"op": "max", "field": "age"}), ("submit", {"answer": t.answer})]
        if t.kind == "lookup":
            return [view, ("inspect", {"name": p["name"]}), ("submit", {"answer": t.answer})]
        if t.kind == "missing":
            return [view, ("inspect", {"name": p["name"]}),
                    ("impossible", {"reason": f"{p['name']} is not in the table."})]  # fmt: skip
        return [view, ("expert", {"question": f"What is the salary forecast for {p['dept']} next year?"})]

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        if not self.started:
            self.started, self.plan = True, self._plan()
        else:
            res = messages[-1].get("content", "")
            try:
                data = json.loads(res) if isinstance(res, str) else res
            except json.JSONDecodeError:
                data = None if str(res).lstrip().startswith("{") else {"text": res}
            err = data.get("error", "") if isinstance(data, dict) else ("" if data is not None else "malformed")
            if err and self.last:
                self.plan.insert(0, self.last)
                if "STALE" in err or "state token" in err:
                    self.plan.insert(0, ("view", {"format": self.env.st.fmt}))
            elif self.last and self.last[0] == "expert" and isinstance(data, dict):
                m = re.search(r"(\d+)", str(data.get("expert", "")))
                self.plan = [("submit", {"answer": m.group(1) if m else ""})]
        canon, args = self.plan.pop(0)
        return self._call(canon, args)


def main(n: int, out: str, seed: int = 5_000) -> None:
    tasks = generate(n, seed)
    save(tasks, Path(out).with_name("tasks_train.jsonl"))
    rows = []
    for i, t in enumerate(tasks):
        env = TableEnv(t, seed=seed * 7 + i)
        ep = run_episode(env, ScriptedTablePolicy(env))
        rows.append({"id": ep.id, "messages": ep.messages, "tools": ep.tools, "summary": episode_summary(env),
                     "meta": {**ep.meta, "teacher": TEACHER}})  # fmt: skip
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"episodes": len(rows), "correct": sum(r["summary"]["correct"] for r in rows)}))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
