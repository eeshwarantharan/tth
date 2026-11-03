"""Arm D control domain: a hidden data table with the same tool discipline as chess.

Same structure as the chess env (hidden state, schema randomization, stale tokens, timeouts,
malformed responses, distractors, same reward table) but no chess. Used to test whether chess
specifically helps, or any verified tool-use practice does.
"""

from __future__ import annotations

import json
import random
import secrets
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tth.core.types import EpisodeStart, StepOutcome, ToolCall, ToolDef

DEPTS = ["sales", "ops", "research", "legal", "support"]
CITIES = ["Oslo", "Pune", "Quito", "Accra", "Kyoto"]
NAMES = ["ada", "bo", "cy", "dee", "ezra", "finn", "gia", "hal", "ines", "jai", "kai", "lou", "mia", "ned", "ola",
         "pax", "quin", "rae", "sol", "tia", "uma", "vic", "wes", "xia", "yan", "zed"]  # fmt: skip
FIELDS = ["dept", "city", "salary", "age"]

NAME_POOL = {
    "view": ["view_table", "read_rows", "peek_table", "show_data", "dump_rows", "observe_table"],
    "query": ["filter_rows", "query", "select_where", "find_rows", "lookup_rows"],
    "inspect": ["row_info", "get_row", "inspect_row", "fetch_record", "record_details"],
    "compute": ["compute", "aggregate", "evaluate", "run_aggregate", "calc"],
    "submit": ["submit", "commit", "final_answer", "answer", "finalize"],
    "impossible": ["report_impossible", "cannot_do", "decline_task", "task_not_possible"],
    "expert": ["ask_expert", "consult_analyst", "escalate", "request_help"],
}  # fmt: skip
ARG_POOL = {
    "format": ["format", "fmt", "encoding", "style"], "field": ["field", "column", "attr", "key"],
    "value": ["value", "equals", "match", "target"], "name": ["name", "person", "who", "row_name"],
    "op": ["op", "operation", "agg", "fn"], "state": ["state", "state_token", "version", "token", "etag"],
    "answer": ["answer", "value_out", "response", "result"], "reason": ["reason", "why", "explanation"],
    "question": ["question", "query_text", "prompt", "ask"],
}  # fmt: skip
SYSTEM = (
    "You are a careful agent working on a data table you cannot see. Use the tools to look at the data "
    "before answering, verify results before committing, and re-check after any error. If the task is "
    "beyond what you can verify yourself, consult the expert (it has a cost). If the task is impossible, "
    "report it. Call exactly one tool per turn. Tool names and arguments change between tasks: read the tool schemas."
)
DISTRACTORS = [
    ToolDef("get_weather", "Weather for a city.", {"type": "object", "properties": {"city": {"type": "string"}}}),
    ToolDef("send_email", "Send an email. Not allowed during a task.", {"type": "object", "properties": {}}),
]


@dataclass
class TableTask:
    id: str
    kind: str  # count | sum | max_name | lookup | missing | forecast
    seed: int
    question: str
    answer_type: str  # int | text | impossible
    answer: str
    hard: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def make_rows(seed: int) -> list[dict[str, Any]]:
    r = random.Random(seed)
    return [{"name": n, "dept": r.choice(DEPTS), "city": r.choice(CITIES), "salary": r.randrange(30, 150) * 1000,
             "age": r.randint(21, 64)} for n in r.sample(NAMES, r.randint(10, 18))]  # fmt: skip


def forecast(rows: list[dict[str, Any]], dept: str) -> int:
    """Hidden analyst model: only the expert can compute it."""
    s = [x["salary"] for x in rows if x["dept"] == dept]
    return int(round((statistics.mean(s) if s else 0) * 1.07 / 1000)) * 1000


def generate(n: int, seed: int, prefix: str = "tb") -> list[TableTask]:
    r = random.Random(seed)
    kinds = ["count", "sum", "max_name", "lookup", "count", "sum", "missing", "forecast"]
    out = []
    for i in range(n):
        s = seed * 1000 + i
        rows = make_rows(s)
        k = kinds[i % len(kinds)]
        dept = r.choice(sorted({x["dept"] for x in rows}))
        if k == "count":
            out.append(TableTask(f"{prefix}{i}", k, s, f"How many people work in {dept}? Answer with a number.", "int",
                                 str(sum(x["dept"] == dept for x in rows)), params={"field": "dept", "value": dept}))  # fmt: skip
        elif k == "sum":
            out.append(TableTask(f"{prefix}{i}", k, s, f"What is the total salary of the {dept} department?", "int",
                                 str(sum(x["salary"] for x in rows if x["dept"] == dept)), params={"dept": dept}))  # fmt: skip
        elif k == "max_name":
            top = max(rows, key=lambda x: (x["age"], x["name"]))
            out.append(TableTask(f"{prefix}{i}", k, s, "Who is the oldest person in the table? Answer with the name.",
                                 "text", top["name"]))  # fmt: skip
        elif k == "lookup":
            p = r.choice(rows)
            out.append(TableTask(f"{prefix}{i}", k, s, f"What is {p['name']}'s salary?", "int", str(p["salary"]),
                                 params={"name": p["name"]}))  # fmt: skip
        elif k == "missing":
            name = r.choice([x for x in NAMES if x not in {y["name"] for y in rows}])
            out.append(TableTask(f"{prefix}{i}", k, s, f"Give {name} a 5% raise and submit the new salary.",
                                 "impossible", "impossible", params={"name": name}))  # fmt: skip
        else:
            out.append(TableTask(f"{prefix}{i}", k, s, f"What is the analyst's salary forecast for {dept} next year?",
                                 "int", str(forecast(rows, dept)), hard=True, params={"dept": dept}))  # fmt: skip
    return out


def save(tasks: list[TableTask], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(t.to_json() + "\n" for t in tasks))


def load(path: str | Path) -> list[TableTask]:
    return [TableTask(**json.loads(x)) for x in Path(path).read_text().splitlines() if x.strip()]


@dataclass
class Rec:
    canon: str
    ok: bool
    error: str | None = None


@dataclass
class TState:
    names: dict[str, str]
    args: dict[str, dict[str, str]]
    fmt: str
    token: str
    stale_at: int | None
    records: list[Rec] = field(default_factory=list)
    stale_pending: bool = False
    escalated: bool = False
    final_answer: str | None = None
    decision: str | None = None


class TableExpert:
    def answer(self, task: TableTask, rows: list[dict[str, Any]]) -> str:
        if task.kind == "forecast":
            return f"Analyst forecast: {forecast(rows, task.params['dept'])}"
        if task.answer_type == "impossible":
            return "That person is not in the table."
        return f"Answer: {task.answer}"

    def close(self) -> None:
        return None


class TableEnv:
    domain = "table"

    def __init__(self, task: TableTask, seed: int = 0, expert: TableExpert | None = None,
                 p_timeout: float = 0.07, p_malformed: float = 0.05, p_stale: float = 0.2) -> None:  # fmt: skip
        self.task, self.rng, self.expert = task, random.Random(seed), expert or TableExpert()
        self.rows = make_rows(task.seed)
        self.p_timeout, self.p_malformed, self.p_stale = p_timeout, p_malformed, p_stale

    def reset(self) -> EpisodeStart:
        r = self.rng
        names = {k: r.choice(v) for k, v in NAME_POOL.items()}
        a = {k: r.choice(v) for k, v in ARG_POOL.items()}
        fmt = r.choice(["csv", "json", "kv"])
        args = {"view": {"format": a["format"]}, "query": {"field": a["field"], "value": a["value"]},
                "inspect": {"name": a["name"]}, "compute": {"op": a["op"], "field": a["field"], "state": a["state"]},
                "submit": {"answer": a["answer"], "state": a["state"]}, "impossible": {"reason": a["reason"]},
                "expert": {"question": a["question"]}}  # fmt: skip

        def obj(props: dict, req: list[str]) -> dict:
            keys = list(props)
            r.shuffle(keys)
            return {"type": "object", "properties": {k: props[k] for k in keys}, "required": req}

        s = {"type": "string"}
        tools = [
            ToolDef(names["view"], f"Show all rows and the current state token. Format: {fmt}.",
                    obj({a["format"]: {"type": "string", "enum": [fmt]}}, [])),
            ToolDef(names["query"], "Names of rows where a field equals a value.",
                    obj({a["field"]: {"type": "string", "enum": FIELDS}, a["value"]: s}, [a["field"], a["value"]])),
            ToolDef(names["inspect"], "All fields of one row, by name.", obj({a["name"]: s}, [a["name"]])),
            ToolDef(names["compute"], "Aggregate a numeric field over rows, optionally only where dept matches "
                    "(pass op as 'sum', 'count', 'max' or 'sum:<dept>' / 'count:<dept>'). Needs the state token.",
                    obj({a["op"]: s, a["field"]: s, a["state"]: s}, [a["op"], a["field"], a["state"]])),
            ToolDef(names["submit"], "Submit the final answer. Ends the task. Needs the state token.",
                    obj({a["answer"]: s, a["state"]: s}, [a["answer"], a["state"]])),
            ToolDef(names["impossible"], "Report the task cannot be done. Ends the task.", obj({a["reason"]: s}, [a["reason"]])),
            ToolDef(names["expert"], "Ask an expensive expert analyst. Use only when you cannot verify it yourself.",
                    obj({a["question"]: s}, [a["question"]])),
        ]  # fmt: skip
        if r.random() < 0.5:
            tools.append(r.choice(DISTRACTORS))
        r.shuffle(tools)
        self.st = TState(names, args, fmt, secrets.token_hex(3), r.randint(1, 3) if r.random() < self.p_stale else None)
        self.canon_of = {v: k for k, v in names.items()}
        return EpisodeStart(SYSTEM, self.task.question, tools,
                            meta={"task_id": self.task.id, "kind": self.task.kind, "names": names})  # fmt: skip

    def _err(self, rec: Rec, msg: str) -> StepOutcome:
        self.st.records.append(rec)
        return StepOutcome({"error": msg}, ok=False)

    def step(self, call: ToolCall) -> StepOutcome:
        st = self.st
        if call.malformed:
            return self._err(Rec("?", False, "malformed_call"), f"could not parse tool call: {call.malformed}")
        canon = self.canon_of.get(call.name)
        if canon is None:
            return self._err(Rec("?", False, "unknown_tool"), f"unknown tool '{call.name}'")
        exp = st.args[canon]
        args: dict[str, Any] = {}
        for k, v in call.args.items():
            ck = next((c for c, n in exp.items() if n == k), None)
            if ck is None:
                return self._err(
                    Rec(canon, False, "bad_args"), f"unknown argument '{k}'. Expected: {sorted(exp.values())}"
                )
            args[ck] = v
        if canon not in ("submit", "impossible"):
            if st.stale_at is not None and len(st.records) >= st.stale_at and canon != "view":
                st.stale_at, st.stale_pending, st.token = None, True, secrets.token_hex(3)
                return self._err(Rec(canon, False, "stale"), "STALE_STATE: the table was refreshed and your state token "
                                 "is no longer valid. View the table again before continuing.")  # fmt: skip
            roll = self.rng.random()
            if roll < self.p_timeout:
                return self._err(Rec(canon, False, "timeout"), "timeout: the data service did not respond; retry")
            if roll < self.p_timeout + self.p_malformed:
                st.records.append(Rec(canon, False, "malformed_response"))
                return StepOutcome('{"partial": tr', ok=False)
        if canon in ("compute", "submit"):
            if st.stale_pending:
                return self._err(Rec(canon, False, "acted_on_stale"), "STALE_STATE: view the table again first.")
            if str(args.get("state", "")) != st.token:
                return self._err(Rec(canon, False, "bad_token"), "invalid state token; view the table to get it")
        return getattr(self, f"_t_{canon}")(args)

    def _ok(self, canon: str, result: Any, **kw: Any) -> StepOutcome:
        self.st.records.append(Rec(canon, True))
        return StepOutcome(result, **kw)

    def _t_view(self, a: dict) -> StepOutcome:
        self.st.stale_pending = False
        rows, fmt = self.rows, self.st.fmt
        if fmt == "csv":
            body: Any = (
                "name,dept,city,salary,age\n"
                + "\n".join(f"{x['name']},{x['dept']},{x['city']},{x['salary']},{x['age']}" for x in rows)
                + f"\nstate: {self.st.token}"
            )
        elif fmt == "json":
            body = {"rows": rows, "state": self.st.token}
        else:
            body = "\n".join(" ".join(f"{k}={v}" for k, v in x.items()) for x in rows) + f"\nstate: {self.st.token}"
        return self._ok("view", body)

    def _t_query(self, a: dict) -> StepOutcome:
        f, v = str(a.get("field", "")), str(a.get("value", ""))
        if f not in FIELDS:
            return self._err(Rec("query", False, "bad_args"), f"unknown field '{f}'")
        names = [x["name"] for x in self.rows if str(x[f]) == v]
        return self._ok("query", {"count": len(names), "names": names})

    def _t_inspect(self, a: dict) -> StepOutcome:
        row = next((x for x in self.rows if x["name"] == str(a.get("name", "")).lower()), None)
        return self._ok("inspect", {"row": row})

    def _t_compute(self, a: dict) -> StepOutcome:
        op, f = str(a.get("op", "")), str(a.get("field", ""))
        base, _, dept = op.partition(":")
        rows = [x for x in self.rows if not dept or x["dept"] == dept]
        if f not in ("salary", "age") or base not in ("sum", "count", "max"):
            return self._err(Rec("compute", False, "bad_args"), "op must be sum|count|max[:dept], field salary|age")
        vals = [x[f] for x in rows]
        res = {"sum": sum(vals), "count": len(vals), "max": max(vals) if vals else None}[base]
        return self._ok("compute", {"op": op, "field": f, "result": res, "rows": len(vals)})

    def _t_submit(self, a: dict) -> StepOutcome:
        self.st.final_answer, self.st.decision = str(a.get("answer", "")).strip(), "answer"
        return self._ok("submit", {"submitted": self.st.final_answer}, done=True, decision="answer")

    def _t_impossible(self, a: dict) -> StepOutcome:
        self.st.final_answer, self.st.decision = "impossible", "impossible"
        return self._ok("impossible", {"acknowledged": True}, done=True, decision="impossible")

    def _t_expert(self, a: dict) -> StepOutcome:
        self.st.escalated = True
        return self._ok("expert", {"expert": self.expert.answer(self.task, self.rows)})

    def on_text(self, text: str) -> StepOutcome:
        self.st.records.append(Rec("?", False, "no_tool_call"))
        return StepOutcome("Reply with exactly one tool call.", ok=False)

    def finish(self, reason: str) -> StepOutcome:
        self.st.decision = self.st.decision or reason
        return StepOutcome({"ended": reason}, done=True, decision=self.st.decision)


def reward(task: TableTask, st: TState, cost: float = 0.3) -> tuple[float, bool, dict[str, float]]:
    if task.answer_type == "impossible":
        correct = st.decision == "impossible"
    else:
        ans = (st.final_answer or "").lower().replace(",", "").strip()
        correct = st.decision == "answer" and (ans == task.answer.lower() or ans.rstrip(".0") == task.answer)
    parts: dict[str, float] = {}
    if st.escalated:
        parts["escalation"] = (0.6 if correct else 0.0) - cost if task.hard else -cost
    else:
        parts["answer"] = 1.0 if correct else 0.0
    recs = st.records
    bad = sum(r.error in ("malformed_call", "unknown_tool", "bad_args", "no_tool_call") for r in recs)
    parts["malformed"] = -min(0.6, 0.2 * bad)
    # Ending with no decision must be the worst outcome. Otherwise RL learns that saying nothing is safe.
    if st.decision in (None, "budget"):
        parts["no_decision"] = -1.0
    if any(r.error == "acted_on_stale" for r in recs):
        parts["stale"] = -0.5
    fi = next((i for i in range(len(recs) - 1, -1, -1) if recs[i].canon in ("submit", "impossible")), None)
    if fi is not None:
        before = recs[:fi]
        need = {"sum": {"compute"}, "count": {"compute", "query"}, "max_name": {"view", "compute"},
                "lookup": {"inspect", "view"}, "missing": {"inspect", "query", "view"}}.get(task.kind)  # fmt: skip
        verified = st.escalated or (
            task.kind != "forecast" and any(r.ok and r.canon in (need or set()) for r in before)
        )
        if not verified:
            parts["unverified"] = -0.4
        if before and not before[-1].ok and before[-1].error in ("timeout", "malformed_response", "stale", "bad_token"):
            if not any(r.ok and r.canon == before[-1].canon for r in before[:-1]):
                parts["claim_after_error"] = -1.0
    return round(sum(parts.values()), 4), correct, parts


def episode_summary(env: TableEnv) -> dict[str, Any]:
    total, correct, parts = reward(env.task, env.st)
    return {"task_id": env.task.id, "kind": env.task.kind, "hard": env.task.hard, "correct": correct, "reward": total,
            "parts": parts, "escalated": env.st.escalated, "decision": env.st.decision, "calls": len(env.st.records)}  # fmt: skip
