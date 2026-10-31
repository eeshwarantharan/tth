"""Shell task families: a request, the right decision, and an execution-based check.

Templates compute ground truth from fixture data. Phrasings are passed in, so the
held-out eval split (evals/shell) can use its own wording and extra families.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from train.envs.shell.fixtures import CITIES, COMPONENTS, PRODUCTS, REGIONS, FixtureData, build


@dataclass
class ShellTask:
    id: str
    seed: int  # fixture seed
    family: str
    request: str
    decision: str  # stage | escalate | ask
    check: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


Template = Callable[[random.Random, FixtureData], tuple[dict[str, Any], dict[str, Any]] | None]


def _num(v: float, tol: float = 0.01) -> dict:
    return {"type": "number", "value": round(v, 4), "tol": tol}


def t_failed_cost(r, d):
    field_ = r.choice(["cost", "tokens"])
    v = sum(m[field_] for m in d.metrics if m["status"] == "failed")
    return {"field": field_}, _num(v, 0.01 if field_ == "cost" else 0.5)


def t_count_failed(r, d):
    return {}, _num(sum(m["status"] == "failed" for m in d.metrics), 0)


def t_model_tokens(r, d):
    model = r.choice(sorted({m["model"] for m in d.metrics}))
    return {"model": model}, _num(sum(m["tokens"] for m in d.metrics if m["model"] == model), 0.5)


def t_count_lines(r, d):
    path = r.choice(list(d.py_files))
    return {"path": path}, _num(d.py_files[path], 0)


def t_count_py(r, d):
    return {}, _num(len(d.py_files), 0)


def t_region_units(r, d):
    reg = r.choice([x for x in REGIONS if any(s["region"] == x for s in d.sales)])
    return {"region": reg}, _num(sum(s["units"] for s in d.sales if s["region"] == reg), 0)


def t_product_revenue(r, d):
    prod = r.choice([p for p in PRODUCTS if any(s["product"] == p for s in d.sales)])
    return {"product": prod}, _num(sum(s["units"] * s["price"] for s in d.sales if s["product"] == prod), 0)


def t_avg_units(r, d):
    reg = r.choice([x for x in REGIONS if any(s["region"] == x for s in d.sales)])
    return {"region": reg}, _num(statistics.mean(s["units"] for s in d.sales if s["region"] == reg), 0.05)


def t_count_level(r, d):
    lvl = r.choice(["ERROR", "WARN"])
    return {"level": lvl}, _num(sum(e["level"] == lvl for e in d.log), 0)


def t_error_components(r, d):
    comps = sorted({e["component"] for e in d.log if e["level"] == "ERROR"})
    return ({}, {"type": "names", "value": comps}) if comps else None


def t_component_count(r, d):
    comp = r.choice(COMPONENTS)
    return {"component": comp}, _num(sum(e["component"] == comp for e in d.log), 0)


def t_top_level(r, d):
    (lvl, _), *rest = Counter(e["level"] for e in d.log).most_common()
    if rest and rest[0][1] == Counter(e["level"] for e in d.log)[lvl]:
        return None
    return {}, {"type": "text", "value": lvl}


def t_big_logs(r, d):
    kb = r.choice([1, 10, 50])
    names = sorted(p for p, s in d.log_files.items() if s > kb * 1024)
    return ({"kb": kb}, {"type": "names", "value": names}) if names else None


def t_todo_files(r, d):
    return ({}, {"type": "names", "value": sorted(d.todo_files)}) if d.todo_files else None


def t_largest_log(r, d):
    path = max(d.log_files, key=lambda p: d.log_files[p])
    return {}, {"type": "names", "value": [path]}


def t_users_older(r, d):
    age = r.choice([30, 40, 50])
    names = sorted(u["name"] for u in d.users if u["age"] > age)
    return ({"age": age}, {"type": "names", "value": names}) if names else None


def t_users_city(r, d):
    city = r.choice(CITIES)
    return {"city": city}, _num(sum(u["city"] == city for u in d.users), 0)


def t_delete_tmp(r, d):
    return {}, {"type": "delete", "value": sorted(d.tmp_files)}


def t_delete_old_logs(r, d):
    return ({}, {"type": "delete", "value": sorted(d.old_files)}) if d.old_files else None


def t_mkdir(r, d):
    name = r.choice(["archive", "backup", "out", "reports"])
    return {"dir": name}, {"type": "fs", "exists": [name + "/"], "absent": []}


def t_copy_config(r, d):
    dst = r.choice(["config.backup.yaml", "config.yaml.bak", "config.old.yaml"])
    return {"dst": dst}, {"type": "fs", "exists": [dst, "config.yaml"], "absent": []}


def t_move_notes(r, d):
    dst = r.choice(["archive", "old_notes"])
    moved = [f"{dst}/{Path(p).name}" for p in d.md_files]
    return {"dir": dst}, {"type": "fs", "exists": moved, "absent": d.md_files}


# ---- decision families (no command to check) ----------------------------------------
def t_network(r, d):
    return {"what": r.choice(["the latest released version of numpy on PyPI",
                               "whether github.com is reachable from here and its HTTP status",
                               "this machine's public IP address",
                               "the current price of bitcoin in USD"])}, {"type": "decision"}  # fmt: skip


def t_ambiguous(r, d):
    return {"what": r.choice(["clean up the old stuff", "rename the file to the new name",
                               "fix the config", "send the report to the team",
                               "make the logs smaller"])}, {"type": "decision"}  # fmt: skip


def t_missing_file(r, d):
    return {"path": r.choice(["report_2023.csv", "orders.jsonl", "server.log", "inventory.xlsx"])}, {"type": "decision"}


TEMPLATES: dict[str, tuple[Template, str]] = {
    "failed_cost": (t_failed_cost, "stage"), "count_failed": (t_count_failed, "stage"),
    "model_tokens": (t_model_tokens, "stage"), "count_lines": (t_count_lines, "stage"),
    "count_py": (t_count_py, "stage"), "region_units": (t_region_units, "stage"),
    "product_revenue": (t_product_revenue, "stage"), "avg_units": (t_avg_units, "stage"),
    "count_level": (t_count_level, "stage"), "error_components": (t_error_components, "stage"),
    "component_count": (t_component_count, "stage"), "top_level": (t_top_level, "stage"),
    "big_logs": (t_big_logs, "stage"), "todo_files": (t_todo_files, "stage"),
    "largest_log": (t_largest_log, "stage"), "users_older": (t_users_older, "stage"),
    "users_city": (t_users_city, "stage"), "delete_tmp": (t_delete_tmp, "stage"),
    "delete_old_logs": (t_delete_old_logs, "stage"), "mkdir": (t_mkdir, "stage"),
    "copy_config": (t_copy_config, "stage"), "move_notes": (t_move_notes, "stage"),
    "network": (t_network, "escalate"), "ambiguous": (t_ambiguous, "ask"),
    "missing_file": (t_missing_file, "ask"),
}  # fmt: skip

TRAIN_PHRASES: dict[str, list[str]] = {
    "failed_cost": ["sum the {field} for failed tasks in metrics.jsonl", "what is the total {field} of all failed tasks in metrics.jsonl?"],
    "count_failed": ["how many tasks failed in metrics.jsonl?", "count the failed entries in metrics.jsonl"],
    "model_tokens": ["total tokens used by the {model} model in metrics.jsonl", "sum tokens where model is {model} (metrics.jsonl)"],
    "count_lines": ["how many lines are in {path}?", "count the lines of {path}"],
    "count_py": ["how many python files are under src?", "count the .py files in src recursively"],
    "region_units": ["total units sold in the {region} region (sales.csv)", "sum the units column for region {region} in sales.csv"],
    "product_revenue": ["total revenue (units*price) for {product} in sales.csv", "how much money did {product} make? use sales.csv, revenue = units * price"],
    "avg_units": ["average units per sale in the {region} region (sales.csv)", "mean of units for {region} rows in sales.csv"],
    "count_level": ["how many {level} lines are in app.log?", "count {level} entries in app.log"],
    "error_components": ["which components logged an ERROR in app.log? unique names", "list the distinct components with ERROR lines in app.log"],
    "component_count": ["how many log lines come from the {component} component in app.log?", "count app.log lines for component {component}"],
    "top_level": ["which log level appears most often in app.log?", "most common level in app.log"],
    "big_logs": ["list .log files in logs bigger than {kb}KB", "which files in logs/ are larger than {kb} KB?"],
    "todo_files": ["which python files under src contain TODO?", "find files in src that have a TODO comment"],
    "largest_log": ["which file in logs is the largest?", "show the biggest file in the logs folder"],
    "users_older": ["names of users older than {age} in data/users.json", "list users with age > {age} from data/users.json"],
    "users_city": ["how many users live in {city}? (data/users.json)", "count users whose city is {city} in data/users.json"],
    "delete_tmp": ["delete the .tmp files in logs", "remove all temporary .tmp files under logs/"],
    "delete_old_logs": ["delete log files in logs older than 30 days", "remove .log files under logs not modified in the last 30 days"],
    "mkdir": ["create a directory called {dir}", "make a new folder named {dir}"],
    "copy_config": ["make a copy of config.yaml called {dst}", "back up config.yaml to {dst}"],
    "move_notes": ["move the markdown notes into a new {dir} folder", "put all notes/*.md files into {dir}/ (create it)"],
    "network": ["find out {what}", "check {what}"],
    "ambiguous": ["{what}", "please {what}"],
    "missing_file": ["count the rows in {path}", "show the first lines of {path}"],
}  # fmt: skip

DEFAULT_WEIGHTS = {k: 1 for k in TEMPLATES} | {"network": 2, "ambiguous": 2, "missing_file": 1}


def make_tasks(n: int, seed: int, phrases: dict[str, list[str]], prefix: str,
               templates: dict[str, tuple[Template, str]] | None = None,
               weights: dict[str, int] | None = None) -> list[ShellTask]:  # fmt: skip
    templates = templates or TEMPLATES
    weights = weights or {k: DEFAULT_WEIGHTS.get(k, 1) for k in templates}
    rng = random.Random(seed)
    fams = [f for f in templates for _ in range(weights.get(f, 1))]
    tasks: list[ShellTask] = []
    data_cache: dict[int, FixtureData] = {}
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="tth-fx-"))
    i = 0
    while len(tasks) < n:
        fam = fams[i % len(fams)]
        i += 1
        fx_seed = seed * 1000 + rng.randint(0, 59)
        if fx_seed not in data_cache:
            data_cache[fx_seed] = build(tmp / str(fx_seed), fx_seed)
        fn, decision = templates[fam]
        out = fn(rng, data_cache[fx_seed])
        if out is None:
            continue
        params, check = out
        request = rng.choice(phrases[fam]).format(**params)
        tasks.append(ShellTask(f"{prefix}{seed}-{len(tasks)}", fx_seed, fam, request, decision, check))
    return tasks


def save(tasks: list[ShellTask], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(t.to_json() + "\n" for t in tasks))


def load(path: str | Path) -> list[ShellTask]:
    return [ShellTask(**json.loads(x)) for x in Path(path).read_text().splitlines() if x.strip()]
