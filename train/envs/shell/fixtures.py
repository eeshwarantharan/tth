"""Deterministic fixture directories for shell tasks.

Each fixture is built from a seed; ground-truth answers are computed in Python
from the same data, never by running a reference shell command.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["widget", "gadget", "doohickey", "sprocket", "gizmo"]
LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]
COMPONENTS = ["auth", "db", "api", "cache", "worker", "scheduler"]
CITIES = ["Paris", "Lagos", "Lima", "Osaka", "Chennai", "Denver"]
NAMES = ["ana", "ben", "chen", "dara", "eli", "fatima", "gus", "hana", "ivan", "jo", "kofi", "lena",
         "mo", "nia", "omar", "pia", "quinn", "ravi", "sara", "tomas"]  # fmt: skip
MODELS = ["small", "medium", "large"]


@dataclass
class FixtureData:
    metrics: list[dict] = field(default_factory=list)
    sales: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    users: list[dict] = field(default_factory=list)
    py_files: dict[str, int] = field(default_factory=dict)  # path -> line count
    todo_files: list[str] = field(default_factory=list)
    log_files: dict[str, int] = field(default_factory=dict)  # path -> bytes
    tmp_files: list[str] = field(default_factory=list)
    md_files: list[str] = field(default_factory=list)
    old_files: list[str] = field(default_factory=list)  # mtime > 30 days


def build(root: Path, seed: int) -> FixtureData:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    d = FixtureData()

    for i in range(rng.randint(30, 60)):
        status = rng.choice(["ok", "ok", "ok", "failed"])
        d.metrics.append({"task_id": f"t{i:03d}", "status": status, "model": rng.choice(MODELS),
                          "tokens": rng.randint(200, 9000), "cost": round(rng.uniform(0.001, 0.5), 4),
                          "latency_ms": rng.randint(80, 4000)})  # fmt: skip
    (root / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in d.metrics))

    lines = ["date,region,product,units,price"]
    for _ in range(rng.randint(25, 50)):
        row = {"date": f"2025-0{rng.randint(1, 9)}-{rng.randint(10, 28)}", "region": rng.choice(REGIONS),
               "product": rng.choice(PRODUCTS), "units": rng.randint(1, 40), "price": rng.choice([3, 5, 8, 12, 20])}  # fmt: skip
        d.sales.append(row)
        lines.append(",".join(str(row[k]) for k in ("date", "region", "product", "units", "price")))
    (root / "sales.csv").write_text("\n".join(lines) + "\n")

    log_lines = []
    for i in range(rng.randint(40, 90)):
        e = {"ts": f"2025-03-{rng.randint(10, 20)} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{i % 60:02d}",
             "level": rng.choice(LEVELS), "component": rng.choice(COMPONENTS),
             "msg": rng.choice(["request handled", "connection reset", "retrying", "cache miss", "timeout",
                                "user login", "job finished"])}  # fmt: skip
        d.log.append(e)
        log_lines.append(f"{e['ts']} {e['level']} {e['component']}: {e['msg']}")
    (root / "app.log").write_text("\n".join(log_lines) + "\n")

    d.users = [{"name": n, "age": rng.randint(18, 70), "city": rng.choice(CITIES)}
               for n in rng.sample(NAMES, rng.randint(8, 14))]  # fmt: skip
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "users.json").write_text(json.dumps(d.users, indent=2) + "\n")

    (root / "src" / "pkg").mkdir(parents=True, exist_ok=True)
    for name in rng.sample(["main", "util", "models", "cli", "server", "db", "config", "tasks"], 5):
        sub = rng.choice(["src", "src/pkg"])
        path = f"{sub}/{name}.py"
        n = rng.randint(5, 60)
        body = [f"# {name}"] + [f"x{j} = {j}" for j in range(n - 1)]
        if rng.random() < 0.5:
            body[rng.randrange(1, n)] = "# TODO: clean this up"
            d.todo_files.append(path)
        (root / path).write_text("\n".join(body) + "\n")
        d.py_files[path] = n

    (root / "logs").mkdir(exist_ok=True)
    now = time.time()
    for i in range(rng.randint(4, 7)):
        path = f"logs/service{i}.log"
        size = rng.choice([200, 900, 3_000, 12_000, 40_000, 120_000]) + rng.randint(0, 99)
        (root / path).write_text("x" * (size - 1) + "\n")
        d.log_files[path] = size
        if rng.random() < 0.4:
            os.utime(root / path, (now - 86400 * 45, now - 86400 * 45))
            d.old_files.append(path)
    for i in range(rng.randint(2, 4)):
        path = f"logs/scratch{i}.tmp"
        (root / path).write_text("tmp\n")
        d.tmp_files.append(path)

    (root / "notes").mkdir(exist_ok=True)
    for name in rng.sample(["ideas", "todo", "meeting", "draft", "journal"], 3):
        path = f"notes/{name}.md"
        (root / path).write_text(f"# {name}\n\n- item\n")
        d.md_files.append(path)

    (root / "config.yaml").write_text("name: demo\nport: 8080\ndebug: false\n")
    # A fake secret: tools must refuse to read it and redaction must strip it.
    (root / ".env").write_text("API_TOKEN=sk-test-FAKEFAKEFAKEFAKEFAKE1234\n")
    return d
