"""Live progress bars for the notebook: reads runs/gpu0.log and runs/gpu1.log (JSON lines)."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _state(path: Path) -> dict:
    st: dict = {"job": "starting", "bar": "", "notes": [], "done": False, "error": None}
    if not path.exists():
        return st
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if "Traceback" in line or line.startswith(("Error", "RuntimeError", "torch.OutOfMemoryError")):
            st["error"] = line
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "round" in e:  # GRPO round summary
            st["notes"].append(f"GRPO round {e['round']}: mean_reward={e['mean_reward']} accuracy={e['accuracy']} "
                               f"train_rows={e['train_rows']}")  # fmt: skip
        if "step" in e and e.get("of"):
            pct = e["step"] / e["of"]
            extra = " ".join(f"{k}={e[k]}" for k in ("loss", "tok_s", "eta_min", "accuracy") if e.get(k) is not None)
            st["job"] = e.get("job", "?")
            st["bar"] = f"[{'█' * int(30 * pct)}{'·' * (30 - int(30 * pct))}] {e['step']}/{e['of']} {extra}"
        if e.get("note"):
            if e["note"] == "ALL DONE":
                st["done"] = True
            else:
                st["notes"].append(f"{e.get('job', '')}: {e['note']}")
    return st


def watch(logs: tuple[str, ...] = ("runs/gpu0.log", "runs/gpu1.log"), every: float = 10.0) -> None:
    from IPython.display import clear_output  # pyright: ignore[reportMissingImports]  (notebook only)

    t0 = time.time()
    while True:
        states = [_state(Path(p)) for p in logs]
        clear_output(wait=True)
        print(f"elapsed {int(time.time() - t0) // 60} min\n")
        for p, s in zip(logs, states, strict=True):
            print(f"── {p}  {'DONE' if s['done'] else s['job']}")
            print(f"   {s['bar']}")
            for n in s["notes"][-4:]:
                print(f"   · {n}")
            if s["error"]:
                print(f"   !! {s['error']}  (see {p})")
            print()
        if all(s["done"] or s["error"] for s in states):
            break
        time.sleep(every)
