"""Keep only verifier-approved teacher trajectories: correct answer and zero penalties."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PENALTIES = ("malformed", "unverified", "stale", "claim_after_error")


def verified(row: dict) -> bool:
    s = row["summary"]
    if not s["correct"]:
        return False
    if any(s["parts"].get(k, 0) < 0 for k in PENALTIES):
        return False
    return any(m.get("tool_calls") for m in row["messages"])


def main(src: str, dst: str) -> None:
    rows = [json.loads(x) for x in Path(src).read_text().splitlines() if x.strip()]
    keep = [r for r in rows if verified(r)]
    default_teacher = "nvidia/nemotron-3.5-super-vl-120b-a12b"  # rows written before the teacher was recorded
    for r in keep:
        r["teacher"] = r.get("meta", {}).get("teacher", default_teacher)
    Path(dst).write_text(
        "".join(json.dumps({k: r[k] for k in ("messages", "tools", "summary", "teacher")}) + "\n" for r in keep)
    )
    key = "kind" if keep and "kind" in keep[0]["summary"] else "family"
    stats = {
        "raw": len(rows),
        "verified": len(keep),
        "rate": round(len(keep) / max(1, len(rows)), 3),
        "by_" + key: dict(Counter(r["summary"].get(key) for r in keep)),
        "by_teacher": dict(Counter(r["teacher"] for r in keep)),
    }
    print(json.dumps(stats))
    Path(dst).with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
