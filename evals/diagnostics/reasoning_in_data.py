"""Did our training data ever contain a model's reasoning, or only its tool calls?

Counts the assistant turns with any text alongside their calls, in every verified training set. This is the evidence
that reasoning transfer hasn't been tested yet: the student has only ever been shown what to call, not why.

uv run python -m evals.diagnostics.reasoning_in_data evals/diagnostics/results/reasoning_in_data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATASETS = {
    "shell_teacher": "data/shell/teacher_verified.jsonl",
    "chess_teacher": "data/chess/teacher_verified.jsonl",
    "chess_scripted": "data/chess/scripted_verified.jsonl",
    "table_scripted": "data/table/scripted_verified.jsonl",
}


def count(path: Path) -> dict[str, float]:
    rows = turns = with_text = chars = 0
    for line in path.open():
        rows += 1
        for message in json.loads(line).get("messages", []):
            if message.get("role") != "assistant":
                continue
            text = (message.get("content") or "").strip()
            turns += 1
            with_text += bool(text)
            chars += len(text)
    return {
        "rows": rows, "assistant_turns": turns, "turns_with_text": with_text,
        "mean_chars_per_turn": round(chars / max(1, turns), 1),
    }  # fmt: skip


def main() -> None:
    results = {name: count(Path(path)) for name, path in DATASETS.items()}
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
