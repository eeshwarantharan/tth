"""Does a grammar generated from the tool schemas help? Rung L1 of the improvement ladder, on the GGUF build.

Same 20 held-out tasks, same verifier and same gate as gate_effect.py. The only difference between the two arms is
whether the MCP tool schemas constrain decoding. n=20 and one seed, so this is directional only.

TMPDIR=<scratch> uv run python -m evals.diagnostics.llama_grammar <out.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.diagnostics.gate_effect import run_task
from tth.core.config import data_dir, state_dir
from tth.inference.llamacpp import ensure_server, llama_policy
from tth_gym.envs.shell.tasks import load

GGUF = data_dir() / "models" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"


def main() -> None:
    out = Path(sys.argv[1])
    ensure_server(GGUF, state_dir() / "llama-server.log")
    tasks = load("evals/shell/tasks.jsonl")[::5]
    for arm, grammar in (("gguf_q4_no_grammar", False), ("gguf_q4_grammar", True)):
        policy = llama_policy(grammar=grammar)
        for i, task in enumerate(tasks):
            row = run_task(task, 10_000 + i * 5, policy)
            row.update(arm=arm, forced=grammar)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
