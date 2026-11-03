"""Does forcing the tool-call wrapper revive the adapters recorded at 0%?

Same adapters, no retraining. Each reply starts with a forced `<tool_call>\\n{"name": "`, and the first JSON
object is parsed. This approximates constrained decoding; it is not a full grammar. Uses every 5th held-out
shell task with the real ShellEnv, verifier and sandbox. n=20 per arm, so this is directional only and does
not replace evals/results/RESULTS.md.

TMPDIR=<scratch> uv run python -m evals.diagnostics.forced_wrapper <out.jsonl>
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from typing import Any

import mlx.core as mx
import tth.inference.mlx as mb
from tth.agent.loop import run_episode
from tth.core.types import ModelTurn, ToolCall, ToolDef
from tth.inference.toolcalls import parse_qwen
from tth_gym.envs.shell.env import ShellEnv, episode_summary
from tth_gym.envs.shell.tasks import load

MODEL = "models/qwen2.5-coder-1.5b-4bit"
PREFIX = '<tool_call>\n{"name": "'
JUNK = re.compile(r"<\|im_start\|>(?:assistant)?|<\|im_end\|>|<\|endoftext\|>|</tool_call>")
ARMS = [
    ("A_base", None, False),
    ("A_base", None, True),
    ("B_shell", "adapters/B_shell/mlx", True),
    ("C0_chess", "adapters/C0_chess_grpo/mlx", True),
    ("C_chess_shell", "adapters/C_chess_shell/mlx", True),
    ("D_table_shell", "adapters/D_table_shell/mlx", True),
]


def forced_parse(text: str) -> ModelTurn:
    """First JSON object after the forced prefix. Anything after it is ignored, as a grammar would enforce."""
    body = JUNK.sub("", '{"name": "' + text)
    try:
        obj, _ = json.JSONDecoder().raw_decode(body.strip())
        if isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            return ModelTurn(content="", tool_calls=[ToolCall(name=obj["name"], args=obj["arguments"])], raw=text)
    except (json.JSONDecodeError, AttributeError):
        pass
    return ModelTurn(content="", tool_calls=[ToolCall(name="", args={}, malformed="unparseable")], raw=text)


class ForcedPolicy(mb.MLXPolicy):
    def render(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> list[int]:
        prefix = self.tokenizer._tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
        return list(super().render(messages, tools)) + prefix


def main() -> None:
    out = sys.argv[1]
    tasks = load("evals/shell/tasks.jsonl")[::5]
    for arm, adapter, forced in ARMS:
        mb.parse_qwen = forced_parse if forced else parse_qwen
        policy = (ForcedPolicy if forced else mb.MLXPolicy)(model_path=MODEL, adapter_path=adapter, temperature=0.0)
        for i, task in enumerate(tasks):
            env = ShellEnv(task, seed=10_000 + i * 5, p_timeout=0)
            policy.reset()
            t0 = time.time()
            episode = run_episode(env, policy, budget=6)
            row = episode_summary(env)
            row.update(arm=arm, forced=forced, seconds=round(time.time() - t0, 1),
                       tools=[e.call.tool for e in episode.events if e.call])  # fmt: skip
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        del policy
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
