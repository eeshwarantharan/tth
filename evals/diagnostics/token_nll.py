"""How well does each adapter predict the tool-call tags on its own training rows?

Finds where the training loss sits, per token class. MLX adapters on the 4-bit base. Results are written
as JSON and quoted by docs/analysis/2026-11-audit.md.

uv run python -m evals.diagnostics.token_nll evals/diagnostics/results/token_nll.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL = "models/qwen2.5-coder-1.5b-4bit"
ARMS = [
    ("A_base", None),
    ("B_shell", "adapters/B_shell/mlx"),
    ("C_chess_sft", "adapters/C_chess_sft/mlx"),
    ("C0_chess_grpo", "adapters/C0_chess_grpo/mlx"),
    ("C_chess_shell", "adapters/C_chess_shell/mlx"),
]
TAGS = {151657: "<tool_call>", 151658: "</tool_call>", 151645: "<|im_end|>"}
IM_START, IM_END = 151644, 151645
MAX_TOKENS = 1800  # keeps the 8 GB machine out of swap


def assistant_mask(ids: list[int], head: list[int]) -> list[int]:
    """Same span logic as train/sft/train_lora.py: the model's own turns, including <|im_end|>."""
    mask, i = [0] * len(ids), 0
    while i < len(ids):
        if ids[i] == IM_START and ids[i + 1 : i + 1 + len(head)] == head:
            j = i + 1 + len(head)
            while j < len(ids) and ids[j] != IM_END:
                mask[j] = 1
                j += 1
            if j < len(ids):
                mask[j] = 1
            i = j
        i += 1
    return mask


def measure(adapter: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = load(MODEL, adapter_path=adapter)
    model, wrapper = loaded[0], loaded[1]
    tok = wrapper._tokenizer
    head = tok("assistant\n", add_special_tokens=False)["input_ids"]
    nlls: list[float] = []
    by_tag: dict[str, list[float]] = {}
    fractions: list[float] = []
    for row in rows:
        text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_TOKENS:
            continue
        mask = assistant_mask(ids, head)
        fractions.append(sum(mask) / len(ids))
        logits = model(mx.array([ids]))[0]
        logprobs = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1)))
        for t in range(1, len(ids)):
            if mask[t]:
                nll = -float(logprobs[t - 1, ids[t]])
                nlls.append(nll)
                if ids[t] in TAGS:
                    by_tag.setdefault(TAGS[ids[t]], []).append(nll)
    del loaded, model, wrapper, tok
    gc.collect()
    mx.clear_cache()
    tags = {
        k: {"nll": round(float(np.mean(v)), 3), "p": float(np.exp(-np.mean(v))), "n": len(v)} for k, v in by_tag.items()
    }
    tag_share = sum(sum(by_tag[k]) for k in ("<tool_call>", "</tool_call>") if k in by_tag) / sum(nlls)
    return {
        "mean_nll": round(float(np.mean(nlls)), 3),
        "n_tokens": len(nlls),
        "trained_fraction": round(float(np.mean(fractions)), 3),
        "tool_call_tags_share_of_loss": round(tag_share, 3),
        "tags": tags,
    }


def main() -> None:
    lines = Path("data/shell/teacher_verified.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in lines[:5]]
    out = {"model": MODEL, "data": "data/shell/teacher_verified.jsonl (first 5 rows)", "arms": {}}
    for arm, adapter in ARMS:
        out["arms"][arm] = measure(adapter, rows)
        print(arm, json.dumps(out["arms"][arm]), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
