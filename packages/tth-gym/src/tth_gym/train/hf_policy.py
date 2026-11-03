"""Batched Hugging Face policy for GPU rollouts (Colab). Same chat template and tool-call
format as the MLX product backend, so trajectories transfer directly."""

from __future__ import annotations

from typing import Any

import torch  # pyright: ignore[reportMissingImports]  (Colab-only dependency)

from tth.core.types import ModelTurn, ToolDef
from tth.inference.toolcalls import parse_qwen


class HFBatchPolicy:
    def __init__(self, model: Any, tokenizer: Any, temperature: float = 0.8, max_new_tokens: int = 256,
                 max_batch: int = 8) -> None:  # fmt: skip
        self.model = model
        self.tok = tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_batch = max_batch

    def render(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> str:
        return self.tok.apply_chat_template(messages, tools=[t.openai() for t in tools], add_generation_prompt=True,
                                            tokenize=False)  # fmt: skip

    @torch.no_grad()
    def chat_batch(self, batch: list[tuple[list[dict[str, Any]], list[ToolDef]]]) -> list[ModelTurn]:
        out: list[ModelTurn] = []
        prompts = [self.render(m, t) for m, t in batch]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        results: dict[int, ModelTurn] = {}
        for s in range(0, len(order), self.max_batch):
            idx = order[s : s + self.max_batch]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.model.device)
            gen = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5), top_p=0.95, pad_token_id=self.tok.pad_token_id,
            )  # fmt: skip
            new = gen[:, enc["input_ids"].shape[1] :]
            for j, i in enumerate(idx):
                # Keep special tokens: Qwen's <tool_call> tags ARE special tokens. Cut at end-of-turn instead.
                text = self.tok.decode(new[j], skip_special_tokens=False)
                for stop in ("<|im_end|>", "<|endoftext|>"):
                    text = text.split(stop)[0]
                if "</tool_call>" in text:
                    text = text[: text.index("</tool_call>") + len("</tool_call>")]
                results[i] = parse_qwen(text)
        out = [results[i] for i in range(len(prompts))]
        return out

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        return self.chat_batch([(messages, tools)])[0]
