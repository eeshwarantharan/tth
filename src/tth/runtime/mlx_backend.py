"""Local MLX backend for Qwen2.5-Coder (optionally with a LoRA adapter).

Keeps one KV cache across the turns of an episode and only feeds new tokens,
which matters on an 8 GB M2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tth.agent.protocol import ModelTurn, ToolDef
from tth.runtime.toolcalls import parse_qwen

DEFAULT_MODEL = os.environ.get(
    "TTH_MODEL", str(Path(__file__).resolve().parents[3] / "models" / "qwen2.5-coder-1.5b-4bit")
)


class MLXPolicy:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        adapter_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        from mlx_lm import load

        loaded = load(model_path, adapter_path=adapter_path)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache: Any = None
        self._cache_tokens: list[int] = []

    def render(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, tools=[t.openai() for t in tools], add_generation_prompt=True, tokenize=True
        )

    def reset(self) -> None:
        self._cache = None
        self._cache_tokens = []

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        prompt = list(self.render(messages, tools))
        k = 0
        if self._cache is not None:
            limit = min(len(prompt), len(self._cache_tokens))
            while k < limit and prompt[k] == self._cache_tokens[k]:
                k += 1
            if k == len(prompt):
                k -= 1
            drop = len(self._cache_tokens) - k
            if k <= 0 or not can_trim_prompt_cache(self._cache):
                self._cache, k = None, 0
            elif drop:
                trim_prompt_cache(self._cache, drop)
        if self._cache is None:
            self._cache = make_prompt_cache(self.model)
            k = 0

        sampler = make_sampler(temp=self.temperature, top_p=0.95 if self.temperature else 1.0)
        text, toks = "", []
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt[k:],
            max_tokens=self.max_tokens,
            sampler=sampler,
            prompt_cache=self._cache,
        ):
            text += resp.text
            toks.append(resp.token)
            if "</tool_call>" in text or "<|im_end|>" in text or "</tools>" in text:
                break
            if text.count("```") >= 2 and "{" in text:
                break
            if "</function>" in text and ("</arguments>" in text or "<arguments>" not in text):
                break
        self._cache_tokens = prompt + toks
        if self._cache[0].offset != len(self._cache_tokens):
            self.reset()
        return parse_qwen(text)
