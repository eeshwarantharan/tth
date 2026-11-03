"""OpenAI-compatible backend: NVIDIA API (teacher/expert) or a local llama-server."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any

from openai import APIError, OpenAI, RateLimitError

from tth.core.config import load_user_env
from tth.core.types import ModelTurn, ToolCall, ToolDef
from tth.inference.toolcalls import to_openai_messages

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


class RateLimiter:
    """Sliding-window limiter shared across threads (NVIDIA free tier: 40 rpm)."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.times: deque[float] = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.times and now - self.times[0] > 60:
                    self.times.popleft()
                if len(self.times) < self.rpm:
                    self.times.append(now)
                    return
                delay = 60 - (now - self.times[0]) + 0.05
            time.sleep(delay)


_LIMITER = RateLimiter(int(os.environ.get("TTH_NVIDIA_RPM", "36")))


class OpenAIPolicy:
    def __init__(
        self,
        model: str = NVIDIA_MODEL,
        base_url: str = NVIDIA_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 16384,
        thinking: bool = True,
        limiter: RateLimiter | None = _LIMITER,
    ) -> None:
        if api_key is None:
            load_user_env()
            api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=120)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.limiter = limiter if base_url == NVIDIA_BASE_URL else None
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, messages: list[dict[str, Any]], tools: list[ToolDef] | None = None) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=to_openai_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        if tools:
            kwargs["tools"] = [t.openai() for t in tools]
            kwargs["parallel_tool_calls"] = False
        for attempt in range(8):
            if self.limiter:
                self.limiter.wait()
            try:
                r = self.client.chat.completions.create(**kwargs)
                self.usage["calls"] += 1
                if r.usage:
                    self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
                    self.usage["completion_tokens"] += r.usage.completion_tokens or 0
                return r
            except RateLimitError:
                time.sleep(10 * (attempt + 1))
            except APIError:
                if attempt == 7:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("rate limited: gave up after 8 attempts")

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        r = self.complete(messages, tools)
        msg = r.choices[0].message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
                calls.append(ToolCall(name=tc.function.name, args=args, id=tc.id))
            except (json.JSONDecodeError, ValueError) as e:
                calls.append(ToolCall(name=tc.function.name, args={}, id=tc.id, malformed=str(e)))
        return ModelTurn(content=(msg.content or "").strip(), tool_calls=calls)
