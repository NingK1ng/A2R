"""Any OpenAI-compatible chat endpoint (OpenAI, vLLM, DeepSeek, Qwen, ...)."""
from __future__ import annotations

from .arbiter import LLM


def openai_chat(model: str, base_url: str | None = None, api_key: str | None = None, temperature: float | None = 0.0) -> LLM:
    """The returned callable keeps running token totals in `chat.usage`; the arbiter reports per-case deltas."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    extra = {} if temperature is None else {"temperature": temperature}

    def chat(prompt: str) -> str:
        reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **extra)
        if reply.usage is not None:
            chat.usage["prompt_tokens"] += reply.usage.prompt_tokens
            chat.usage["completion_tokens"] += reply.usage.completion_tokens
        return reply.choices[0].message.content or ""

    chat.usage = {"prompt_tokens": 0, "completion_tokens": 0}
    return chat
