from __future__ import annotations

from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from schemas import ChatMessage, LLMError, Provider

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC
    default_model = "claude-haiku-4-5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = AsyncAnthropic(
            api_key=self._api_key, timeout=self.config.timeout, max_retries=0
        )

    def _params(self, messages: list[ChatMessage]) -> dict:
        # Anthropic recibe el system prompt aparte y no lo admite como mensaje.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        # La API solo acepta temperature en [0, 1]. Las versiones recientes del SDK
        # ya no la tipan como argumento, así que se envía por extra_body.
        body = {"temperature": min(self.config.temperature, 1.0)}
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        params = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
            "max_tokens": self.config.max_tokens,
            "extra_body": body,
        }
        if system:
            params["system"] = system
        return params

    async def _generate_once(self, messages):
        resp = await self._client.messages.create(**self._params(messages))
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, usage

    async def _stream_once(self, messages) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    def _classify(self, exc: Exception) -> LLMError:
        msg = str(exc)
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return LLMError(code="auth", message=msg)
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(code="rate_limit", message=msg, retryable=True)
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMError(code="timeout", message=msg, retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(code="network", message=msg, retryable=True)
        if isinstance(exc, anthropic.InternalServerError):  # incluye 529 overloaded
            return LLMError(code="server", message=msg, retryable=True)
        if isinstance(exc, anthropic.APIStatusError):
            return LLMError(code="bad_request", message=msg)
        return LLMError(code="unknown", message=f"{type(exc).__name__}: {msg}")
