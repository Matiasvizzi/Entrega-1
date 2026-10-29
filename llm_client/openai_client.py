from __future__ import annotations

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from schemas import ChatMessage, LLMError, Provider

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI
    default_model = "gpt-4o-mini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # max_retries=0: los reintentos los controla BaseLLMClient
        self._client = AsyncOpenAI(
            api_key=self._api_key, timeout=self.config.timeout, max_retries=0
        )

    def _params(self, messages: list[ChatMessage]) -> dict:
        params = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        return params

    async def _generate_once(self, messages):
        resp = await self._client.chat.completions.create(**self._params(messages))
        usage = {}
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return resp.choices[0].message.content or "", usage

    async def _stream_once(self, messages) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._params(messages), stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _classify(self, exc: Exception) -> LLMError:
        msg = str(exc)
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return LLMError(code="auth", message=msg)
        if isinstance(exc, openai.RateLimitError):
            return LLMError(code="rate_limit", message=msg, retryable=True)
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(code="timeout", message=msg, retryable=True)
        if isinstance(exc, openai.APIConnectionError):
            return LLMError(code="network", message=msg, retryable=True)
        if isinstance(exc, openai.InternalServerError):
            return LLMError(code="server", message=msg, retryable=True)
        if isinstance(exc, openai.APIStatusError):
            return LLMError(code="bad_request", message=msg)
        return LLMError(code="unknown", message=f"{type(exc).__name__}: {msg}")
