from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from schemas import ChatMessage, LLMError, Provider

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    provider = Provider.GEMINI
    default_model = "gemini-3.8-flash"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # timeout en milisegundos; sin reintentos propios (los maneja BaseLLMClient)
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self.config.timeout * 1000)),
        )

    def _params(self, messages: list[ChatMessage]) -> dict:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            # Los tokens de "thinking" cuentan dentro de max_output_tokens y cortaban
            # la respuesta; se reduce para que max_tokens limite solo el texto.
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        )
        return {"model": self.model, "contents": contents, "config": config}

    async def _generate_once(self, messages):
        resp = await self._client.aio.models.generate_content(**self._params(messages))
        usage = {}
        if resp.usage_metadata:
            usage = {
                "input_tokens": resp.usage_metadata.prompt_token_count or 0,
                "output_tokens": resp.usage_metadata.candidates_token_count or 0,
            }
        return resp.text or "", usage

    async def _stream_once(self, messages) -> AsyncIterator[str]:
        stream = await self._client.aio.models.generate_content_stream(
            **self._params(messages)
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _classify(self, exc: Exception) -> LLMError:
        msg = str(exc)
        if isinstance(exc, errors.APIError):
            code = exc.code
            # Gemini responde 400 (no 401) cuando la API key es inválida.
            if code in (401, 403) or "API key" in msg:
                return LLMError(code="auth", message=msg)
            if code == 429:
                return LLMError(code="rate_limit", message=msg, retryable=True)
            if code in (408, 504):
                return LLMError(code="timeout", message=msg, retryable=True)
            if code >= 500:
                return LLMError(code="server", message=msg, retryable=True)
            return LLMError(code="bad_request", message=msg)
        if isinstance(exc, httpx.TimeoutException):
            return LLMError(code="timeout", message=msg, retryable=True)
        if isinstance(exc, httpx.TransportError):
            return LLMError(code="network", message=msg, retryable=True)
        return LLMError(code="unknown", message=f"{type(exc).__name__}: {msg}")
