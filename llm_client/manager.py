from __future__ import annotations

import os
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamChunk

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient

_CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}


class AsyncLLMManager:
    """Fachada: elige el proveedor por argumento o por la variable LLM_PROVIDER."""

    def __init__(
        self,
        provider: str | Provider | None = None,
        config: ModelConfig | None = None,
        api_key: str | None = None,
    ):
        raw = provider or os.getenv("LLM_PROVIDER", "openai")
        try:
            self.provider = Provider(str(raw).lower())
        except ValueError:
            valid = ", ".join(p.value for p in Provider)
            raise ValueError(f"Proveedor '{raw}' inválido. Opciones: {valid}") from None
        self.client = _CLIENTS[self.provider](api_key=api_key, config=config)

    @staticmethod
    def _to_messages(prompt: str | list[ChatMessage]) -> list[ChatMessage]:
        if isinstance(prompt, str):
            return [ChatMessage(role="user", content=prompt)]
        return prompt

    async def generate(self, prompt: str | list[ChatMessage]) -> ModelResponse:
        return await self.client.generate(self._to_messages(prompt))

    async def stream(self, prompt: str | list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async for chunk in self.client.stream(self._to_messages(prompt)):
            yield chunk
