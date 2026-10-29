"""Clase base abstracta con la lógica común de reintentos y manejo de errores."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import (
    ChatMessage,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
)


class BaseLLMClient(ABC):
    provider: Provider
    default_model: str

    def __init__(self, api_key: str | None = None, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = self.config.model or self.default_model
        self._api_key = api_key

    # --- Lo que implementa cada proveedor -----------------------------------
    @abstractmethod
    async def _generate_once(
        self, messages: list[ChatMessage]
    ) -> tuple[str, dict[str, int]]:
        """Una llamada no-streaming. Devuelve (texto, {input_tokens, output_tokens})."""

    @abstractmethod
    def _stream_once(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Una llamada streaming: async generator de fragmentos de texto."""

    @abstractmethod
    def _classify(self, exc: Exception) -> LLMError:
        """Traduce una excepción del SDK a un LLMError."""

    # --- API pública --------------------------------------------------------
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Nunca lanza por errores de API: los devuelve en `response.error`."""
        error: LLMError | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                text, usage = await self._generate_once(messages)
                return ModelResponse(
                    provider=self.provider,
                    model=self.model,
                    content=text,
                    usage=usage or None,
                )
            except Exception as exc:  # noqa: BLE001 - se clasifica abajo
                error = self._classify(exc).model_copy(update={"attempts": attempt})
                if not await self._should_retry(error, attempt):
                    break
        return ModelResponse(provider=self.provider, model=self.model, error=error)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Async generator de tokens. Reintenta solo si falla antes del primer token
        (después, repetir duplicaría texto); en error final emite un chunk con `error`."""
        for attempt in range(1, self.config.max_retries + 2):
            started = False
            try:
                async for delta in self._stream_once(messages):
                    started = True
                    yield StreamChunk(delta=delta)
                return
            except Exception as exc:  # noqa: BLE001
                error = self._classify(exc).model_copy(update={"attempts": attempt})
                if started or not await self._should_retry(error, attempt):
                    yield StreamChunk(error=error)
                    return

    async def _should_retry(self, error: LLMError, attempt: int) -> bool:
        if not error.retryable or attempt > self.config.max_retries:
            return False
        await asyncio.sleep(self.config.retry_base_delay * 2 ** (attempt - 1))
        return True
