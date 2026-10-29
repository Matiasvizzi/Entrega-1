"""Modelos Pydantic: mensajes de entrada, configuración del modelo y respuestas."""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de generación, validados antes de llegar a la API."""

    model: str | None = None  # None -> modelo por defecto del proveedor
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=128_000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0)  # backoff exponencial


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(BaseModel):
    """Error controlado: reemplaza a la excepción cruda del SDK."""

    code: Literal[
        "auth", "rate_limit", "network", "timeout", "bad_request", "server", "unknown"
    ]
    message: str
    retryable: bool = False
    attempts: int = 1


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str = ""
    usage: Usage | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Elemento del stream: un fragmento de texto o, al final, un error."""

    delta: str = ""
    error: LLMError | None = None
