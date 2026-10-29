# Usa el almacén de certificados del sistema: en redes con proxy/antivirus que
# interceptan TLS (común en Windows), certifi falla con CERTIFICATE_VERIFY_FAILED.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .manager import AsyncLLMManager
from .openai_client import OpenAIClient

__all__ = [
    "AnthropicClient",
    "AsyncLLMManager",
    "BaseLLMClient",
    "GeminiClient",
    "OpenAIClient",
]
