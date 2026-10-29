# Unified Async LLM Client

Cliente asíncrono (Python 3.12) con interfaz común para **OpenAI** y **Anthropic**: `generate()` normal, `stream()` token a token, validación con Pydantic y errores controlados (sin crashes).

## Estructura
```
schemas.py                      # ChatMessage, ModelConfig, ModelResponse, StreamChunk, LLMError
llm_client/base.py              # BaseLLMClient (abstracta): reintentos + manejo de errores
llm_client/openai_client.py     # AsyncOpenAI
llm_client/anthropic_client.py  # AsyncAnthropic
llm_client/gemini_client.py     # google-genai (client.aio)
llm_client/manager.py           # AsyncLLMManager: elige proveedor por config
main.py                         # script de prueba (normal + streaming)
```

## Instalación
```bash
python3.12 -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env  -> completar las keys
```

## Variables de entorno
| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | API key de OpenAI (necesaria para el proveedor `openai`) |
| `ANTHROPIC_API_KEY` | API key de Anthropic (necesaria para `anthropic`) |
| `GOOGLE_API_KEY` | API key de Google AI Studio (https://aistudio.google.com/apikey), para `gemini` |
| `LLM_PROVIDER` | `openai` (default), `anthropic` o `gemini`; se usa cuando no se pasa `provider` al manager |

## Ejecutar la prueba
```bash
python main.py
```
Pregunta "¿Qué es la entropía?" a ambos proveedores, primero en modo normal y luego en streaming. Si falta una key, ese proveedor muestra un error controlado y el script sigue con el otro.

## Uso
```python
manager = AsyncLLMManager("anthropic", ModelConfig(temperature=0.3, max_tokens=500))

resp = await manager.generate("Hola")           # str o list[ChatMessage]
print(resp.content if resp.ok else resp.error)

async for chunk in manager.stream("Hola"):      # async generator (yield)
    print(chunk.delta, end="")
```

## Decisiones de diseño
- **No bloqueante:** solo `AsyncOpenAI` / `AsyncAnthropic` con `await`; el streaming usa `async for` + `yield`.
- **Validación:** `ModelConfig` exige `temperature` ∈ [0, 2], `max_tokens` > 0, `top_p` ∈ (0, 1], etc. Un valor inválido lanza `ValidationError` al construirlo. (Anthropic solo admite temperatura ≤ 1: se recorta automáticamente.)
- **Errores controlados:** auth, rate limit, red, timeout y 5xx se traducen a `LLMError(code, message, retryable, attempts)`. `generate()` lo devuelve en `response.error`; `stream()` emite al final un `StreamChunk(error=...)`.
- **Reintentos:** backoff exponencial (`max_retries`, `retry_base_delay`) para rate limit, red, timeout y 5xx; no se reintentan errores de auth o de petición. En streaming solo se reintenta si falló antes del primer token, para no duplicar texto. Los reintentos internos de los SDKs se desactivan (`max_retries=0`) para que haya un único control.
- **Modelos por defecto:** `gpt-4o-mini`, `claude-haiku-4-5` y `gemini-3.8-flash`; se cambian con `ModelConfig(model="...")`.
