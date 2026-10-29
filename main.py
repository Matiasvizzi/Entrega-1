"""Script de validación: pregunta en modo normal y en streaming."""
import asyncio

from dotenv import load_dotenv

from llm_client import AsyncLLMManager
from schemas import ModelConfig

QUESTION = "¿Qué es la entropía?"


async def run(provider: str) -> None:
    print(f"\n===== {provider.upper()} =====")
    manager = AsyncLLMManager(provider, ModelConfig(temperature=0.5, max_tokens=300))

    print("[normal]")
    resp = await manager.generate(QUESTION)
    if resp.ok:
        print(resp.content)
        print(f"-- {resp.model} | tokens: {resp.usage}")
    else:
        print(f"Error controlado: {resp.error}")

    print("\n[streaming]")
    async for chunk in manager.stream(QUESTION):
        if chunk.error:
            print(f"\nError controlado: {chunk.error}")
        else:
            print(chunk.delta, end="", flush=True)
    print()


async def main() -> None:
    load_dotenv()
    # Un fallo en un proveedor (p. ej. key faltante) no rompe al otro.
    for provider in ("openai", "anthropic", "gemini"):
        await run(provider)


if __name__ == "__main__":
    asyncio.run(main())
