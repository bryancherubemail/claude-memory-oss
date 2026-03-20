"""Embedding generation via local Ollama (nomic-embed-text)."""

import httpx

from ..config import OLLAMA_URL, EMBEDDING_MODEL


async def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
    """Get embedding vector from Ollama. Returns None on failure."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": model, "input": text},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            embeddings = data.get("embeddings", [])
            return embeddings[0] if embeddings else None
        except (httpx.RequestError, KeyError, IndexError):
            return None


def get_embedding_sync(text: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
    """Synchronous version for use in non-async contexts."""
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": model, "input": text},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            embeddings = data.get("embeddings", [])
            return embeddings[0] if embeddings else None
    except (httpx.RequestError, KeyError, IndexError):
        return None
