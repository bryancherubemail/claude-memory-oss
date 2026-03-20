"""
Memory extraction via local Ollama models.
Primary: gemma3:27b (best quality). Fallback: gemma3:latest (4b, faster).
All processing stays on-machine -- no external API calls.
"""

import hashlib
import json
from string import Template
from typing import Optional

import httpx

from ..config import (
    OLLAMA_URL,
    EXTRACTION_MODEL_PRIMARY,
    EXTRACTION_MODEL_FALLBACK,
    SIMILARITY_THRESHOLD,
    RELATED_THRESHOLD,
)

EXTRACTION_PROMPT = Template(
    'Extract ONLY actionable insights from this Claude Code interaction.\n\n'
    'RULES:\n'
    '- Max 25 words per item\n'
    '- Skip: greetings, confirmations, explanations of common knowledge\n'
    '- Keep: decisions, preferences, constraints, blockers, specific technical choices\n'
    '- Be specific: "Use PostgreSQL" not "Use a database"\n\n'
    'CATEGORIES (use exactly these labels):\n'
    '- DECISION: A choice made with brief rationale\n'
    '- FACT: A technical detail established (config, architecture, etc.)\n'
    '- PREFERENCE: User prefers X over Y\n'
    '- GOTCHA: Something that breaks unexpectedly\n'
    '- CONVENTION: A rule or pattern to follow\n'
    '- THREAD: An open/unresolved item (TODO, blocker, question)\n\n'
    'OUTPUT FORMAT (JSON array only, no markdown, no explanation):\n'
    '[{"category": "DECISION", "content": "...", "tags": ["tag1", "tag2"]}]\n\n'
    'If nothing worth extracting, output: []\n\n'
    '---\n'
    'USER MESSAGE SUMMARY:\n'
    '$message_summary\n\n'
    'CLAUDE RESPONSE SUMMARY:\n'
    '$response_summary\n'
    '---\n\n'
    'Extract:'
)


def compute_content_hash(content: str) -> str:
    """Generate hash for deduplication."""
    normalized = content.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _parse_json_from_text(text: str) -> list:
    """Extract JSON array from model output."""
    text = text.strip()

    # Direct parse
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    # Markdown code blocks
    for marker in ("```json", "```"):
        if marker in text:
            try:
                block = text.split(marker)[1].split("```")[0]
                result = json.loads(block.strip())
                return result if isinstance(result, list) else []
            except (json.JSONDecodeError, IndexError):
                pass

    # Find array bounds
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            pass

    return []


async def extract_insights(
    message_summary: str,
    response_summary: str,
    model: Optional[str] = None,
) -> list[dict]:
    """
    Use local Ollama to extract structured insights from an interaction.
    Tries primary model first, falls back to secondary.
    """
    prompt = EXTRACTION_PROMPT.substitute(
        message_summary=message_summary[:1000],
        response_summary=response_summary[:1000],
    )

    models_to_try = [model] if model else [EXTRACTION_MODEL_PRIMARY, EXTRACTION_MODEL_FALLBACK]

    async with httpx.AsyncClient() as client:
        for m in models_to_try:
            result = await _call_ollama(client, m, prompt)
            if result is not None:
                return result

    return []


async def _call_ollama(client: httpx.AsyncClient, model: str, prompt: str) -> list[dict] | None:
    """Call Ollama and parse the response."""
    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 600},
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            return None

        text = resp.json().get("response", "")
        parsed = _parse_json_from_text(text)

        # Validate structure
        valid = []
        for item in parsed:
            if isinstance(item, dict) and "content" in item and "category" in item:
                item["category"] = item["category"].lower()
                if "tags" not in item or not isinstance(item["tags"], list):
                    item["tags"] = []
                valid.append(item)

        return valid if valid else None  # None triggers fallback

    except (httpx.RequestError, json.JSONDecodeError, KeyError):
        return None


async def deduplicate_and_store(
    insights: list[dict],
    namespace: str,
    session_db,
    get_embedding_fn,
    session_id: str | None = None,
) -> dict:
    """
    Check for duplicates before storing. Creates relations for similar items.
    """
    stored = 0
    deduplicated = 0
    relations_created = 0

    for insight in insights:
        content = insight.get("content", "")
        if not content:
            continue

        content_hash = compute_content_hash(content)

        # Fast path: exact hash match
        existing = session_db.get_by_hash(content_hash, namespace)
        if existing:
            session_db.touch(existing["id"])
            deduplicated += 1
            continue

        # Get embedding for semantic comparison
        embedding = await get_embedding_fn(content)

        # Check semantic similarity
        if embedding:
            similar = session_db.find_similar(embedding, namespace, threshold=SIMILARITY_THRESHOLD, limit=1)
            if similar:
                session_db.touch(similar[0]["id"])
                deduplicated += 1
                continue

        # Store new memory
        new_id = session_db.insert_memory(
            namespace=namespace,
            content=content,
            content_hash=content_hash,
            category=insight.get("category", "fact"),
            tags=insight.get("tags", []),
            importance=5,
            session_id=session_id,
            embedding=embedding,
        )
        stored += 1

        # Find and create relations to similar (but not duplicate) memories
        if embedding:
            related = session_db.find_similar(embedding, namespace, threshold=RELATED_THRESHOLD, limit=5)
            for rel in related:
                if rel["id"] != new_id:
                    session_db.add_relation(
                        source_id=new_id,
                        target_id=rel["id"],
                        relation_type="related_to",
                        strength=rel.get("_vec_similarity", 0.7),
                    )
                    relations_created += 1

    return {"stored": stored, "deduplicated": deduplicated, "relations_created": relations_created}
