#!/usr/bin/env python3
"""PreCompact hook: extract insights and save session state before compaction."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow importing the server package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from claude_memory.server.storage.session_db import SessionDB
from claude_memory.server.extraction.extractor import extract_insights, deduplicate_and_store
from claude_memory.server.utils.embeddings import get_embedding


def detect_namespace():
    cwd = os.environ.get("CLAUDE_CWD", os.getcwd())
    return Path(cwd).name.lower().replace(" ", "-")


def find_recent_transcript():
    """Find the most recent Claude transcript (excluding subagent logs)."""
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None
    transcripts = [
        t for t in claude_dir.rglob("*.jsonl")
        if "subagent" not in str(t)
    ]
    return max(transcripts, key=lambda p: p.stat().st_mtime) if transcripts else None


def _extract_text(msg_obj):
    """Extract text content from a transcript message object."""
    inner = msg_obj.get("message", {})
    content = inner.get("content", "")
    if isinstance(content, str):
        return inner.get("role", "unknown"), content
    if isinstance(content, list):
        texts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return inner.get("role", "unknown"), " ".join(texts)
    return inner.get("role", "unknown"), ""


def read_recent_messages(transcript_path, limit=20):
    """Read recent user/assistant message pairs from transcript."""
    raw = []
    with open(transcript_path) as f:
        for line in f:
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    pairs = []
    last_user = None
    for obj in raw:
        role, text = _extract_text(obj)
        if role == "user" and text.strip():
            last_user = text
        elif role == "assistant" and text.strip() and last_user:
            pairs.append((last_user, text))
            last_user = None

    return pairs[-limit:]


async def main():
    namespace = detect_namespace()
    session_id = datetime.now().strftime("%Y-%m-%d-%H%M")
    db = SessionDB()

    transcript_path = find_recent_transcript()
    if transcript_path:
        pairs = read_recent_messages(transcript_path)

        for user_msg, assistant_msg in pairs:
            user_msg = user_msg[:800]
            assistant_msg = assistant_msg[:800]

            if len(user_msg) < 20:
                continue

            insights = await extract_insights(
                message_summary=user_msg,
                response_summary=assistant_msg,
            )

            if insights:
                await deduplicate_and_store(
                    insights=insights,
                    namespace=namespace,
                    session_db=db,
                    get_embedding_fn=get_embedding,
                    session_id=session_id,
                )

    # Save session state
    db.save_session(session_id, namespace)
    db.increment_compaction(session_id)
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
