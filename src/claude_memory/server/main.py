#!/usr/bin/env python3
"""
Claude Memory v2 MCP Server
Single-file SQLite backend with FTS5 + sqlite-vec. Fully local.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from .config import CATEGORIES, EMBEDDING_DIM
from .storage.session_db import SessionDB
from .storage.hot_cache import HotCache
from .storage.tiered import TieredRetrieval
from .extraction.extractor import extract_insights, deduplicate_and_store, compute_content_hash
from .utils.embeddings import get_embedding
from .utils.fusion import hybrid_search, compute_smart_alpha
from .utils.compression import format_results_compact, format_session_context
from .utils.prefetch import prefetch_context

# Initialize
app = Server("claude-memory-v2")
db = SessionDB()
cache = HotCache(max_size=100, ttl_minutes=10)
tiered = TieredRetrieval(cache, db)


def _detect_namespace(ns=None):
    """Detect namespace from argument or CWD."""
    if ns:
        return ns
    cwd = os.environ.get("CLAUDE_CWD", os.getcwd())
    return Path(cwd).name.lower().replace(" ", "-")


# -- Tool Definitions --

TOOL_DEFS = [
    Tool(
        name="search_memory",
        description="Search memories using hybrid BM25 + vector search. Use for finding past decisions, facts, conventions, gotchas.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "namespace": {"type": "string", "description": "Project namespace (auto-detected if omitted)"},
                "category": {"type": "string", "enum": list(CATEGORIES), "description": "Filter by category"},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
                "max_tokens": {"type": "integer", "default": 500, "description": "Token budget for response"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="recall_context",
        description="Get relevant context for current session. Use after compaction or at session start.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "depth": {"type": "string", "enum": ["minimal", "standard", "full"], "default": "standard"},
            },
        },
    ),
    Tool(
        name="add_memory",
        description="Store a memory (decision, fact, preference, gotcha, convention, or thread).",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content (max 500 chars recommended)"},
                "category": {"type": "string", "enum": list(CATEGORIES)},
                "namespace": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "relates_to": {"type": "array", "items": {"type": "integer"}, "description": "IDs of related memories"},
            },
            "required": ["content", "category"],
        },
    ),
    Tool(
        name="extract",
        description="Extract insights from a conversation exchange. Called by hooks or manually.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "message_summary": {"type": "string", "description": "User message summary"},
                "response_summary": {"type": "string", "description": "Claude response summary"},
                "session_id": {"type": "string"},
            },
            "required": ["message_summary", "response_summary"],
        },
    ),
    Tool(
        name="get_session",
        description="Get current session state for recovery after compaction.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "include_context": {"type": "boolean", "default": True},
            },
        },
    ),
    Tool(
        name="save_session",
        description="Save session state. Called before compaction.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "session_id": {"type": "string"},
                "focus": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="get_threads",
        description="List open threads (unresolved items) for a project.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "include_resolved": {"type": "boolean", "default": False},
            },
        },
    ),
    Tool(
        name="resolve_thread",
        description="Mark a thread as resolved.",
        inputSchema={
            "type": "object",
            "properties": {
                "thread_id": {"type": "integer"},
                "resolution": {"type": "string"},
            },
            "required": ["thread_id"],
        },
    ),
    Tool(
        name="get_relations",
        description="Get related memories via knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer"},
                "relation_types": {"type": "array", "items": {"type": "string"}},
                "depth": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="find_contradictions",
        description="Find memories that may contradict each other.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "recent_only": {"type": "boolean", "default": True},
            },
        },
    ),
    Tool(
        name="memory_stats",
        description="Get memory system statistics.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
            },
        },
    ),
    Tool(
        name="decay_old",
        description="Run importance decay on old unused memories.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "dry_run": {"type": "boolean", "default": True},
            },
        },
    ),
    Tool(
        name="update_memory",
        description="Update an existing memory's importance.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["memory_id"],
        },
    ),
]


@app.list_tools()
async def list_tools():
    return TOOL_DEFS


@app.call_tool()
async def call_tool(name, arguments):
    try:
        result = await _dispatch(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _dispatch(name, args):
    ns = _detect_namespace(args.get("namespace"))

    if name == "search_memory":
        return await _search_memory(ns, args)
    elif name == "recall_context":
        return await _recall_context(ns, args)
    elif name == "add_memory":
        return await _add_memory(ns, args)
    elif name == "extract":
        return await _extract(ns, args)
    elif name == "get_session":
        return _get_session(ns, args)
    elif name == "save_session":
        return _save_session(ns, args)
    elif name == "get_threads":
        return _get_threads(ns, args)
    elif name == "resolve_thread":
        return _resolve_thread(args)
    elif name == "get_relations":
        return _get_relations(args)
    elif name == "find_contradictions":
        return _find_contradictions(ns, args)
    elif name == "memory_stats":
        return _memory_stats(ns)
    elif name == "decay_old":
        return _decay_old(ns, args)
    elif name == "update_memory":
        return _update_memory(args)
    return {"error": f"Unknown tool: {name}"}


async def _search_memory(ns, args):
    query = args["query"]
    embedding = await get_embedding(query)
    alpha = compute_smart_alpha(query)

    results = await hybrid_search(
        query=query, namespace=ns, session_db=db,
        embedding=embedding, alpha=alpha, limit=args.get("limit", 10),
    )

    max_tokens = args.get("max_tokens", 500)
    formatted = format_results_compact(results, max_tokens=max_tokens)
    return {"results": formatted, "count": len(results), "alpha_used": round(alpha, 2), "namespace": ns}


async def _recall_context(ns, args):
    depth = args.get("depth", "standard")
    recall = await tiered.recall(query="", namespace=ns, depth=depth)

    threads = [r for r in recall["results"] if r.get("category") == "thread" and not r.get("resolved_at")]
    decisions = [r for r in recall["results"] if r.get("category") == "decision"]
    facts = [r for r in recall["results"] if r.get("category") not in ("thread", "decision")]

    context = format_session_context(threads, decisions, facts, ns)
    return {"context": context, "tiers_searched": recall["tiers_searched"], "tokens_used": recall["tokens_used"]}


async def _add_memory(ns, args):
    content = args["content"]
    content_hash = compute_content_hash(content)
    embedding = await get_embedding(content)

    existing = db.get_by_hash(content_hash, ns)
    if existing:
        db.touch(existing["id"])
        return {"id": existing["id"], "deduplicated": True, "message": "Existing memory updated"}

    memory_id = db.insert_memory(
        namespace=ns, content=content, content_hash=content_hash,
        category=args["category"], tags=args.get("tags"),
        importance=args.get("importance", 5), embedding=embedding,
    )

    relations = 0
    for rel_id in args.get("relates_to") or []:
        if db.add_relation(memory_id, rel_id, "related_to"):
            relations += 1

    cache.put(memory_id, db.get_by_id(memory_id))
    return {"id": memory_id, "deduplicated": False, "relations": relations}


async def _extract(ns, args):
    insights = await extract_insights(
        message_summary=args["message_summary"],
        response_summary=args["response_summary"],
    )
    if not insights:
        return {"extracted": 0, "message": "Nothing actionable found"}

    result = await deduplicate_and_store(
        insights=insights, namespace=ns, session_db=db,
        get_embedding_fn=get_embedding, session_id=args.get("session_id"),
    )
    return {"extracted": result["stored"], **result}


def _get_session(ns, args):
    session = db.get_latest_session(ns)
    if not session:
        return {"message": f"No session found for {ns}"}

    result = dict(session)
    if args.get("include_context", True):
        recent = db.get_recent(namespace=ns, days=1, limit=20)
        result["open_threads"] = [r for r in recent if r.get("category") == "thread" and not r.get("resolved_at")][:5]
        result["recent_decisions"] = [r for r in recent if r.get("category") == "decision"][:5]
    return result


def _save_session(ns, args):
    session_id = args["session_id"]
    state = json.dumps({"focus": args.get("focus"), "notes": args.get("notes"), "saved_at": datetime.now().isoformat()})
    db.save_session(session_id, ns, state_json=state)
    return {"saved": True, "session_id": session_id, "namespace": ns}


def _get_threads(ns, args):
    all_threads = []
    rows = db.conn.execute(
        "SELECT * FROM memories WHERE namespace = ? AND category = 'thread' AND archived = 0 ORDER BY created_at DESC",
        (ns,),
    ).fetchall()
    all_threads = [db._row_to_dict(r) for r in rows]

    if not args.get("include_resolved", False):
        all_threads = [t for t in all_threads if not t.get("resolved_at")]
    return {"threads": all_threads, "count": len(all_threads)}


def _resolve_thread(args):
    db.resolve_thread(args["thread_id"], args.get("resolution"))
    return {"resolved": True, "thread_id": args["thread_id"]}


def _get_relations(args):
    relations = db.get_relations(
        args["memory_id"],
        relation_types=args.get("relation_types"),
        depth=args.get("depth", 1),
    )
    source = db.get_by_id(args["memory_id"])
    return {"memory": source, "relations": relations}


def _find_contradictions(ns, args):
    contradictions = db.find_contradictions(
        namespace=ns if ns != "global" else None,
        recent_only=args.get("recent_only", True),
    )
    return {"contradictions": contradictions, "count": len(contradictions)}


def _memory_stats(ns):
    stats = db.get_stats(namespace=ns if ns != "global" else None)
    stats["hot_cache_size"] = cache.size
    stats["namespace"] = ns
    return stats


def _decay_old(ns, args):
    return db.decay_old(
        namespace=ns if ns != "global" else None,
        dry_run=args.get("dry_run", True),
    )


def _update_memory(args):
    if "importance" in args:
        db.update_importance(args["memory_id"], args["importance"])
    return {"updated": True, "memory_id": args["memory_id"]}


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
