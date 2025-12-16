#!/usr/bin/env python3
"""
Claude Memory MCP Server
Local long-term memory for Claude Code via MCP protocol.

Stores memories in SQLite with optional semantic search via ChromaDB.
"""

import asyncio
import hashlib
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Configuration - override with environment variables
DATA_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
MEMORY_DB = DATA_DIR / "data" / "memory.db"
SESSIONS_DIR = DATA_DIR / "sessions"
CHROMA_URL = os.environ.get("CHROMA_URL", "http://localhost:8000")
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "http://localhost:8080")
COLLECTION_NAME = "claude_memories"
GLOBAL_PROJECT = "global"


def detect_project() -> str:
    """Detect current project from git repo name or directory."""
    try:
        cwd = Path.cwd()

        # ~/Claude is global context
        if cwd == Path.home() / "Claude":
            return GLOBAL_PROJECT

        # Try git repo name
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=1, cwd=cwd
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).name

        # Fallback to directory name
        return cwd.name if cwd.name else GLOBAL_PROJECT

    except Exception:
        return GLOBAL_PROJECT


class MemoryStore:
    """SQLite + optional ChromaDB memory storage."""

    def __init__(self):
        self.db_path = MEMORY_DB
        self.chroma_url = CHROMA_URL
        self.embeddings_url = EMBEDDINGS_URL
        self.http_client = None
        self.project = detect_project()
        self.chroma_available = False

    async def initialize(self):
        """Initialize database and optional vector store."""
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_db()

        # Try to connect to ChromaDB (optional)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        try:
            await self._init_chroma()
            self.chroma_available = True
            print(f"ChromaDB available at {self.chroma_url}", file=sys.stderr)
        except Exception as e:
            print(f"ChromaDB not available (using SQLite only): {e}", file=sys.stderr)

    def _init_db(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                relevance_score REAL DEFAULT 1.0,
                project TEXT DEFAULT 'global'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON memories(project)")
        conn.commit()
        conn.close()

    async def _init_chroma(self):
        """Initialize ChromaDB collection."""
        response = await self.http_client.get(
            f"{self.chroma_url}/api/v2/collections/{COLLECTION_NAME}"
        )
        if response.status_code != 200:
            await self.http_client.post(
                f"{self.chroma_url}/api/v2/collections",
                json={"name": COLLECTION_NAME, "metadata": {"description": "Claude memories"}}
            )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector from local service."""
        if not self.chroma_available:
            return []
        try:
            response = await self.http_client.post(
                f"{self.embeddings_url}/embed",
                json={"inputs": text}
            )
            embeddings = response.json()
            return embeddings[0] if embeddings else []
        except Exception:
            return []

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search memories - semantic if available, else text."""
        if self.chroma_available:
            results = await self._search_semantic(query, limit)
            if results:
                return results
        return self._search_text(query, limit)

    async def _search_semantic(self, query: str, limit: int) -> list[dict]:
        """Semantic search via ChromaDB."""
        try:
            embedding = await self.get_embedding(query)
            if not embedding:
                return []

            response = await self.http_client.post(
                f"{self.chroma_url}/api/v2/collections/{COLLECTION_NAME}/query",
                json={"query_embeddings": [embedding], "n_results": limit * 3}
            )
            results = response.json()

            memories = []
            if results.get("ids") and results["ids"][0]:
                for i, mem_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    project = metadata.get("project", "global")

                    # Only current project + global
                    if project in (self.project, "global"):
                        relevance = 1.0 - distance
                        if project == self.project:
                            relevance += 0.2  # Boost current project
                        memories.append({
                            "id": mem_id,
                            "content": metadata.get("content", ""),
                            "type": metadata.get("type", ""),
                            "timestamp": metadata.get("timestamp", ""),
                            "project": project,
                            "relevance": min(1.0, relevance),
                        })

            memories.sort(key=lambda x: x["relevance"], reverse=True)
            return memories[:limit]
        except Exception:
            return []

    def _search_text(self, query: str, limit: int) -> list[dict]:
        """Text search via SQLite."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT id, content, type, timestamp, context, relevance_score, project
            FROM memories
            WHERE (content LIKE ? OR context LIKE ?)
              AND (project = ? OR project = 'global')
            ORDER BY
                CASE WHEN project = ? THEN 2 WHEN project = 'global' THEN 1 ELSE 0 END DESC,
                relevance_score DESC, timestamp DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", self.project, self.project, limit))

        results = [{
            "id": row["id"],
            "content": row["content"],
            "type": row["type"],
            "timestamp": row["timestamp"],
            "project": row["project"] or "global",
            "relevance": row["relevance_score"],
        } for row in cursor.fetchall()]

        conn.close()
        return results

    async def get_recent(self, days: int = 7, limit: int = 20) -> list[dict]:
        """Get recent memories."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor = conn.execute("""
            SELECT id, content, type, timestamp, context, relevance_score, project
            FROM memories
            WHERE timestamp > ? AND (project = ? OR project = 'global')
            ORDER BY
                CASE WHEN project = ? THEN 2 WHEN project = 'global' THEN 1 ELSE 0 END DESC,
                timestamp DESC
            LIMIT ?
        """, (cutoff, self.project, self.project, limit))

        results = [{
            "id": row["id"],
            "content": row["content"],
            "type": row["type"],
            "timestamp": row["timestamp"],
            "project": row["project"] or "global",
            "relevance": row["relevance_score"],
        } for row in cursor.fetchall()]

        conn.close()
        return results

    async def add(self, content: str, memory_type: str = "learning", context: str = "") -> str:
        """Add a memory."""
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()
        timestamp = datetime.now().isoformat()

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO memories (id, timestamp, type, content, context, relevance_score, project) VALUES (?, ?, ?, ?, ?, 1.0, ?)",
            (memory_id, timestamp, memory_type, content, context, self.project)
        )
        conn.commit()
        conn.close()

        # Add to ChromaDB if available
        if self.chroma_available:
            try:
                embedding = await self.get_embedding(content)
                if embedding:
                    await self.http_client.post(
                        f"{self.chroma_url}/api/v2/collections/{COLLECTION_NAME}/add",
                        json={
                            "ids": [memory_id],
                            "embeddings": [embedding],
                            "metadatas": [{
                                "content": content,
                                "type": memory_type,
                                "timestamp": timestamp,
                                "context": context,
                                "project": self.project,
                            }]
                        }
                    )
            except Exception:
                pass

        return memory_id

    async def stats(self) -> dict:
        """Get memory statistics."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        by_type = {}
        for row in conn.execute("SELECT type, COUNT(*) as count FROM memories GROUP BY type"):
            by_type[row["type"]] = row["count"]

        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = conn.execute("SELECT COUNT(*) FROM memories WHERE timestamp > ?", (cutoff,)).fetchone()[0]

        conn.close()
        return {
            "total_memories": total,
            "by_type": by_type,
            "recent_7_days": recent,
            "database_path": str(self.db_path),
            "chroma_available": self.chroma_available,
        }

    async def sync_to_chroma(self) -> int:
        """Sync all SQLite memories to ChromaDB."""
        if not self.chroma_available:
            raise Exception("ChromaDB not available")

        conn = self._get_conn()
        cursor = conn.execute("SELECT id, content, type, timestamp, context, project FROM memories")

        count = 0
        for row in cursor.fetchall():
            try:
                embedding = await self.get_embedding(row["content"])
                if embedding:
                    await self.http_client.post(
                        f"{self.chroma_url}/api/v2/collections/{COLLECTION_NAME}/add",
                        json={
                            "ids": [row["id"]],
                            "embeddings": [embedding],
                            "metadatas": [{
                                "content": row["content"],
                                "type": row["type"],
                                "timestamp": row["timestamp"],
                                "context": row["context"] or "",
                                "project": row["project"] or "global",
                            }]
                        }
                    )
                    count += 1
            except Exception:
                pass

        conn.close()
        return count

    async def get_session(self, include_learnings: bool = True, hours: int = 24) -> str:
        """Get session context."""
        formatted = f"# Session Context: {self.project}\n\n"

        # Read session file if exists
        session_file = SESSIONS_DIR / f"{self.project}.md"
        if session_file.exists():
            try:
                formatted += session_file.read_text() + "\n\n"
            except Exception:
                pass

        # Include recent learnings
        if include_learnings:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            conn = self._get_conn()
            cursor = conn.execute("""
                SELECT content, type, timestamp FROM memories
                WHERE timestamp > ? AND (project = ? OR project = 'global')
                ORDER BY timestamp DESC LIMIT 10
            """, (cutoff, self.project))

            rows = cursor.fetchall()
            conn.close()

            if rows:
                formatted += f"## Recent Learnings (Last {hours}h)\n\n"
                for row in rows:
                    content = row['content'][:200] + '...' if len(row['content']) > 200 else row['content']
                    formatted += f"- **{row['type']}**: {content}\n"

        return formatted

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()


# MCP Server
app = Server("claude-memory")
store = MemoryStore()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_memory",
            description="Search memories using semantic similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_recent_learnings",
            description="Get recent memories from the last N days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="add_learning",
            description="Store a new learning or decision in memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The learning to remember"},
                    "category": {
                        "type": "string",
                        "enum": ["decisions", "conventions", "tech_stack", "gotchas", "solutions", "general"],
                        "default": "general",
                    },
                    "context": {"type": "string", "default": ""},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="memory_stats",
            description="Get statistics about the memory system.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="sync_memories",
            description="Sync SQLite memories to ChromaDB for semantic search.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="wm_get_session",
            description="Get current session context. Call at session start or after compaction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_recent_learnings": {"type": "boolean", "default": True},
                    "learning_hours": {"type": "integer", "default": 24},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    if name == "search_memory":
        results = await store.search(arguments.get("query", ""), arguments.get("limit", 10))
        if not results:
            return [TextContent(type="text", text=f"No memories found matching: {arguments.get('query')}")]

        text = f"# Memory Search: '{arguments.get('query')}'\n\nFound {len(results)} memories:\n\n"
        for i, m in enumerate(results, 1):
            proj = "Global" if m["project"] == "global" else m["project"]
            text += f"## {i}. {m['type'].title()} ({proj})\n"
            text += f"**When:** {m['timestamp'][:10]} | **Relevance:** {m['relevance']:.2f}\n"
            text += f"{m['content']}\n\n"
        return [TextContent(type="text", text=text)]

    elif name == "get_recent_learnings":
        results = await store.get_recent(arguments.get("days", 7), arguments.get("limit", 20))
        if not results:
            return [TextContent(type="text", text="No recent memories found")]

        text = f"# Recent Learnings\n\nFound {len(results)} memories:\n\n"
        for i, m in enumerate(results, 1):
            text += f"- **{m['type']}** ({m['timestamp'][:10]}): {m['content']}\n"
        return [TextContent(type="text", text=text)]

    elif name == "add_learning":
        content = arguments.get("content", "")
        if not content:
            return [TextContent(type="text", text="Error: content is required")]

        memory_id = await store.add(content, arguments.get("category", "general"), arguments.get("context", ""))
        return [TextContent(type="text", text=f"Stored learning: {memory_id[:8]}...")]

    elif name == "memory_stats":
        stats = await store.stats()
        text = f"# Memory Statistics\n\n"
        text += f"**Total:** {stats['total_memories']} | **Recent (7d):** {stats['recent_7_days']}\n"
        text += f"**ChromaDB:** {'Available' if stats['chroma_available'] else 'Not available (SQLite only)'}\n\n"
        text += "**By type:**\n"
        for t, c in stats["by_type"].items():
            text += f"- {t}: {c}\n"
        return [TextContent(type="text", text=text)]

    elif name == "sync_memories":
        try:
            count = await store.sync_to_chroma()
            return [TextContent(type="text", text=f"Synced {count} memories to ChromaDB")]
        except Exception as e:
            return [TextContent(type="text", text=f"Sync failed: {e}")]

    elif name == "wm_get_session":
        text = await store.get_session(
            arguments.get("include_recent_learnings", True),
            arguments.get("learning_hours", 24)
        )
        return [TextContent(type="text", text=text)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    await store.initialize()
    print(f"Claude Memory MCP Server started", file=sys.stderr)
    print(f"Project: {store.project} | DB: {store.db_path}", file=sys.stderr)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def run():
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
