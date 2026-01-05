#!/usr/bin/env python3
"""
Claude Memory MCP Server
Local long-term memory for Claude Code via MCP protocol.

Features:
- SQLite for persistent storage
- Optional ChromaDB for semantic search
- Scope-aware memories (project, global, language-specific)
- BM25 hybrid scoring
- Access count tracking
- Duplicate detection and consolidation
"""

import asyncio
import hashlib
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Configuration - override with environment variables
DATA_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
MEMORY_DB = DATA_DIR / "data" / "memory.db"
SESSIONS_DIR = DATA_DIR / "sessions"
CHROMA_URL = os.environ.get("CHROMA_URL", "http://localhost:8000")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
COLLECTION_NAME = "claude_memories"
CHROMA_API_BASE = f"{CHROMA_URL}/api/v2/tenants/default_tenant/databases/default_database/collections"
GLOBAL_PROJECT = "global"

# Language detection patterns for scope inference
LANGUAGE_PATTERNS = {
    "python": [".py", "django", "flask", "fastapi", "asyncio", "pip", "pytest"],
    "javascript": [".js", ".ts", "node", "npm", "react", "vue", "next"],
    "go": [".go", "golang", "goroutine", "gorm"],
    "rust": [".rs", "cargo", "tokio", "async-std"],
    "sql": ["postgres", "mysql", "sqlite", "database", "query"],
}


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


def detect_language_from_content(content: str) -> Optional[str]:
    """Detect programming language from content for scope inference."""
    content_lower = content.lower()
    for lang, patterns in LANGUAGE_PATTERNS.items():
        if any(p in content_lower for p in patterns):
            return lang
    return None


def compute_bm25_score(query: str, document: str) -> float:
    """Simple BM25-inspired keyword boost score."""
    query_terms = set(query.lower().split())
    doc_terms = document.lower().split()
    doc_term_set = set(doc_terms)

    matches = sum(1 for term in query_terms if term in doc_term_set)

    # Bonus for exact phrase matches
    if query.lower() in document.lower():
        matches += 2

    if len(query_terms) > 0:
        return matches / len(query_terms) * 0.2
    return 0.0


class MemoryStore:
    """SQLite + optional ChromaDB memory storage with scope awareness."""

    def __init__(self):
        self.db_path = MEMORY_DB
        self.chroma_url = CHROMA_URL
        self.ollama_url = OLLAMA_URL
        self.embedding_model = EMBEDDING_MODEL
        self.http_client = None
        self.project = detect_project()
        self.chroma_available = False
        self.collection_id = None

    async def initialize(self):
        """Initialize database and optional vector store."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        self._init_db()

        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Try ChromaDB
        try:
            await self._init_chroma()
            self.chroma_available = True
            print(f"ChromaDB available at {self.chroma_url}", file=sys.stderr)
        except Exception as e:
            print(f"ChromaDB not available (using SQLite only): {e}", file=sys.stderr)

    def _init_db(self):
        """Initialize SQLite schema with scope support."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                relevance_score REAL DEFAULT 1.0,
                project TEXT DEFAULT 'global',
                scope TEXT DEFAULT 'project',
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON memories(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scope ON memories(scope)")
        conn.commit()
        conn.close()

        # Ensure schema columns exist
        self._ensure_schema_columns()

    def _ensure_schema_columns(self):
        """Ensure all required columns exist (for upgrades)."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("PRAGMA table_info(memories)")
            columns = [col[1] for col in cursor.fetchall()]

            required = {
                "scope": "TEXT DEFAULT 'project'",
                "access_count": "INTEGER DEFAULT 0",
            }

            for col_name, col_def in required.items():
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
                    conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    async def _init_chroma(self):
        """Initialize ChromaDB collection (1.0.0 API)."""
        # List collections to find ours
        try:
            response = await self.http_client.get(CHROMA_API_BASE)
            if response.status_code == 200:
                collections = response.json()
                for coll in collections:
                    if coll.get("name") == COLLECTION_NAME:
                        self.collection_id = coll["id"]
                        return
        except Exception:
            pass

        # Create collection if not exists
        response = await self.http_client.post(
            CHROMA_API_BASE,
            json={"name": COLLECTION_NAME, "metadata": {"description": "Claude memories"}}
        )
        if response.status_code in (200, 201):
            result = response.json()
            self.collection_id = result["id"]

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector from Ollama."""
        if not self.chroma_available:
            return []
        try:
            response = await self.http_client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            result = response.json()
            return result.get("embedding", [])
        except Exception:
            return []

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search memories - semantic if available, else text."""
        if self.chroma_available and self.collection_id:
            results = await self._search_semantic(query, limit)
            if results:
                return results
        return self._search_text(query, limit)

    async def _search_semantic(self, query: str, limit: int) -> list[dict]:
        """Semantic search with scope awareness and BM25 boost."""
        try:
            embedding = await self.get_embedding(query)
            if not embedding:
                return []

            query_language = detect_language_from_content(query)

            response = await self.http_client.post(
                f"{CHROMA_API_BASE}/{self.collection_id}/query",
                json={"query_embeddings": [embedding], "n_results": limit * 5}
            )
            results = response.json()

            memories = []
            retrieved_ids = []

            if results.get("ids") and results["ids"][0]:
                for i, mem_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    content = metadata.get("content", "")
                    project = metadata.get("project", "global")
                    scope = metadata.get("scope", "project")

                    # Scope-aware filtering
                    include = False
                    scope_boost = 0.0

                    if project == self.project:
                        include = True
                        scope_boost = 0.25
                    elif project == "global" or scope == "global":
                        include = True
                        scope_boost = 0.1

                    if scope and scope.startswith("language:"):
                        mem_language = scope.split(":")[1]
                        if query_language and mem_language == query_language:
                            include = True
                            scope_boost = 0.15

                    if not include:
                        continue

                    base_relevance = 1.0 - distance
                    bm25_boost = compute_bm25_score(query, content)
                    final_relevance = min(1.0, base_relevance + scope_boost + bm25_boost)

                    memories.append({
                        "id": mem_id,
                        "content": content,
                        "type": metadata.get("type", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "project": project,
                        "scope": scope,
                        "relevance": round(final_relevance, 3),
                    })
                    retrieved_ids.append(mem_id)

            memories.sort(key=lambda x: x["relevance"], reverse=True)
            top_memories = memories[:limit]

            if retrieved_ids:
                self._increment_access_counts([m["id"] for m in top_memories])

            return top_memories

        except Exception:
            return []

    def _increment_access_counts(self, memory_ids: list[str]):
        """Increment access_count for retrieved memories."""
        if not memory_ids:
            return
        try:
            conn = self._get_conn()
            placeholders = ",".join("?" * len(memory_ids))
            conn.execute(
                f"UPDATE memories SET access_count = access_count + 1 WHERE id IN ({placeholders})",
                memory_ids
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _search_text(self, query: str, limit: int) -> list[dict]:
        """Text search via SQLite with project priority."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT id, content, type, timestamp, context, relevance_score, project, scope
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
            "scope": row["scope"] or "project",
            "relevance": row["relevance_score"],
        } for row in cursor.fetchall()]

        conn.close()
        return results

    async def get_recent(self, days: int = 7, limit: int = 20) -> list[dict]:
        """Get recent memories with project priority."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor = conn.execute("""
            SELECT id, content, type, timestamp, context, relevance_score, project, scope
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
            "scope": row["scope"] or "project",
            "relevance": row["relevance_score"],
        } for row in cursor.fetchall()]

        conn.close()
        return results

    async def add(self, content: str, memory_type: str = "learning", context: str = "", scope: str = None) -> str:
        """Add a memory with scope detection."""
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()
        timestamp = datetime.now().isoformat()

        # Auto-detect scope
        if scope is None:
            detected_lang = detect_language_from_content(content)
            if detected_lang:
                scope = f"language:{detected_lang}"
            else:
                scope = "project"

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO memories (id, timestamp, type, content, context, relevance_score, project, scope, access_count)
               VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, 0)""",
            (memory_id, timestamp, memory_type, content, context, self.project, scope)
        )
        conn.commit()
        conn.close()

        # Add to ChromaDB if available
        if self.chroma_available and self.collection_id:
            try:
                embedding = await self.get_embedding(content)
                if embedding:
                    await self.http_client.post(
                        f"{CHROMA_API_BASE}/{self.collection_id}/add",
                        json={
                            "ids": [memory_id],
                            "embeddings": [embedding],
                            "metadatas": [{
                                "content": content,
                                "type": memory_type,
                                "timestamp": timestamp,
                                "context": context,
                                "project": self.project,
                                "scope": scope,
                            }]
                        }
                    )
            except Exception:
                pass

        return memory_id

    async def stats(self) -> dict:
        """Get memory statistics including scope breakdown."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        by_type = {}
        for row in conn.execute("SELECT type, COUNT(*) as count FROM memories GROUP BY type"):
            by_type[row["type"]] = row["count"]

        by_scope = {}
        for row in conn.execute("SELECT scope, COUNT(*) as count FROM memories GROUP BY scope"):
            by_scope[row["scope"] or "project"] = row["count"]

        # Most accessed
        most_accessed = []
        for row in conn.execute(
            "SELECT content, type, access_count FROM memories WHERE access_count > 0 ORDER BY access_count DESC LIMIT 5"
        ):
            most_accessed.append({
                "content": row["content"][:80],
                "type": row["type"],
                "access_count": row["access_count"]
            })

        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = conn.execute("SELECT COUNT(*) FROM memories WHERE timestamp > ?", (cutoff,)).fetchone()[0]

        conn.close()
        return {
            "total_memories": total,
            "by_type": by_type,
            "by_scope": by_scope,
            "most_accessed": most_accessed,
            "recent_7_days": recent,
            "database_path": str(self.db_path),
            "chroma_available": self.chroma_available,
        }

    async def find_duplicates(self, threshold: float = 0.85) -> list[dict]:
        """Find duplicate memories using embedding similarity."""
        if not self.chroma_available or not self.collection_id:
            return []

        try:
            response = await self.http_client.post(
                f"{CHROMA_API_BASE}/{self.collection_id}/get",
                json={"include": ["embeddings", "metadatas"]}
            )
            data = response.json()

            ids = data.get("ids", [])
            embeddings = data.get("embeddings", [])
            metadatas = data.get("metadatas", [])

            if not embeddings:
                return []

            import numpy as np
            duplicates = []
            embeddings_array = np.array(embeddings)

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    emb_i = embeddings_array[i]
                    emb_j = embeddings_array[j]
                    similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))

                    if similarity >= threshold:
                        duplicates.append({
                            "id_1": ids[i],
                            "id_2": ids[j],
                            "content_1": metadatas[i].get("content", "")[:100],
                            "content_2": metadatas[j].get("content", "")[:100],
                            "similarity": round(float(similarity), 3),
                        })

            return duplicates
        except Exception:
            return []

    async def consolidate(self, threshold: float = 0.85, dry_run: bool = True) -> dict:
        """Consolidate duplicate memories."""
        duplicates = await self.find_duplicates(threshold)

        if not duplicates:
            return {"merged": 0, "message": "No duplicates found"}

        to_delete = set()
        for dup in duplicates:
            id_1, id_2 = dup["id_1"], dup["id_2"]

            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT id, access_count, timestamp FROM memories WHERE id IN (?, ?)",
                (id_1, id_2)
            )
            rows = {row["id"]: row for row in cursor.fetchall()}
            conn.close()

            if id_1 in rows and id_2 in rows:
                r1, r2 = rows[id_1], rows[id_2]
                if (r1["access_count"] or 0) > (r2["access_count"] or 0):
                    to_delete.add(id_2)
                elif (r2["access_count"] or 0) > (r1["access_count"] or 0):
                    to_delete.add(id_1)
                elif r1["timestamp"] >= r2["timestamp"]:
                    to_delete.add(id_2)
                else:
                    to_delete.add(id_1)

        if dry_run:
            return {
                "would_delete": len(to_delete),
                "duplicates_found": len(duplicates),
                "message": "Dry run - no changes made. Call with dry_run=False to delete.",
            }

        deleted = 0
        for memory_id in to_delete:
            try:
                conn = self._get_conn()
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
                conn.close()

                if self.chroma_available and self.collection_id:
                    await self.http_client.post(
                        f"{CHROMA_API_BASE}/{self.collection_id}/delete",
                        json={"ids": [memory_id]}
                    )
                deleted += 1
            except Exception:
                pass

        return {
            "merged": deleted,
            "duplicates_found": len(duplicates),
            "message": f"Deleted {deleted} duplicate memories",
        }

    async def sync_to_chroma(self) -> int:
        """Sync all SQLite memories to ChromaDB."""
        if not self.chroma_available or not self.collection_id:
            raise Exception("ChromaDB not available")

        conn = self._get_conn()
        cursor = conn.execute("SELECT id, content, type, timestamp, context, project, scope FROM memories")

        count = 0
        for row in cursor.fetchall():
            try:
                embedding = await self.get_embedding(row["content"])
                if embedding:
                    await self.http_client.post(
                        f"{CHROMA_API_BASE}/{self.collection_id}/add",
                        json={
                            "ids": [row["id"]],
                            "embeddings": [embedding],
                            "metadatas": [{
                                "content": row["content"],
                                "type": row["type"],
                                "timestamp": row["timestamp"],
                                "context": row["context"] or "",
                                "project": row["project"] or "global",
                                "scope": row["scope"] or "project",
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

        session_file = SESSIONS_DIR / f"{self.project}.md"
        if session_file.exists():
            try:
                formatted += session_file.read_text() + "\n\n"
            except Exception:
                pass

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
            description="Search memories using semantic similarity. Returns relevant past learnings, decisions, and context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'authentication', 'database decisions')"},
                    "limit": {"type": "integer", "default": 10, "description": "Maximum number of results (default: 10)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_recent_learnings",
            description="Get recent memories from the last N days. Useful for understanding recent context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "Number of days to look back (default: 7)"},
                    "limit": {"type": "integer", "default": 20, "description": "Maximum number of results (default: 20)"},
                },
            },
        ),
        Tool(
            name="add_learning",
            description="Store a new learning or decision in memory. Use this to remember important context for future sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The learning or decision to remember"},
                    "category": {
                        "type": "string",
                        "enum": ["decisions", "conventions", "tech_stack", "gotchas", "solutions", "general"],
                        "default": "general",
                        "description": "Category (decisions, conventions, tech_stack, gotchas, solutions)",
                    },
                    "context": {"type": "string", "default": "", "description": "Additional context about this learning"},
                    "scope": {
                        "type": "string",
                        "default": None,
                        "description": "Scope: 'project' (current project only), 'global' (all projects), or 'language:X' (e.g., 'language:python'). Auto-detected if not provided.",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="consolidate_memories",
            description="Find and optionally remove duplicate memories based on semantic similarity. Use dry_run=true first to preview.",
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "default": 0.85, "description": "Similarity threshold (0.0-1.0). Default 0.85 means 85% similar."},
                    "dry_run": {"type": "boolean", "default": True, "description": "If true, only report duplicates without deleting. Default true."},
                },
            },
        ),
        Tool(
            name="memory_stats",
            description="Get statistics about the memory system (total memories, by category, recent activity)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="sync_memories",
            description="One-time sync of all SQLite memories to ChromaDB for semantic search. Run this once to enable vector search.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="wm_get_session",
            description="Get current session context after compaction or at session start. Returns working state, recent decisions, and relevant context for the current project. Call this when: (1) starting a new session, (2) context seems stale, (3) you notice confusion about previously discussed topics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_recent_learnings": {"type": "boolean", "default": True, "description": "Include recent learnings from memory DB (default: true)"},
                    "learning_hours": {"type": "integer", "default": 24, "description": "Hours of recent learnings to include (default: 24)"},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    if name == "search_memory":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)

        results = await store.search(query, limit)

        if not results:
            return [TextContent(type="text", text=f"No memories found matching: {query}")]

        text = f"# Memory Search: '{query}'\n\n"
        text += f"**Project Context:** {store.project}\n"
        text += f"Found {len(results)} memories:\n\n"

        for i, m in enumerate(results, 1):
            proj = "🌐 Global" if m["project"] == "global" else f"📁 {m['project']}"
            scope = f" [{m.get('scope', 'project')}]" if m.get('scope') != 'project' else ""
            text += f"## {i}. {m['type'].title()} ({proj}){scope}\n"
            text += f"**When:** {m['timestamp'][:10]} | **Relevance:** {m['relevance']:.2f}\n"
            text += f"{m['content']}\n\n"

        return [TextContent(type="text", text=text)]

    elif name == "get_recent_learnings":
        results = await store.get_recent(arguments.get("days", 7), arguments.get("limit", 20))

        if not results:
            return [TextContent(type="text", text="No recent memories found")]

        text = f"# Recent Learnings\n\n"
        text += f"**Project Context:** {store.project}\n"
        text += f"Found {len(results)} memories:\n\n"

        for i, m in enumerate(results, 1):
            proj = "🌐 Global" if m["project"] == "global" else f"📁 {m['project']}"
            text += f"- **{m['type']}** ({m['timestamp'][:10]}, {proj}): {m['content']}\n"

        return [TextContent(type="text", text=text)]

    elif name == "add_learning":
        content = arguments.get("content", "")
        if not content:
            return [TextContent(type="text", text="Error: content is required")]

        scope = arguments.get("scope")
        memory_id = await store.add(
            content,
            arguments.get("category", "general"),
            arguments.get("context", ""),
            scope
        )

        proj_label = "🌐 Global" if store.project == "global" else f"📁 {store.project}"
        scope_label = scope if scope else "auto-detected"

        return [TextContent(
            type="text",
            text=f"✅ Stored new {arguments.get('category', 'general')} learning ({proj_label}, scope: {scope_label})\n\n**ID:** {memory_id[:8]}...\n**Content:** {content}"
        )]

    elif name == "consolidate_memories":
        threshold = arguments.get("threshold", 0.85)
        dry_run = arguments.get("dry_run", True)

        result = await store.consolidate(threshold, dry_run)

        text = "# Memory Consolidation\n\n"
        text += f"**Threshold:** {threshold}\n"
        text += f"**Mode:** {'Dry Run (preview)' if dry_run else 'Live (deleted)'}\n\n"
        text += f"**Duplicates Found:** {result.get('duplicates_found', 0)}\n"

        if dry_run:
            text += f"**Would Delete:** {result.get('would_delete', 0)}\n"
        else:
            text += f"**Deleted:** {result.get('merged', 0)}\n"

        text += f"\n{result.get('message', '')}"

        return [TextContent(type="text", text=text)]

    elif name == "memory_stats":
        stats = await store.stats()

        text = "# Memory System Statistics\n\n"
        text += f"**Total:** {stats['total_memories']} | **Recent (7d):** {stats['recent_7_days']}\n"
        text += f"**ChromaDB:** {'Available' if stats['chroma_available'] else 'Not available (SQLite only)'}\n\n"

        text += "**By type:**\n"
        for t, c in stats["by_type"].items():
            text += f"- {t}: {c}\n"

        text += "\n**By scope:**\n"
        for s, c in stats.get("by_scope", {}).items():
            text += f"- {s}: {c}\n"

        if stats.get("most_accessed"):
            text += "\n**Most accessed:**\n"
            for m in stats["most_accessed"]:
                text += f"- [{m['type']}] {m['content']}... ({m['access_count']}x)\n"

        return [TextContent(type="text", text=text)]

    elif name == "sync_memories":
        try:
            count = await store.sync_to_chroma()
            return [TextContent(type="text", text=f"✅ Synced {count} memories to ChromaDB")]
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
