"""
Unified SQLite storage with FTS5 (BM25) + sqlite-vec (vector search).
Single file, zero network hops. Replaces ChromaDB + TEI containers.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sqlite_vec

from ..config import DB_PATH, EMBEDDING_DIM


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get a connection with sqlite-vec loaded."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize the v2 database schema."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    # Core memory table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT,
            importance INTEGER DEFAULT 5,
            source_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1,
            resolved_at TIMESTAMP,
            promoted BOOLEAN DEFAULT 0,
            ttl_hours INTEGER,
            session_id TEXT,
            archived BOOLEAN DEFAULT 0
        )
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_namespace ON memories(namespace)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(namespace, category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_accessed ON memories(last_accessed DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_archived ON memories(archived)")

    # FTS5 for BM25 keyword search
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            category,
            tags,
            namespace,
            content_hash UNINDEXED,
            tokenize='porter unicode61'
        )
    """)

    # Vector table for semantic search (nomic-embed-text = 768 dims)
    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            embedding float[{EMBEDDING_DIM}]
        )
    """)

    # Knowledge graph relations
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
            target_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id, target_id, relation_type)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relations(source_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relations(target_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON memory_relations(relation_type)")

    # Session tracking
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            compaction_count INTEGER DEFAULT 0,
            summary TEXT,
            state_json TEXT
        )
    """)

    # Project registry
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            namespace TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            config_json TEXT
        )
    """)

    # FTS sync triggers
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category, tags, namespace, content_hash)
            VALUES (NEW.id, NEW.content, NEW.category, NEW.tags, NEW.namespace, NEW.content_hash);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
            DELETE FROM memories_fts WHERE rowid = OLD.id;
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
            DELETE FROM memories_fts WHERE rowid = OLD.id;
            INSERT INTO memories_fts(rowid, content, category, tags, namespace, content_hash)
            VALUES (NEW.id, NEW.content, NEW.category, NEW.tags, NEW.namespace, NEW.content_hash);
        END
    """)

    conn.commit()
    return conn


class SessionDB:
    """Main storage interface for Claude Memory v2."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = init_db(db_path)

    def close(self):
        self.conn.close()

    # -- Insert --

    def insert_memory(
        self,
        namespace: str,
        content: str,
        content_hash: str,
        category: str,
        tags: list[str] | None = None,
        importance: int = 5,
        source_summary: str | None = None,
        session_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert a memory with optional embedding."""
        tags_json = json.dumps(tags) if tags else "[]"

        cur = self.conn.execute(
            """INSERT INTO memories
               (namespace, content, content_hash, category, tags, importance, source_summary, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (namespace, content, content_hash, category, tags_json, importance, source_summary, session_id),
        )
        memory_id = cur.lastrowid

        # Insert embedding if provided
        if embedding is not None:
            self._insert_embedding(memory_id, embedding)

        self.conn.commit()
        return memory_id

    def _insert_embedding(self, memory_id: int, embedding: list[float]):
        """Insert or replace an embedding for a memory."""
        import struct
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        try:
            self.conn.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (memory_id, blob),
            )
        except sqlite3.IntegrityError:
            self.conn.execute(
                "UPDATE memories_vec SET embedding = ? WHERE rowid = ?",
                (blob, memory_id),
            )

    def update_embedding(self, memory_id: int, embedding: list[float]):
        """Update embedding for an existing memory."""
        self._insert_embedding(memory_id, embedding)
        self.conn.commit()

    # -- Search: BM25 --

    def fts_search(
        self,
        query: str,
        namespace: str | None = None,
        category: str | None = None,
        limit: int = 20,
        exclude_ids: list[int] | None = None,
    ) -> list[dict]:
        """Full-text BM25 search via FTS5."""
        fts_query = query.replace('"', '""')

        sql = """
            SELECT m.*, fts.rank AS bm25_score
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid
            WHERE memories_fts MATCH ?
              AND m.archived = 0
        """
        params: list = [fts_query]

        if namespace:
            sql += " AND m.namespace = ?"
            params.append(namespace)

        if category:
            sql += " AND m.category = ?"
            params.append(category)

        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            sql += f" AND m.id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error -- fall back to simpler query
            return self._like_search(query, namespace, category, limit, exclude_ids)

        return [self._row_to_dict(r) for r in rows]

    def _like_search(
        self,
        query: str,
        namespace: str | None,
        category: str | None,
        limit: int,
        exclude_ids: list[int] | None,
    ) -> list[dict]:
        """Fallback LIKE search for queries FTS5 can't parse."""
        sql = "SELECT * FROM memories WHERE content LIKE ? AND archived = 0"
        params: list = [f"%{query}%"]

        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        if category:
            sql += " AND category = ?"
            params.append(category)
        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            sql += f" AND id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Search: Vector --

    def vector_search(
        self,
        embedding: list[float],
        namespace: str | None = None,
        limit: int = 20,
        exclude_ids: list[int] | None = None,
    ) -> list[dict]:
        """Cosine-similarity vector search via sqlite-vec."""
        import struct
        blob = struct.pack(f"{len(embedding)}f", *embedding)

        fetch_limit = limit * 3 if namespace or exclude_ids else limit

        rows = self.conn.execute(
            """SELECT rowid, distance
               FROM memories_vec
               WHERE embedding MATCH ?
               ORDER BY distance
               LIMIT ?""",
            (blob, fetch_limit),
        ).fetchall()

        results = []
        for row in rows:
            rid = row[0]
            dist = row[1]

            if exclude_ids and rid in exclude_ids:
                continue

            mem = self.conn.execute("SELECT * FROM memories WHERE id = ? AND archived = 0", (rid,)).fetchone()
            if mem is None:
                continue

            if namespace and mem["namespace"] != namespace and mem["namespace"] != "global":
                continue

            d = self._row_to_dict(mem)
            d["_vec_distance"] = dist
            d["_vec_similarity"] = 1.0 - dist
            results.append(d)

            if len(results) >= limit:
                break

        return results

    # -- Search: Similar (for deduplication) --

    def find_similar(
        self,
        embedding: list[float],
        namespace: str,
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[dict]:
        """Find semantically similar memories above threshold."""
        results = self.vector_search(embedding, namespace=namespace, limit=limit * 2)
        return [r for r in results if r.get("_vec_similarity", 0) >= threshold][:limit]

    # -- Lookup --

    def get_by_id(self, memory_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", ids).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_hash(self, content_hash: str, namespace: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE content_hash = ? AND namespace = ?",
            (content_hash, namespace),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_recent(
        self, namespace: str | None = None, days: int = 7, limit: int = 20
    ) -> list[dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        sql = "SELECT * FROM memories WHERE created_at > ? AND archived = 0"
        params: list = [cutoff]

        if namespace:
            sql += " AND (namespace = ? OR namespace = 'global')"
            params.append(namespace)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Touch / Update --

    def touch(self, memory_id: int):
        """Update access time and count."""
        self.conn.execute(
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (datetime.now().isoformat(), memory_id),
        )
        self.conn.commit()

    def update_importance(self, memory_id: int, importance: int):
        self.conn.execute(
            "UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?",
            (importance, datetime.now().isoformat(), memory_id),
        )
        self.conn.commit()

    def resolve_thread(self, memory_id: int, resolution: str | None = None):
        now = datetime.now().isoformat()
        if resolution:
            self.conn.execute(
                "UPDATE memories SET resolved_at = ?, updated_at = ?, content = content || ' [Resolved: ' || ? || ']' WHERE id = ?",
                (now, now, resolution, memory_id),
            )
        else:
            self.conn.execute(
                "UPDATE memories SET resolved_at = ?, updated_at = ? WHERE id = ?",
                (now, now, memory_id),
            )
        self.conn.commit()

    def archive_memory(self, memory_id: int):
        self.conn.execute("UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                          (datetime.now().isoformat(), memory_id))
        self.conn.commit()

    # -- Knowledge Graph --

    def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        strength: float = 1.0,
    ) -> int | None:
        try:
            cur = self.conn.execute(
                """INSERT INTO memory_relations (source_id, target_id, relation_type, strength)
                   VALUES (?, ?, ?, ?)""",
                (source_id, target_id, relation_type, strength),
            )
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_relations(
        self,
        memory_id: int,
        relation_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        """Get related memories, optionally traversing depth > 1."""
        visited = set()
        results = []
        self._traverse_relations(memory_id, relation_types, depth, visited, results)
        return results

    def _traverse_relations(self, memory_id, relation_types, depth, visited, results):
        if depth <= 0 or memory_id in visited:
            return
        visited.add(memory_id)

        sql = """
            SELECT mr.*, m.content, m.category, m.namespace, m.importance
            FROM memory_relations mr
            JOIN memories m ON m.id = mr.target_id
            WHERE mr.source_id = ?
        """
        params: list = [memory_id]

        if relation_types:
            placeholders = ",".join("?" * len(relation_types))
            sql += f" AND mr.relation_type IN ({placeholders})"
            params.extend(relation_types)

        rows = self.conn.execute(sql, params).fetchall()
        for row in rows:
            results.append({
                "id": row["target_id"],
                "relation_type": row["relation_type"],
                "strength": row["strength"],
                "content": row["content"],
                "category": row["category"],
                "namespace": row["namespace"],
            })
            if depth > 1:
                self._traverse_relations(row["target_id"], relation_types, depth - 1, visited, results)

    def find_contradictions(self, namespace: str | None = None, recent_only: bool = True) -> list[dict]:
        """Find memories with 'contradicts' relations."""
        sql = """
            SELECT mr.*,
                   m1.content AS source_content, m1.category AS source_category, m1.created_at AS source_created,
                   m2.content AS target_content, m2.category AS target_category, m2.created_at AS target_created
            FROM memory_relations mr
            JOIN memories m1 ON m1.id = mr.source_id
            JOIN memories m2 ON m2.id = mr.target_id
            WHERE mr.relation_type = 'contradicts'
        """
        params: list = []

        if namespace:
            sql += " AND (m1.namespace = ? OR m2.namespace = ?)"
            params.extend([namespace, namespace])

        if recent_only:
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            sql += " AND mr.created_at > ?"
            params.append(cutoff)

        sql += " ORDER BY mr.created_at DESC"
        rows = self.conn.execute(sql, params).fetchall()

        return [{
            "source": {"id": r["source_id"], "content": r["source_content"], "category": r["source_category"], "created_at": r["source_created"]},
            "target": {"id": r["target_id"], "content": r["target_content"], "category": r["target_category"], "created_at": r["target_created"]},
            "strength": r["strength"],
        } for r in rows]

    # -- Sessions --

    def save_session(self, session_id: str, namespace: str, state_json: str | None = None):
        self.conn.execute(
            """INSERT OR REPLACE INTO sessions (id, namespace, started_at, state_json)
               VALUES (?, ?, ?, ?)""",
            (session_id, namespace, datetime.now().isoformat(), state_json),
        )
        self.conn.commit()

    def end_session(self, session_id: str, summary: str | None = None):
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (datetime.now().isoformat(), summary, session_id),
        )
        self.conn.commit()

    def increment_compaction(self, session_id: str):
        self.conn.execute(
            "UPDATE sessions SET compaction_count = compaction_count + 1 WHERE id = ?",
            (session_id,),
        )
        self.conn.commit()

    def get_latest_session(self, namespace: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE namespace = ? ORDER BY started_at DESC LIMIT 1",
            (namespace,),
        ).fetchone()
        return dict(row) if row else None

    # -- Projects --

    def register_project(self, namespace: str, path: str, display_name: str | None = None):
        self.conn.execute(
            """INSERT OR REPLACE INTO projects (namespace, path, display_name, last_accessed)
               VALUES (?, ?, ?, ?)""",
            (namespace, path, display_name or namespace, datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_project(self, namespace: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM projects WHERE namespace = ?", (namespace,)).fetchone()
        return dict(row) if row else None

    # -- Stats --

    def get_stats(self, namespace: str | None = None) -> dict:
        where = "WHERE archived = 0"
        params: list = []
        if namespace:
            where += " AND namespace = ?"
            params.append(namespace)

        total = self.conn.execute(f"SELECT COUNT(*) FROM memories {where}", params).fetchone()[0]

        by_category = {}
        for row in self.conn.execute(f"SELECT category, COUNT(*) FROM memories {where} GROUP BY category", params):
            by_category[row[0]] = row[1]

        by_namespace = {}
        base_where = "WHERE archived = 0"
        for row in self.conn.execute(f"SELECT namespace, COUNT(*) FROM memories {base_where} GROUP BY namespace"):
            by_namespace[row[0]] = row[1]

        open_threads = self.conn.execute(
            f"SELECT COUNT(*) FROM memories {where} AND category = 'thread' AND resolved_at IS NULL", params
        ).fetchone()[0]

        relations = self.conn.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0]

        return {
            "total_memories": total,
            "by_category": by_category,
            "by_namespace": by_namespace,
            "open_threads": open_threads,
            "relations": relations,
        }

    # -- Decay --

    def decay_old(
        self, namespace: str | None = None, decay_rate: float = 0.95, dry_run: bool = True
    ) -> dict:
        """Decay importance of old, unused memories."""
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        archive_cutoff = (datetime.now() - timedelta(days=30)).isoformat()

        where = "WHERE last_accessed < ? AND archived = 0"
        params: list = [cutoff]
        if namespace:
            where += " AND namespace = ?"
            params.append(namespace)

        rows = self.conn.execute(
            f"SELECT id, importance, last_accessed, created_at FROM memories {where}", params
        ).fetchall()

        to_decay = []
        to_archive = []
        for row in rows:
            new_importance = max(1, int(row["importance"] * decay_rate))
            if new_importance <= 1 and row["created_at"] < archive_cutoff:
                to_archive.append(row["id"])
            elif new_importance < row["importance"]:
                to_decay.append((new_importance, row["id"]))

        if not dry_run:
            for new_imp, mid in to_decay:
                self.conn.execute("UPDATE memories SET importance = ? WHERE id = ?", (new_imp, mid))
            for mid in to_archive:
                self.conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (mid,))
            self.conn.commit()

        return {"decayed": len(to_decay), "archived": len(to_archive), "dry_run": dry_run}

    # -- Helpers --

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        if "tags" in d and isinstance(d["tags"], str):
            try:
                d["tags"] = json.loads(d["tags"])
            except (json.JSONDecodeError, TypeError):
                d["tags"] = []
        return d
