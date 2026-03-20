#!/usr/bin/env python3
"""
Migrate v1 memories (memory.db) to v2 unified SQLite with FTS5 + sqlite-vec.
Re-embeds all memories using nomic-embed-text via Ollama.

Usage:
    python3 migrate_v1_to_v2.py [--dry-run] [--batch-size 50]
"""

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

import httpx
import sqlite_vec

MEMORY_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
V1_DB = MEMORY_DIR / "data" / "memory.db"
V2_DB = MEMORY_DIR / "db" / "memory_v2.db"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_DIM = 768


def get_v2_connection():
    V2_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(V2_DB), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def get_embedding_sync(text, client):
    """Get embedding from Ollama synchronously."""
    try:
        resp = client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        embeddings = data.get("embeddings", [])
        return embeddings[0] if embeddings else None
    except Exception as e:
        print(f"  Embedding error: {e}", file=sys.stderr)
        return None


def compute_hash(content):
    return hashlib.sha256(content.lower().strip().encode()).hexdigest()[:16]


def map_category(v1_type):
    """Map v1 types to v2 categories."""
    mapping = {
        "decisions": "decision",
        "decision": "decision",
        "solutions": "fact",
        "solution": "fact",
        "conventions": "convention",
        "convention": "convention",
        "gotchas": "gotcha",
        "gotcha": "gotcha",
        "tech_stack": "fact",
        "general": "fact",
        "learning": "fact",
    }
    return mapping.get(v1_type, "fact")


def main():
    parser = argparse.ArgumentParser(description="Migrate v1 to v2")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    args = parser.parse_args()

    if not V1_DB.exists():
        print(f"v1 database not found at {V1_DB}")
        sys.exit(1)

    # Read v1 data
    v1 = sqlite3.connect(str(V1_DB))
    v1.row_factory = sqlite3.Row
    v1_memories = v1.execute(
        "SELECT id, timestamp, type, content, context, relevance_score, project FROM memories ORDER BY timestamp"
    ).fetchall()
    v1.close()

    print(f"Found {len(v1_memories)} v1 memories")

    if args.dry_run:
        by_project = {}
        by_type = {}
        for m in v1_memories:
            p = m["project"] or "global"
            t = m["type"]
            by_project[p] = by_project.get(p, 0) + 1
            by_type[t] = by_type.get(t, 0) + 1

        print("\nBy project:")
        for p, c in sorted(by_project.items(), key=lambda x: -x[1]):
            print(f"  {p}: {c}")
        print(f"\nBy type:")
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t} -> {map_category(t)}: {c}")
        print(f"\nWould create v2 database at: {V2_DB}")
        print("Run without --dry-run to proceed.")
        return

    # Initialize v2 schema
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from claude_memory.server.storage.session_db import init_db
    v2 = init_db(V2_DB)

    # Check for existing v2 data
    existing = v2.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if existing > 0:
        print(f"v2 database already has {existing} memories.")
        resp = input("Continue and add new ones? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # Migrate with embeddings
    client = httpx.Client(timeout=30.0)
    migrated = 0
    skipped = 0
    embed_errors = 0
    seen_hashes = set()

    for row in v2.execute("SELECT content_hash FROM memories"):
        seen_hashes.add(row[0])

    start_time = time.time()

    for i, m in enumerate(v1_memories):
        content = m["content"]
        if not content or len(content.strip()) < 5:
            skipped += 1
            continue

        content_hash = compute_hash(content)
        if content_hash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(content_hash)

        namespace = m["project"] or "global"
        category = map_category(m["type"])
        importance = min(10, max(1, int((m["relevance_score"] or 1.0) * 5)))

        tags = []
        ctx = m["context"] or ""
        if ctx:
            try:
                parsed = json.loads(ctx)
                if isinstance(parsed, list):
                    tags = parsed
                elif isinstance(parsed, dict):
                    tags = list(parsed.keys())
            except (json.JSONDecodeError, TypeError):
                pass

        tags_json = json.dumps(tags)

        cur = v2.execute(
            """INSERT INTO memories
               (namespace, content, content_hash, category, tags, importance,
                source_summary, created_at, updated_at, last_accessed, session_id)
               VALUES (?, ?, ?, ?, ?, ?, 'migrated_from_v1', ?, ?, ?, ?)""",
            (namespace, content, content_hash, category, tags_json, importance,
             m["timestamp"], m["timestamp"], m["timestamp"], f"migration-{m['id']}"),
        )
        new_id = cur.lastrowid

        embedding = get_embedding_sync(content, client)
        if embedding and len(embedding) == EMBEDDING_DIM:
            blob = struct.pack(f"{EMBEDDING_DIM}f", *embedding)
            v2.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (new_id, blob),
            )
        else:
            embed_errors += 1

        migrated += 1

        if migrated % args.batch_size == 0:
            v2.commit()
            elapsed = time.time() - start_time
            rate = migrated / elapsed
            remaining = (len(v1_memories) - i) / rate if rate > 0 else 0
            print(f"  Migrated: {migrated}/{len(v1_memories)} "
                  f"({migrated/len(v1_memories)*100:.1f}%) "
                  f"| {rate:.1f}/s "
                  f"| ETA: {remaining/60:.1f}m "
                  f"| Skipped: {skipped} "
                  f"| Embed errors: {embed_errors}")

    v2.commit()
    client.close()

    elapsed = time.time() - start_time
    print(f"\nMigration complete!")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (dups/empty): {skipped}")
    print(f"  Embedding errors: {embed_errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  v2 DB: {V2_DB} ({V2_DB.stat().st_size / 1024 / 1024:.1f} MB)")

    total = v2.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    vec_count = v2.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
    fts_count = v2.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
    print(f"\nVerification:")
    print(f"  memories table: {total}")
    print(f"  memories_vec: {vec_count}")
    print(f"  memories_fts: {fts_count}")

    v2.close()


if __name__ == "__main__":
    main()
