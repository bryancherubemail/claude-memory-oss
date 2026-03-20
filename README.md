# claude-memory

Local long-term memory for Claude Code via MCP. No cloud, no containers, no rent.

## The Problem

Every time you start a new Claude Code session, Claude starts fresh. It doesn't remember:

- Decisions you made yesterday ("we're using PostgreSQL, not MySQL")
- Conventions you established ("always use camelCase in this project")
- Gotchas you discovered ("the auth middleware must come before the rate limiter")
- Solutions that worked ("fixed the CORS issue by adding the credentials header")

You end up re-explaining context, or worse, Claude suggests things you've already tried and rejected.

## The Solution

This system gives Claude persistent memory across sessions using a single SQLite database with hybrid search. No Docker containers, no cloud services, no API keys required.

### What's New in v2

v2 is a complete rewrite of the memory system:

| Feature         | v1                         | v2                                   |
| --------------- | -------------------------- | ------------------------------------ |
| Storage         | SQLite + optional ChromaDB | SQLite + FTS5 + sqlite-vec (unified) |
| Search          | Text or ChromaDB semantic  | Hybrid BM25 + vector with RRF fusion |
| Retrieval       | Flat                       | Tiered (hot cache -> FTS -> hybrid)  |
| Deduplication   | Embedding similarity       | Hash + semantic similarity           |
| Extraction      | Single model               | Primary + fallback model             |
| Knowledge graph | None                       | Memory relations with traversal      |
| Threads         | None                       | Open/resolved thread tracking        |
| Importance      | Static                     | Decay over time, access-boosted      |
| Containers      | ChromaDB + TEI Docker      | None needed                          |

### Key Features

- **Hybrid search** - BM25 keyword + vector semantic search with Reciprocal Rank Fusion (RRF)
- **Smart alpha** - Auto-tunes keyword vs semantic weighting based on query type
- **Tiered retrieval** - Hot cache (0ms) -> FTS5 (10ms) -> hybrid (50ms), stops early
- **Knowledge graph** - Memory relations, contradiction detection, graph traversal
- **Thread tracking** - Open items persist across sessions, mark resolved when done
- **Importance decay** - Unused memories fade, frequently accessed ones stay prominent
- **Deduplication** - Hash + semantic similarity prevents duplicate storage
- **Session recovery** - Preserves context across compaction events
- **Fully local** - All processing via Ollama, data stays on your machine

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MCP Server  │    │   Hooks      │    │  Extraction  │  │
│  │  (13 tools)  │    │              │    │   (Ollama)   │  │
│  │              │    │ PreCompact   │    │              │  │
│  │ search       │    │ UserPrompt   │    │ gemma3:27b   │  │
│  │ add_memory   │    │ SessionEnd   │    │ (primary)    │  │
│  │ recall       │    │              │    │              │  │
│  │ threads      │    │ Prefetch +   │    │ gemma3:4b    │  │
│  │ relations    │    │ Extract      │    │ (fallback)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐    │
│  │              Tiered Retrieval                       │    │
│  │  Tier 0: Hot Cache (LRU, <1ms, 0 tokens)          │    │
│  │  Tier 1: FTS5 BM25 (SQLite, <10ms, ~100 tokens)   │    │
│  │  Tier 2: Hybrid RRF (BM25+vec, <50ms, ~300 tokens)│    │
│  └──────────────────────────┬──────────────────────────┘    │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐    │
│  │         SQLite (single file, WAL mode)              │    │
│  │  memories + FTS5 index + sqlite-vec embeddings      │    │
│  │  memory_relations (knowledge graph)                 │    │
│  │  sessions + projects                                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull nomic-embed-text    # embeddings (768 dims)
ollama pull gemma3:27b          # extraction (or gemma3:latest for smaller GPUs)
```

### 2. Install

```bash
# Clone
git clone https://github.com/bryancherubemail/claude-memory-oss.git ~/.claude-memory
cd ~/.claude-memory

# Install dependencies
pip install mcp httpx sqlite-vec
```

### 3. Configure Claude Code

Add to your project's `.mcp.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "claude-memory-v2": {
      "command": "python3",
      "args": ["-m", "claude_memory.server"],
      "cwd": "~/.claude-memory/src"
    }
  }
}
```

Restart Claude Code. Memory tools are now available.

### 4. Enable Hooks (Recommended)

Hooks automate memory extraction and context injection. Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude-memory/src/claude_memory/server/hooks/pre_compact.py",
            "timeout": 30,
            "statusMessage": "Extracting memories..."
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude-memory/src/claude_memory/server/hooks/memory_prefetch.py",
            "timeout": 5,
            "statusMessage": "Loading context..."
          }
        ]
      }
    ]
  }
}
```

See `hooks-example.json` for the full example including SessionEnd.

## MCP Tools Reference

| Tool                  | Purpose                                                           |
| --------------------- | ----------------------------------------------------------------- |
| `search_memory`       | Hybrid BM25 + vector search with smart alpha tuning               |
| `recall_context`      | Tiered context retrieval (hot cache -> FTS -> hybrid)             |
| `add_memory`          | Store a decision, fact, preference, gotcha, convention, or thread |
| `extract`             | Extract insights from a conversation exchange via Ollama          |
| `get_session`         | Get session state for recovery after compaction                   |
| `save_session`        | Save session state before compaction                              |
| `get_threads`         | List open/unresolved threads for a project                        |
| `resolve_thread`      | Mark a thread as resolved                                         |
| `get_relations`       | Traverse the memory knowledge graph                               |
| `find_contradictions` | Find memories that contradict each other                          |
| `memory_stats`        | System statistics (counts, categories, namespaces)                |
| `decay_old`           | Run importance decay on old unused memories                       |
| `update_memory`       | Update a memory's importance score                                |

## How It Works

### Hybrid Search with Smart Alpha

Every search query runs through two signals fused with Reciprocal Rank Fusion:

1. **BM25 (FTS5)** - Keyword matching with Porter stemming
2. **Vector (sqlite-vec)** - Semantic similarity via nomic-embed-text embeddings

The system auto-tunes the balance (`alpha`) based on query characteristics:

- Technical queries (camelCase, snake_case, error messages) -> more BM25 (exact matching)
- Conceptual queries ("how to", "best practice", "why") -> more vector (semantic)
- Alpha range: 0.2 (mostly keyword) to 0.8 (mostly semantic)

### Tiered Retrieval

Context recall uses progressive loading to minimize latency and token usage:

| Tier | Source                     | Latency | Token Cost |
| ---- | -------------------------- | ------- | ---------- |
| 0    | Hot cache (LRU, in-memory) | <1ms    | 0          |
| 1    | FTS5 BM25 (SQLite)         | <10ms   | ~100       |
| 2    | Hybrid RRF (BM25 + vector) | <50ms   | ~300       |

Stops early when enough results are found or the token budget is exceeded.

### Auto-Extraction

The PreCompact hook reads Claude's conversation transcript and uses a local Ollama model to extract structured insights:

- **DECISION** - Choices made with rationale
- **FACT** - Technical details established
- **PREFERENCE** - User prefers X over Y
- **GOTCHA** - Things that break unexpectedly
- **CONVENTION** - Rules or patterns to follow
- **THREAD** - Open/unresolved items

Each extracted insight is deduplicated (hash + semantic similarity) before storage. Related memories are automatically linked in the knowledge graph.

### Importance Decay

Memories that haven't been accessed in 7+ days get their importance reduced by 5% per week. Memories that decay to importance=1 and are older than 30 days get archived. Frequently accessed memories maintain their importance naturally.

## Migrating from v1

If you have an existing v1 database (`~/.claude-memory/data/memory.db`), run the migration script:

```bash
# Preview what will be migrated
python3 migrate_v1_to_v2.py --dry-run

# Run migration (re-embeds all memories with nomic-embed-text)
python3 migrate_v1_to_v2.py
```

The migration:

- Maps v1 types to v2 categories
- Re-embeds all memories with nomic-embed-text (768 dims)
- Deduplicates by content hash
- Preserves timestamps and project associations

v1 data is not modified - the migration creates a new v2 database.

## Configuration

Environment variables:

| Variable                    | Default                   | Purpose                   |
| --------------------------- | ------------------------- | ------------------------- |
| `CLAUDE_MEMORY_DIR`         | `~/.claude-memory`        | Data directory            |
| `OLLAMA_URL`                | `http://localhost:11434`  | Ollama API URL            |
| `EMBEDDING_MODEL`           | `nomic-embed-text:latest` | Embedding model           |
| `EXTRACTION_MODEL`          | `gemma3:27b`              | Primary extraction model  |
| `EXTRACTION_MODEL_FALLBACK` | `gemma3:latest`           | Fallback extraction model |

## Data Storage

All data stays local in a single SQLite file:

```
~/.claude-memory/
├── db/
│   └── memory_v2.db       # Single SQLite file (FTS5 + sqlite-vec)
├── sessions/               # Session state files
├── logs/                   # Extraction logs
├── src/
│   └── claude_memory/
│       ├── server/         # v2 MCP server (modular)
│       │   ├── main.py     # MCP entry point (13 tools)
│       │   ├── config.py   # Central configuration
│       │   ├── storage/    # SessionDB, HotCache, TieredRetrieval
│       │   ├── extraction/ # Ollama-based insight extraction
│       │   ├── utils/      # Fusion, embeddings, compression, prefetch
│       │   └── hooks/      # PreCompact, UserPromptSubmit
│       ├── server.py       # v1 server (kept for compatibility)
│       └── auto_learn.py   # v1 extraction (standalone)
├── scripts/
│   ├── mem                 # CLI tool
│   ├── async-extract       # Background extraction
│   └── inject-session-state
├── migrate_v1_to_v2.py     # v1 -> v2 migration
└── hooks-example.json      # Hook configuration example
```

Nothing leaves your machine. No telemetry, no cloud, no API keys.

## CLI Tool (`mem`)

The `mem` command provides direct access to your v1 memories:

```bash
# Add to PATH
export PATH="$PATH:~/.claude-memory/scripts"

mem search <query>       # Search memories
mem stats                # Show statistics
mem recent [days]        # Recent memories
mem consolidate [--apply] # Find duplicates
mem prune --days N       # Remove old memories
mem export [file]        # Export to JSON
```

## Why This Exists

There are paid services that do this (Mem0, Zep, etc). They charge $25+/month for what amounts to SQLite + embeddings + an LLM call.

I built this because:

1. The core functionality is simple
2. My data should stay on my machine
3. I shouldn't pay rent for a database

This is that system, shared for anyone who feels the same way.

## Status

This is a personal project shared as-is.

- No support
- No guarantees
- No roadmap

Fork it, extend it, make it your own. If it breaks, you get to keep both pieces.

## License

MIT - do whatever you want with it.
