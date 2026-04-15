# claude-memory

**Claude Code forgets everything between sessions. This fixes that.**

Local-first persistent memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via [MCP](https://modelcontextprotocol.io/). Hybrid BM25 + vector search. Knowledge graph. Importance decay. Zero cloud dependencies. All data stays on your machine.

## Quick Start

### 1. Install Ollama models

```bash
ollama pull nomic-embed-text    # embeddings (required)
ollama pull gemma3               # extraction (optional, for auto-learning)
```

### 2. Install claude-memory

```bash
pip install git+https://github.com/bryancherubemail/claude-memory-oss.git
```

### 3. Add to Claude Code

Add this to your Claude Code MCP settings (`~/.claude.json` or project `.claude.json`):

```json
{
  "mcpServers": {
    "claude-memory": {
      "command": "claude-memory",
      "env": {}
    }
  }
}
```

That's it. Claude Code now has persistent memory across sessions.

## What It Does

- **Remembers across sessions** — decisions, facts, conventions, gotchas, preferences survive compaction and restarts
- **Hybrid search** — BM25 keyword matching + semantic vector search, auto-tuned per query via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **Knowledge graph** — related memories are linked automatically; contradictions are detected
- **Importance decay** — unused memories fade over time, keeping results relevant
- **Auto-extraction** — hooks extract learnings from conversations before compaction
- **Thread tracking** — ongoing work items persist and can be resolved
- **Fully local** — SQLite on your machine, Ollama for embeddings. No API keys, no cloud, no subscriptions

## Tools

| Tool                  | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| `search_memory`       | Hybrid BM25 + vector search across all memories                               |
| `recall_context`      | Get relevant context for current session (use at startup or after compaction) |
| `add_memory`          | Store a new memory with category, tags, and importance                        |
| `extract`             | Extract insights from a conversation exchange                                 |
| `get_session`         | Get current session state for recovery                                        |
| `save_session`        | Save session state before compaction                                          |
| `get_threads`         | List open threads (unresolved work items)                                     |
| `resolve_thread`      | Mark a thread as resolved                                                     |
| `get_relations`       | Traverse the knowledge graph from a memory                                    |
| `find_contradictions` | Find memories that may contradict each other                                  |
| `memory_stats`        | Get memory system statistics                                                  |
| `decay_old`           | Run importance decay on old unused memories                                   |
| `update_memory`       | Update an existing memory's importance                                        |

## Hooks (Optional)

Hooks let Claude Code automatically extract memories from conversations. Copy `hooks-example.json` to your Claude Code hooks config:

| Hook               | When                      | What it does                                             |
| ------------------ | ------------------------- | -------------------------------------------------------- |
| `PreCompact`       | Before context compaction | Extracts decisions, facts, gotchas from the conversation |
| `UserPromptSubmit` | On each user message      | Prefetches relevant context into hot cache               |
| `SessionEnd`       | When session ends         | Final extraction pass                                    |

## Configuration

All settings are in `src/claude_memory/server/config.py` and can be overridden via environment variables:

| Variable                    | Default                   | Description                          |
| --------------------------- | ------------------------- | ------------------------------------ |
| `CLAUDE_MEMORY_DIR`         | `~/.claude-memory`        | Data storage directory               |
| `OLLAMA_URL`                | `http://localhost:11434`  | Ollama API endpoint                  |
| `EMBEDDING_MODEL`           | `nomic-embed-text:latest` | Model for embeddings (768-dim)       |
| `EXTRACTION_MODEL`          | `gemma3:27b`              | Primary model for insight extraction |
| `EXTRACTION_MODEL_FALLBACK` | `gemma3:latest`           | Fallback extraction model            |

### Search Tuning

| Parameter              | Default | Description                                |
| ---------------------- | ------- | ------------------------------------------ |
| `RRF_K`                | 60      | Reciprocal Rank Fusion constant            |
| `DEFAULT_ALPHA`        | 0.5     | Balance: 0 = pure BM25, 1 = pure vector    |
| `SIMILARITY_THRESHOLD` | 0.85    | Deduplication threshold                    |
| `RELATED_THRESHOLD`    | 0.7     | Auto-linking threshold for knowledge graph |

### Memory Lifecycle

| Parameter          | Default | Description                                        |
| ------------------ | ------- | -------------------------------------------------- |
| `DECAY_RATE`       | 0.95    | Importance multiplier per week of inactivity       |
| `DECAY_FLOOR`      | 1       | Minimum importance before archival eligible        |
| `ARCHIVE_AGE_DAYS` | 30      | Archive if at floor importance and older than this |

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                   Claude Code                        │
│                                                      │
│  search_memory("how did we handle auth?")            │
│         │                                            │
└─────────┼────────────────────────────────────────────┘
          │ MCP
          ▼
┌─────────────────────────────────────────────────────┐
│              claude-memory server                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Hot Cache │→ │  FTS5    │→ │ Hybrid RRF       │   │
│  │ (Tier 0)  │  │ (Tier 1) │  │ BM25+Vec (Tier 2)│   │
│  │ <1ms      │  │ ~10ms    │  │ ~50ms            │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │         SQLite (single file)                  │    │
│  │  FTS5 virtual table ← BM25 keyword search     │    │
│  │  sqlite-vec table   ← 768-dim vector search   │    │
│  │  memory_relations   ← knowledge graph         │    │
│  │  sessions           ← compaction recovery     │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  Ollama (local)                                      │
│  ├── nomic-embed-text → embeddings                   │
│  └── gemma3           → extraction                   │
└─────────────────────────────────────────────────────┘
```

**Tiered retrieval** stops early when enough results are found — most queries resolve from the hot cache or FTS5 without touching vectors at all.

**Smart alpha** auto-tunes the BM25/vector balance per query: code identifiers (`camelCase`, `snake_case`) get more BM25 weight; conceptual questions ("how to", "best practice") get more vector weight.

## Data Storage

```
~/.claude-memory/
├── db/
│   └── memory_v2.db      # All memories, embeddings, relations, sessions
├── sessions/
│   └── {project}.md       # Session state files for compaction recovery
└── logs/
    └── extraction.log     # Extraction activity log
```

Everything is in one SQLite file. Back it up, move it, inspect it with any SQLite tool.

## Categories

Memories are categorized for filtering and relevance:

| Category     | Use for                                                     |
| ------------ | ----------------------------------------------------------- |
| `decision`   | Architecture choices, technology selections, tradeoffs made |
| `fact`       | Project details, API specifics, environment info            |
| `preference` | User preferences, workflow choices                          |
| `gotcha`     | Bugs, non-obvious issues, workarounds                       |
| `convention` | File organization, naming patterns, code style              |
| `thread`     | Ongoing work items, open questions                          |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with `nomic-embed-text` model
- Claude Code with MCP support

## License

MIT - see [LICENSE](LICENSE)

---

Built for my own workflow. Shared because others wanted it. Issues and PRs welcome.
