# claude-memory

Local long-term memory for Claude Code via MCP.

## The Problem

Every time you start a new Claude Code session, Claude starts fresh. It doesn't remember:

- Decisions you made yesterday ("we're using PostgreSQL, not MySQL")
- Conventions you established ("always use camelCase in this project")
- Gotchas you discovered ("the auth middleware must come before the rate limiter")
- Solutions that worked ("fixed the CORS issue by adding the credentials header")

You end up re-explaining context, or worse, Claude suggests things you've already tried and rejected.

## The Solution

This system gives Claude persistent memory across sessions. It has three parts:

1. **MCP Server** - Tools for Claude to search/store memories with scope awareness
2. **Auto-learning** - Automatically extracts insights from conversations using Ollama
3. **CLI Tool** - `mem` command for managing and querying memories

### What Gets Remembered

- Architectural decisions (project-specific or global)
- Code conventions
- Tech stack choices
- Language-specific gotchas (Python, JavaScript, Go, Rust, SQL)
- Solutions that worked

### Key Features

- **Scope-aware memories** - Project-specific, global, or language-specific (e.g., `language:python`)
- **BM25 hybrid search** - Combines semantic similarity with keyword matching
- **Access count tracking** - Frequently accessed memories rank higher
- **Duplicate detection** - Find and consolidate similar memories
- **Session state recovery** - Preserves context across compaction

### How It Works

```
You: "Add authentication to the API"

Claude (with memory):
  - Recalls you decided on JWT tokens last week
  - Knows you're using Express.js
  - Remembers the token expiration gotcha you hit
  - Gives advice that fits YOUR project's context
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MCP Server  │    │  Auto-learn  │    │Session State │  │
│  │              │    │   (Ollama)   │    │   Recovery   │  │
│  │ search_memory│    │              │    │              │  │
│  │ add_learning │    │ Extracts     │    │ Preserves    │  │
│  │ get_recent   │    │ insights     │    │ context on   │  │
│  │ consolidate  │    │ w/ scope     │    │ compaction   │  │
│  │ memory_stats │    │ awareness    │    │              │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             │                                │
│                    ┌────────▼────────┐                      │
│                    │   SQLite DB     │                      │
│                    │  (~/.claude-    │                      │
│                    │   memory/data)  │                      │
│                    └────────┬────────┘                      │
│                             │                                │
│  Optional: ChromaDB + Embeddings for semantic search        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install the MCP Server

```bash
pip install mcp httpx
```

Clone this repo to `~/.claude-memory` (or anywhere):

```bash
git clone https://github.com/bryancherubemail/claude-memory-oss.git ~/.claude-memory
```

Add to Claude Code config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "claude-memory": {
      "command": "python",
      "args": ["~/.claude-memory/src/claude_memory/server.py"]
    }
  }
}
```

Restart Claude Code. Memory tools are now available.

### 2. Enable Auto-learning (Recommended)

Install Ollama and pull a model:

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (gemma3:4b is fast, gemma3:12b is better)
ollama pull gemma3:4b
```

Make scripts executable:

```bash
chmod +x ~/.claude-memory/scripts/*
```

Add hooks to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude-memory/scripts/async-extract --incremental",
            "timeout": 5,
            "statusMessage": "Extracting learnings..."
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude-memory/src/claude_memory/auto_learn.py --final",
            "timeout": 30,
            "statusMessage": "Final extraction..."
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude-memory/scripts/inject-session-state",
            "timeout": 5,
            "statusMessage": "Restoring context..."
          }
        ]
      }
    ]
  }
}
```

See `hooks-example.json` for the full example.

### 3. Optional: Semantic Search

For semantic search (meaning-based, not just keywords), run the Docker stack:

```bash
cd ~/.claude-memory/docker
docker compose up -d
```

This starts:

- **ChromaDB** on port 8000
- **Ollama** for embeddings (using `nomic-embed-text`)

The MCP server auto-detects these and uses them if available.

## CLI Tool (`mem`)

The `mem` command provides direct access to your memories:

```bash
# Add to PATH
export PATH="$PATH:~/.claude-memory/scripts"

# Or run directly
~/.claude-memory/scripts/mem stats
```

### Available Commands

| Command              | Purpose                           |
| -------------------- | --------------------------------- |
| `mem search <query>` | Search memories by keyword        |
| `mem stats`          | Show memory statistics            |
| `mem recent [days]`  | Show memories from last N days    |
| `mem consolidate`    | Find duplicate memories           |
| `mem prune --days N` | Remove old, unused memories       |
| `mem scope-fix`      | Auto-detect and fix memory scopes |
| `mem export [file]`  | Export all memories to JSON       |

### Examples

```bash
# Search for authentication-related memories
mem search authentication

# See what's stored
mem stats

# Find duplicates (preview)
mem consolidate

# Actually remove duplicates
mem consolidate --apply

# Clean up old unused memories (older than 90 days with 0 access)
mem prune --days 90 --apply

# Fix scope for existing memories
mem scope-fix --apply
```

## How Auto-learning Works

The system hooks into Claude Code's lifecycle:

| Hook                     | When                      | What Happens                            |
| ------------------------ | ------------------------- | --------------------------------------- |
| `PreCompact`             | Before context compaction | Extracts learnings, saves session state |
| `SessionEnd`             | When you close Claude     | Final extraction pass                   |
| `SessionStart` (compact) | After compaction          | Injects saved session state             |

### Extraction Process

1. Reads Claude's conversation logs (`~/.claude/projects/`)
2. Sends to Ollama with extraction prompt
3. Parses structured output with scope (DECISION|SCOPE|CONFIDENCE|content)
4. Stores in SQLite with project and scope tags

### What Gets Extracted

The LLM looks for high-value learnings with scope awareness:

```
DECISION|PROJECT|HIGH|Using PostgreSQL for better JSON support
CONVENTION|PROJECT|MEDIUM|All API endpoints validate input before processing
GOTCHA|PYTHON|HIGH|asyncio.gather() silently swallows exceptions without return_exceptions=True
SOLUTION|GLOBAL|HIGH|Fixed CORS by ensuring OPTIONS returns 204, not 200 with body
GOTCHA|GO|HIGH|GORM .Find() without .Limit() loads entire table into memory
```

Generic advice and obvious facts are filtered out. LOW confidence items are skipped.

## MCP Tools Reference

| Tool                   | Purpose                              |
| ---------------------- | ------------------------------------ |
| `search_memory`        | Semantic search with scope awareness |
| `add_learning`         | Store a decision/convention/gotcha   |
| `get_recent_learnings` | Get memories from last N days        |
| `consolidate_memories` | Find and remove duplicate memories   |
| `memory_stats`         | Check what's stored (by type, scope) |
| `sync_memories`        | Sync SQLite to ChromaDB              |
| `wm_get_session`       | Get session context (for recovery)   |

## Scope Awareness

Memories are tagged with scope:

| Scope             | Description           | Example                                    |
| ----------------- | --------------------- | ------------------------------------------ |
| `project`         | Current project only  | "Using Supabase for auth in this project"  |
| `global`          | Applies everywhere    | "Always validate user input at boundaries" |
| `language:python` | Python-specific       | "asyncio.gather needs return_exceptions"   |
| `language:go`     | Go-specific           | "GORM Find without Limit causes OOM"       |
| `language:sql`    | Database/SQL-specific | "PostgreSQL uses $1, MySQL uses ?"         |

When searching:

- Current project memories get priority boost (+0.25)
- Global memories are always included (+0.1)
- Language-specific memories match when the query relates to that language (+0.15)

## Project Awareness

Memories are tagged by git repository. When working in a project:

- You see that project's memories
- Plus global memories
- Plus language-specific memories for detected languages
- Other projects stay separate

This prevents "use PostgreSQL" in Project A from confusing Claude in Project B.

## Configuration

Environment variables:

| Variable            | Default                   | Purpose              |
| ------------------- | ------------------------- | -------------------- |
| `CLAUDE_MEMORY_DIR` | `~/.claude-memory`        | Data directory       |
| `OLLAMA_MODEL`      | `gemma3:4b`               | Model for extraction |
| `OLLAMA_URL`        | `http://localhost:11434`  | Ollama API URL       |
| `EMBEDDING_MODEL`   | `nomic-embed-text:latest` | Model for embeddings |
| `CHROMA_URL`        | `http://localhost:8000`   | ChromaDB URL         |

## Data Storage

All data stays local:

```
~/.claude-memory/
├── data/
│   └── memory.db          # SQLite database
├── sessions/
│   └── {project}.md       # Session state files
├── scripts/
│   ├── mem                # CLI tool
│   ├── async-extract      # Background extraction
│   └── inject-session-state
└── src/
    └── claude_memory/
        ├── server.py      # MCP server
        └── auto_learn.py  # Extraction logic
```

Nothing leaves your machine. No telemetry, no cloud.

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
