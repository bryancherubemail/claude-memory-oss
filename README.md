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

This system gives Claude persistent memory across sessions. It has two parts:

1. **MCP Server** - Tools for Claude to search/store memories manually
2. **Auto-learning** - Automatically extracts insights from conversations using Ollama

### What Gets Remembered

- Architectural decisions
- Code conventions
- Tech stack choices
- Gotchas and edge cases
- Solutions that worked

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
│  │ memory_stats │    │ automatically│    │ compaction   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             │                                │
│                    ┌────────▼────────┐                      │
│                    │   SQLite DB     │                      │
│                    │  (~/.claude-    │                      │
│                    │   memory/data)  │                      │
│                    └─────────────────┘                      │
│                                                              │
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
git clone https://github.com/YOUR_USERNAME/claude-memory.git ~/.claude-memory
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
- **HuggingFace embeddings** on port 8080

The MCP server auto-detects these and uses them if available.

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
3. Parses structured output (DECISION, CONVENTION, GOTCHA, etc.)
4. Stores in SQLite with project tag

### What Gets Extracted

The LLM looks for high-value learnings:

```
DECISION: Using PostgreSQL for better JSON support
CONVENTION: All API endpoints validate input before processing
GOTCHA: GORM .Find() without .Limit() causes memory issues
SOLUTION: Fixed CORS by adding credentials header
```

Generic advice and obvious facts are filtered out.

## MCP Tools Reference

| Tool                   | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| `search_memory`        | Semantic search for relevant context        |
| `add_learning`         | Manually store a decision/convention/gotcha |
| `get_recent_learnings` | Get memories from last N days               |
| `memory_stats`         | Check what's stored                         |
| `sync_memories`        | Sync SQLite to ChromaDB                     |
| `wm_get_session`       | Get session context (for recovery)          |

## Project Awareness

Memories are tagged by git repository. When working in a project:

- You see that project's memories
- Plus global memories
- Other projects stay separate

This prevents "use PostgreSQL" in Project A from confusing Claude in Project B.

## Configuration

Environment variables:

| Variable              | Default                 | Purpose                    |
| --------------------- | ----------------------- | -------------------------- |
| `CLAUDE_MEMORY_DIR`   | `~/.claude-memory`      | Data directory             |
| `OLLAMA_MODEL`        | `gemma3:4b`             | Model for extraction       |
| `EXTRACTION_MAX_TIME` | `120`                   | Max seconds per extraction |
| `CHROMA_URL`          | `http://localhost:8000` | ChromaDB URL               |
| `EMBEDDINGS_URL`      | `http://localhost:8080` | Embeddings service         |

## Data Storage

All data stays local:

```
~/.claude-memory/
├── data/
│   └── memory.db          # SQLite database
├── sessions/
│   └── {project}.md       # Session state files
├── logs/
│   └── extraction.log     # Auto-learn logs
└── locks/
    └── extraction.lock    # Prevents parallel runs
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
