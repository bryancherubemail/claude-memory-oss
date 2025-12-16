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

This MCP server gives Claude a persistent memory. It stores learnings in a local SQLite database and retrieves relevant context when you need it.

**What gets remembered:**

- Architectural decisions
- Code conventions
- Tech stack choices
- Gotchas and edge cases
- Solutions that worked

**How it retrieves:**

- Semantic search (if Docker services are running)
- Text search fallback (always works, zero dependencies)
- Project-aware: memories are tagged by git repo, so project-specific context stays with that project

## How It Works

```
You: "Add authentication to the API"

Claude (with memory):
  - Recalls you decided on JWT tokens last week
  - Knows you're using Express.js
  - Remembers the token expiration gotcha you hit
  - Gives advice that fits YOUR project's context
```

The server exposes these tools to Claude:

| Tool                   | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| `search_memory`        | Find relevant past context by semantic similarity |
| `add_learning`         | Store a decision, convention, or gotcha           |
| `get_recent_learnings` | See what was learned in the last N days           |
| `memory_stats`         | Check how many memories exist, by type            |
| `sync_memories`        | One-time sync to ChromaDB for semantic search     |
| `wm_get_session`       | Restore context after session compaction          |

## Why This Exists

There are paid services that do this (Mem0, Zep, etc). They charge $25+/month for what amounts to SQLite + embeddings.

I built this because:

1. The core functionality is simple
2. My data should stay on my machine
3. I shouldn't pay rent for a database

This is that system, shared for anyone who feels the same way.

## Install

**Requirements:** Python 3.10+, `mcp` and `httpx` packages

```bash
pip install mcp httpx
```

Clone this repo, then add to your Claude Code config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "claude-memory": {
      "command": "python",
      "args": ["/path/to/claude-memory/src/claude_memory/server.py"]
    }
  }
}
```

Restart Claude Code. The memory tools are now available.

## Optional: Semantic Search

By default, the server uses SQLite text search (works fine, zero setup).

For semantic search (understands meaning, not just keywords), run the Docker stack:

```bash
cd docker
docker compose up -d
```

This starts:

- **ChromaDB** (vector database) on port 8000
- **HuggingFace text-embeddings-inference** on port 8080 (uses `all-MiniLM-L6-v2`)

The server auto-detects these services. If they're not running, it falls back to text search.

## Project Awareness

Memories are automatically tagged with your current git repository name. When you're working in a project:

- You see that project's memories
- Plus global memories (from `~/Claude` or non-git directories)
- Other projects' memories stay separate

This means "use PostgreSQL" in Project A won't confuse Claude when you're in Project B (which uses MongoDB).

## Configuration

Environment variables (all optional):

| Variable            | Default                 | Purpose             |
| ------------------- | ----------------------- | ------------------- |
| `CLAUDE_MEMORY_DIR` | `~/.claude-memory`      | Where to store data |
| `CHROMA_URL`        | `http://localhost:8000` | ChromaDB connection |
| `EMBEDDINGS_URL`    | `http://localhost:8080` | Embeddings service  |

## Data Storage

All data stays local:

```
~/.claude-memory/
├── data/
│   └── memory.db      # SQLite database (your memories)
└── sessions/
    └── {project}.md   # Session state files
```

Nothing leaves your machine. No telemetry, no cloud sync, no API keys needed.

## Status

This is a personal project shared as-is.

- No support
- No guarantees
- No roadmap

Fork it, extend it, make it your own. If it breaks, you get to keep both pieces.

## License

MIT - do whatever you want with it.
