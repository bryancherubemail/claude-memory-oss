# claude-memory

Local long-term memory for Claude Code via MCP.

Stores learnings, decisions, and context in SQLite with optional semantic search via ChromaDB.

## Why

I built this because I didn't want to pay for something that should be simple. Maybe it helps you too.

## Status

This is a personal project shared as-is. No support, no guarantees, no roadmap. Fork it, extend it, make it your own.

## Install

```bash
pip install mcp httpx
```

Clone this repo, then add to your Claude Code MCP config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "claude-memory": {
      "command": "python",
      "args": ["/path/to/claude-memory-oss/src/claude_memory/server.py"]
    }
  }
}
```

## Optional: Semantic Search

For semantic search (instead of just text matching), run the Docker stack:

```bash
cd docker
docker compose up -d
```

This starts:

- ChromaDB (vector database) on port 8000
- HuggingFace text-embeddings-inference on port 8080

The server auto-detects these and uses them if available. Without them, it falls back to SQLite text search.

## What It Does

- **search_memory** - Find relevant past context
- **add_learning** - Store decisions, conventions, gotchas
- **get_recent_learnings** - See what was learned recently
- **memory_stats** - Check what's stored
- **sync_memories** - Sync SQLite to ChromaDB
- **wm_get_session** - Get session context after compaction

## Project Awareness

Memories are tagged by git repo (or directory name). When you're in a project, you see that project's memories plus global ones.

## Configuration

Environment variables:

- `CLAUDE_MEMORY_DIR` - Where to store data (default: `~/.claude-memory`)
- `CHROMA_URL` - ChromaDB URL (default: `http://localhost:8000`)
- `EMBEDDINGS_URL` - Embeddings service URL (default: `http://localhost:8080`)

## License

MIT
