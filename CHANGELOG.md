# Changelog

## 2.1.0 (2026-04-14)

- Removed legacy v1 artifacts (Docker/ChromaDB stack, old CLI, v1 server)
- Rewrote documentation for quick start
- PyPI-ready packaging with proper metadata
- Fixed hook examples for v2 extraction pipeline
- Cleaned up repository structure

## 2.0.0 (2026-03-20)

- Complete rewrite: SQLite-only backend (no Docker, no ChromaDB, no TEI)
- Hybrid BM25 + vector search with Reciprocal Rank Fusion
- Knowledge graph with automatic relation detection
- Importance decay for memory relevance
- Tiered retrieval (hot cache → FTS5 → hybrid)
- Predictive prefetch based on file activity
- 13 MCP tools
- Zero external infrastructure — just SQLite + Ollama

## 1.0.0 (2025-12)

- Initial release with ChromaDB + Docker + TEI embeddings
