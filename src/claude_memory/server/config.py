"""Central configuration for Claude Memory v2."""

import os
from pathlib import Path

# Paths
MEMORY_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
DB_PATH = MEMORY_DIR / "db" / "memory_v2.db"
SESSIONS_DIR = MEMORY_DIR / "sessions"
LOGS_DIR = MEMORY_DIR / "logs"

# Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_DIM = 768  # nomic-embed-text output dimension
EXTRACTION_MODEL_PRIMARY = os.environ.get("EXTRACTION_MODEL", "gemma3:27b")
EXTRACTION_MODEL_FALLBACK = os.environ.get("EXTRACTION_MODEL_FALLBACK", "gemma3:latest")

# Search tuning
RRF_K = 60  # Standard RRF constant
DEFAULT_ALPHA = 0.5  # Balance between BM25 (0) and vector (1)
SIMILARITY_THRESHOLD = 0.85  # For deduplication
RELATED_THRESHOLD = 0.7  # For auto-linking relations

# Hot cache
HOT_CACHE_MAX_SIZE = 100
HOT_CACHE_TTL_MINUTES = 10

# Tiered retrieval token caps
TIER_TOKEN_CAPS = {
    "minimal": 100,
    "standard": 300,
    "full": 600,
}

# Importance decay
DECAY_RATE = 0.95  # Multiplied per week of inactivity
DECAY_FLOOR = 1  # Minimum importance before archival
ARCHIVE_AGE_DAYS = 30  # Archive if importance < DECAY_FLOOR and older than this

# Valid categories
CATEGORIES = {"decision", "fact", "preference", "gotcha", "convention", "thread"}

# Ensure directories exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
