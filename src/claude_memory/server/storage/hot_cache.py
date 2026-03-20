"""In-memory LRU cache for Tier 0 retrieval. Zero-token, sub-millisecond."""

import time
from collections import OrderedDict
from typing import Optional


class HotCache:
    """LRU cache with TTL for recently accessed memories."""

    def __init__(self, max_size: int = 100, ttl_minutes: int = 10):
        self.max_size = max_size
        self.ttl_seconds = ttl_minutes * 60
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._timestamps: dict[int, float] = {}

    def put(self, memory_id: int, memory: dict):
        """Add or update a memory in the cache."""
        if memory_id in self._cache:
            self._cache.move_to_end(memory_id)
        elif len(self._cache) >= self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

        self._cache[memory_id] = memory
        self._timestamps[memory_id] = time.time()

    def get(self, memory_id: int) -> Optional[dict]:
        """Get a memory by ID, returns None if expired or missing."""
        if memory_id not in self._cache:
            return None

        if self._is_expired(memory_id):
            self._evict(memory_id)
            return None

        self._cache.move_to_end(memory_id)
        return self._cache[memory_id]

    def search(self, query: str, namespace: str | None = None, limit: int = 5) -> list[dict]:
        """Simple substring search across cached items. Fast but imprecise."""
        self._evict_expired()
        query_lower = query.lower()
        results = []

        for mid, mem in reversed(self._cache.items()):
            if namespace and mem.get("namespace") != namespace:
                continue
            content = mem.get("content", "").lower()
            tags = " ".join(mem.get("tags", [])).lower()
            if query_lower in content or query_lower in tags:
                results.append(mem)
                if len(results) >= limit:
                    break

        return results

    def invalidate(self, memory_id: int):
        self._evict(memory_id)

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    def _is_expired(self, memory_id: int) -> bool:
        ts = self._timestamps.get(memory_id, 0)
        return (time.time() - ts) > self.ttl_seconds

    def _evict(self, memory_id: int):
        self._cache.pop(memory_id, None)
        self._timestamps.pop(memory_id, None)

    def _evict_expired(self):
        expired = [mid for mid in self._cache if self._is_expired(mid)]
        for mid in expired:
            self._evict(mid)
