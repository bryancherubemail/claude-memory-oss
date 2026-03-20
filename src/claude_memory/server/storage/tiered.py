"""
Tiered retrieval: progressive loading from fastest/cheapest to deepest.

Tier 0: Hot cache (in-memory, <1ms, 0 tokens)
Tier 1: SQLite FTS5 BM25 (local, <10ms, ~100 tokens)
Tier 2: SQLite-vec hybrid search (local, <50ms, ~300 tokens)
"""

from ..config import TIER_TOKEN_CAPS
from ..utils.compression import estimate_tokens
from ..utils.fusion import hybrid_search, compute_smart_alpha
from ..utils.embeddings import get_embedding


class TieredRetrieval:
    def __init__(self, hot_cache, session_db):
        self.hot_cache = hot_cache
        self.session_db = session_db

    async def recall(
        self,
        query: str,
        namespace: str,
        depth: str = "standard",
        min_results: int = 5,
    ) -> dict:
        """
        Retrieve context using tiered approach.
        Stops when min_results reached or token cap exceeded.

        depth: minimal (~100 tokens), standard (~300), full (~600)
        """
        max_tokens = TIER_TOKEN_CAPS.get(depth, 300)
        results = []
        tokens_used = 0
        tiers_searched = []

        # Tier 0: Hot cache
        tier0 = self.hot_cache.search(query, namespace, limit=min_results)
        if tier0:
            results.extend(tier0)
            tiers_searched.append("hot_cache")

        if len(results) >= min_results:
            return self._response(results[:min_results], tiers_searched, tokens_used)

        # Tier 1: FTS5 BM25
        exclude = [r["id"] for r in results]
        remaining = min_results - len(results)
        tier1 = self.session_db.fts_search(
            query=query, namespace=namespace, limit=remaining * 2, exclude_ids=exclude,
        )
        if tier1:
            results.extend(tier1)
            tiers_searched.append("fts5_bm25")
            tokens_used += sum(estimate_tokens(r.get("content", "")) for r in tier1)

        if len(results) >= min_results or tokens_used >= max_tokens:
            return self._response(results[:min_results], tiers_searched, tokens_used)

        # Tier 2: Hybrid search (BM25 + vector via RRF)
        embedding = await get_embedding(query)
        exclude = [r["id"] for r in results]
        remaining = min_results - len(results)

        tier2 = await hybrid_search(
            query=query,
            namespace=namespace,
            session_db=self.session_db,
            embedding=embedding,
            limit=remaining * 2,
            exclude_ids=exclude,
        )
        if tier2:
            results.extend(tier2)
            tiers_searched.append("hybrid_rrf")
            tokens_used += sum(estimate_tokens(r.get("content", "")) for r in tier2)

        return self._response(results[:min_results * 2], tiers_searched, tokens_used)

    def _response(self, results, tiers, tokens):
        # Update hot cache with retrieved items
        for r in results:
            self.hot_cache.put(r["id"], r)
            self.session_db.touch(r["id"])

        return {
            "results": results,
            "tiers_searched": tiers,
            "tokens_used": tokens,
            "count": len(results),
        }
