"""Hybrid search: BM25 + vector with Reciprocal Rank Fusion and smart alpha."""

import re
from typing import Optional

from ..config import RRF_K, DEFAULT_ALPHA


def compute_smart_alpha(query: str) -> float:
    """
    Auto-tune alpha based on query characteristics.
    Alpha: 0 = pure BM25 (keyword), 1 = pure vector (semantic).
    """
    alpha = DEFAULT_ALPHA

    # Technical indicators -> lower alpha (more BM25, exact matches matter)
    technical_patterns = [
        r'\b[A-Z][a-z]+[A-Z]\w*\b',   # camelCase
        r'\b[a-z]+_[a-z]+\b',          # snake_case
        r'\b\d+\.\d+\.\d+\b',          # version numbers
        r'\b[A-Z]{2,}\b',              # ACRONYMS
        r'error|exception|bug|fix',     # error-related
        r'\.(py|js|ts|go|rs|sql)\b',   # file extensions
        r'port \d+|localhost',          # network
        r'function|method|class|def',   # code terms
    ]

    for pattern in technical_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            alpha -= 0.1

    # Conceptual indicators -> higher alpha (more semantic)
    conceptual_patterns = [
        r'\bhow (do|does|to|can|should)\b',
        r'\bwhy (is|are|did|does)\b',
        r'\bwhat (is|are|does|should)\b',
        r'\bbest practice|recommend|suggest\b',
        r'\bsimilar to|like|alternative\b',
    ]

    for pattern in conceptual_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            alpha += 0.1

    return max(0.2, min(0.8, alpha))


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists using RRF.
    Returns list of (memory_id, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}

    for rank_list in rankings:
        for rank, item in enumerate(rank_list, start=1):
            item_id = item.get("id")
            if item_id is not None:
                scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def hybrid_search(
    query: str,
    namespace: str,
    session_db,
    embedding: list[float] | None = None,
    alpha: Optional[float] = None,
    limit: int = 10,
    include_global: bool = True,
    exclude_ids: list[int] | None = None,
) -> list[dict]:
    """
    Perform hybrid BM25 + vector search with RRF fusion.

    If embedding is None, falls back to BM25-only.
    Alpha controls the weighting but RRF naturally balances both signals.
    """
    if alpha is None:
        alpha = compute_smart_alpha(query)

    fetch_limit = limit * 3

    # BM25 search
    bm25_results = session_db.fts_search(
        query=query,
        namespace=namespace,
        limit=fetch_limit,
        exclude_ids=exclude_ids,
    )

    # Also search global namespace if requested
    if include_global and namespace != "global":
        global_bm25 = session_db.fts_search(
            query=query,
            namespace="global",
            limit=fetch_limit // 2,
            exclude_ids=exclude_ids,
        )
        bm25_results.extend(global_bm25)

    # Vector search (if embedding available)
    vec_results = []
    if embedding is not None:
        vec_results = session_db.vector_search(
            embedding=embedding,
            namespace=namespace,
            limit=fetch_limit,
            exclude_ids=exclude_ids,
        )

        if include_global and namespace != "global":
            global_vec = session_db.vector_search(
                embedding=embedding,
                namespace="global",
                limit=fetch_limit // 2,
                exclude_ids=exclude_ids,
            )
            vec_results.extend(global_vec)

    # If only one signal, return directly
    if not vec_results:
        return bm25_results[:limit]
    if not bm25_results:
        return vec_results[:limit]

    # Fuse with RRF
    fused = reciprocal_rank_fusion([bm25_results, vec_results])

    # Fetch full memory data for top results
    top_ids = [mid for mid, _ in fused[:limit]]
    results = session_db.get_by_ids(top_ids)

    # Add fusion scores and sort
    score_map = dict(fused)
    for r in results:
        r["_fusion_score"] = score_map.get(r["id"], 0)
    results.sort(key=lambda x: x.get("_fusion_score", 0), reverse=True)

    return results
