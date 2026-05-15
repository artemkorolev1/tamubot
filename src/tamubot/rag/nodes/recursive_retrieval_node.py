"""Recursive retrieval node — first-pass hybrid search on anchor course(s)."""

from __future__ import annotations

from tamubot.core import config
from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware
from tamubot.rag.state.pipeline_state import PipelineState
from tamubot.rag.utils import compute_dynamic_k, make_cache_key


@timing_middleware
@error_guard_middleware
def recursive_retrieval_node(state: PipelineState) -> dict:
    """Fetch anchor course chunks via hybrid search. Only runs on recursive path."""
    from tamubot.rag.tools.mongo import hybrid_search
    from tamubot.rag.tools.voyage import rerank as voyage_rerank

    course_ids = state.get("course_ids", [])
    rewritten_query = state.get("rewritten_query") or state.get("query", "")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("recursive_retrieval")

    dk = compute_dynamic_k("recursive", len(course_ids), recursive_anchor=True)
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]

    # Cache check
    if config.SESSION_CACHE_ENABLED:
        cache_key = make_cache_key("recursive_anchor", course_ids, rewritten_query)
        cached = state.get("retrieval_cache", {}).get(cache_key)
        if cached is not None:
            node_trace.append("retrieval_cache_hit")
            return {"recursive_chunks": cached, "node_trace": node_trace}

    try:
        all_chunks = []
        for cid in course_ids:
            chunks = hybrid_search(rewritten_query, cid, retrieve_k)
            all_chunks.extend(chunks)
        reranked = voyage_rerank(rewritten_query, all_chunks, top_k=rerank_k)

        retrieval_cache_update = {}
        if config.SESSION_CACHE_ENABLED:
            cache_key = make_cache_key("recursive_anchor", course_ids, rewritten_query)
            existing = state.get("retrieval_cache", {})
            retrieval_cache_update = {**existing, cache_key: reranked}

        return {
            "recursive_chunks": reranked,
            "retrieval_cache": retrieval_cache_update,
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "recursive_chunks": [],
            "error": f"Recursive retrieval failed: {e}",
            "node_trace": node_trace,
        }
