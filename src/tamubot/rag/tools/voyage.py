"""Voyage AI tool — embedding and reranking.

Exposes:
  embed_query(text) -> list[float]
  rerank(query, chunks, top_k) -> list[dict]
  stratified_select(chunks, k) -> list[dict]

Canonical location: rag/tools/voyage.py
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, defaultdict
from typing import Optional

import voyageai
from langfuse import observe

from tamubot.core import config

EMBEDDING_MODEL = "voyage-3"
RERANK_MODEL = config.VOYAGE_RERANK_MODEL

_voyage: Optional[voyageai.Client] = None


def _get_client() -> voyageai.Client:
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage


# ---------------------------------------------------------------------------
# Embedding cache — thread-safe OrderedDict bounds memory to 512 strings.
# Implements Double-Checked Locking to guarantee exactly 1 Voyage API call
# per unique query string, structurally preventing the thundering herd problem
# when multiple retrieval workers fan out simultaneously on a cold cache miss.
# ---------------------------------------------------------------------------

_EMBED_CACHE_MAXSIZE = 512
_embed_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
_embed_lock = threading.Lock()


def _embed_query_cached(text: str) -> tuple[float, ...]:
    """Return a cached embedding tuple, fetching from Voyage API if missing.

    Uses Double-Checked Locking:
    1. Outer lock-free read (fast path for cache hits)
    2. Lock acquisition on miss
    3. Inner re-check (prevents duplicate API calls from racing threads)
    4. API fetch and LRU eviction if bounds exceeded
    """
    # 1. Outer check (Fast path)
    cached = _embed_cache.get(text)
    if cached is not None:
        return cached

    # 2. Lock acquire (Slow path for cold misses)
    with _embed_lock:
        # 3. Inner re-check
        cached = _embed_cache.get(text)
        if cached is not None:
            # Another thread handled it while we waited
            return cached

        client = _get_client()
        result = client.embed([text], model=EMBEDDING_MODEL, input_type="query")
        emb = tuple(result.embeddings[0])

        # 4. Insert and evict LRU
        _embed_cache[text] = emb
        if len(_embed_cache) > _EMBED_CACHE_MAXSIZE:
            _embed_cache.popitem(last=False)  # pop oldest

        return emb


@observe(name="node.retrieval.embed")
def embed_query(text: str) -> list[float]:
    """Embed a query string using Voyage AI voyage-3.

    Results are cached in-process (up to 512 strings, thread-safe) so that
    parallel retrieval threads share the embedding and strictly avoid
    duplicate API round-trips for the same string.
    """
    return list(_embed_query_cached(text))


def knee_filter(
    chunks: list[dict],
    *,
    min_floor: int = 2,
    abs_threshold: float = 0.15,
    min_gap_fallback: float = 0.05,
) -> list[dict]:
    """Cut reranked chunks at the first meaningful score drop.

    Primary: cut at the first consecutive gap exceeding abs_threshold.
    Fallback: cut at the largest gap, but only if it exceeds min_gap_fallback
              (avoids over-filtering when all scores are uniformly high).
    Always keeps at least min_floor chunks regardless of gap size.
    """
    if len(chunks) <= min_floor:
        return chunks

    scores = [c.get("score", 0.0) for c in chunks]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    # Primary: first absolute gap exceeding threshold
    for i, gap in enumerate(gaps):
        if gap > abs_threshold:
            return chunks[: max(i + 1, min_floor)]

    # Fallback: largest gap (only if meaningful noise floor exceeded)
    max_gap = max(gaps)
    if max_gap > min_gap_fallback:
        cut = gaps.index(max_gap)
        return chunks[: max(cut + 1, min_floor)]

    return chunks


@observe(name="node.retrieval.rerank")
def rerank(query: str, chunks: list[dict], top_k: int, *, apply_knee: bool = True) -> list[dict]:
    """Cross-encoder rerank chunks by relevance to query, return top_k.

    Preserves all original chunk fields. Adds/updates 'score' field.
    Returns chunks sorted descending by score.
    Always drops chunks below config.RERANK_SCORE_THRESHOLD (floor: config.RERANK_SCORE_MIN_CHUNKS).
    If config.RERANK_KNEE_ENABLED and apply_knee, additionally applies knee-point filtering.
    Falls back to original order (sliced to top_k) on any Voyage error.
    """
    from langfuse import get_client as _lf

    if not chunks:
        return []
    top_k = min(top_k, len(chunks))
    client = _get_client()
    texts = [c.get("content", "") for c in chunks]
    try:
        response = client.rerank(query=query, documents=texts, model=RERANK_MODEL, top_k=top_k)
        results = []
        for item in response.results:
            chunk = dict(chunks[item.index])
            chunk["score"] = item.relevance_score
            results.append(chunk)

        pre_rerank_count = len(chunks)
        post_rerank_count = len(results)
        all_scores = [c["score"] for c in results]

        # Fixed score threshold — always active
        min_chunks = config.RERANK_SCORE_MIN_CHUNKS
        filtered = [c for c in results if c.get("score", 0.0) >= config.RERANK_SCORE_THRESHOLD]
        pre_threshold_count = len(results)
        results = filtered if len(filtered) >= min_chunks else results[:min_chunks]
        post_threshold_count = len(results)

        # Knee-point filter — optional, off by default
        knee_applied = config.RERANK_KNEE_ENABLED and apply_knee
        pre_knee_count = len(results)
        if knee_applied:
            results = knee_filter(
                results,
                min_floor=config.RERANK_KNEE_MIN_CHUNKS,
                abs_threshold=config.RERANK_KNEE_ABS_THRESHOLD,
                min_gap_fallback=config.RERANK_KNEE_MIN_GAP_FALLBACK,
            )
        post_knee_count = len(results)

        # Trace metadata: full reranking breakdown
        final_scores = [c.get("score", 0.0) for c in results]
        chunk_summary = [
            {
                "course_id": c.get("course_id", ""),
                "category": c.get("category", ""),
                "score": round(c.get("score", 0.0), 4),
                "rrf_source": c.get("rrf_source", "unknown"),
            }
            for c in results
        ]
        try:
            _lf().update_current_span(
                metadata={
                    "pre_rerank_count": pre_rerank_count,
                    "post_rerank_count": post_rerank_count,
                    "scores_all": [round(s, 4) for s in all_scores],
                    "score_max": round(max(all_scores), 4) if all_scores else 0,
                    "score_min": round(min(all_scores), 4) if all_scores else 0,
                    "threshold": {
                        "value": config.RERANK_SCORE_THRESHOLD,
                        "pre_count": pre_threshold_count,
                        "post_count": post_threshold_count,
                        "min_chunks_floor": min_chunks,
                    },
                    "knee": {
                        "applied": knee_applied,
                        "pre_count": pre_knee_count,
                        "post_count": post_knee_count,
                    },
                    "final_chunks": chunk_summary,
                    "final_scores": [round(s, 4) for s in final_scores],
                }
            )
        except Exception:
            pass

        return results
    except Exception:
        return chunks[:top_k]


def stratified_select(chunks: list[dict], k: int) -> list[dict]:
    """Select up to k chunks with at most ceil(k/n_courses) per course.

    Used as a fallback when Voyage reranking is unavailable.
    """
    if not chunks or k <= 0:
        return []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        buckets[c.get("course_id", "_")].append(c)
    n = len(buckets)
    per_course = math.ceil(k / n) if n else k
    selected = []
    for course_chunks in buckets.values():
        selected.extend(course_chunks[:per_course])
    return selected[:k]
