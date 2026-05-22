"""Retrieval node — handles hybrid_course and semantic_general retrieval passes.

Multi-subquery fanout: when the router emits >1 entry in ``state["subqueries"]``,
each variant runs the retrieve+rerank pipeline independently and the per-variant
results are merged via Reciprocal Rank Fusion. Single-subquery state uses the
existing single-query path unchanged.
"""

from __future__ import annotations

import contextvars
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from langfuse import observe

from tamubot.core import config
from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware
from tamubot.rag.router import deduplicate_chunks
from tamubot.rag.state.pipeline_state import PipelineState
from tamubot.rag.tools.rrf import rrf_fuse
from tamubot.rag.utils import compute_dynamic_k, make_cache_key

# Per-course cap applied on the semantic_general branch to prevent a single
# course from dominating the rerank output (the hybrid_course branch is scoped
# to user-named courses, so concentration is expected there).
MAX_CHUNKS_PER_COURSE = 4


def _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query=None):
    """Backward-compat shim — use make_cache_key() directly."""
    return make_cache_key("retrieval", course_ids, rewritten_query)


def _chunk_rrf_key(chunk: dict):
    """Stable identity for RRF deduplication across subquery variants.

    Chunks coming out of ``hybrid_search`` / ``semantic_search`` carry neither
    ``_id`` nor ``chunk_id`` (the Mongo projection strips them), so we fall
    back to the same composite key ``deduplicate_chunks`` already uses.
    """
    cid = chunk.get("course_id", "")
    idx = chunk.get("chunk_index", -1)
    hdr = chunk.get("header_path", "")
    return (cid, idx, hdr)


def _build_context_primer(course_ids: list[str]) -> str | None:
    """Fetch per-course summaries and concatenate into a single primer string.

    Returns None when the flag is off, when no course is named, or when no
    summary is found. Missing summaries are silently skipped — telemetry is
    already emitted by ``get_course_summary_chunks``.
    """
    if not config.SUMMARY_AS_PRIMER or not course_ids:
        return None
    from tamubot.rag.tools.mongo import get_course_summary_chunks

    chunks = get_course_summary_chunks(course_ids) or []
    if not chunks:
        return None
    parts: list[str] = []
    for doc in chunks:
        cid = doc.get("course_id", "")
        content = doc.get("content", "") or ""
        if not content:
            continue
        parts.append(f"[Course {cid}]\n{content}")
    if not parts:
        return None
    return "\n\n".join(parts)


def _cap_per_course(chunks: list[dict], cap: int) -> list[dict]:
    """Walk ``chunks`` in order, keeping at most ``cap`` items per ``course_id``.

    Preserves the input rank order. Chunks missing ``course_id`` are bucketed
    under the empty-string key (kept, not dropped).
    """
    counts: dict[str, int] = {}
    capped: list[dict] = []
    for chunk in chunks:
        key = chunk.get("course_id", "")
        if counts.get(key, 0) < cap:
            capped.append(chunk)
            counts[key] = counts.get(key, 0) + 1
    return capped


def _hybrid_course_for_query(
    hybrid_search,
    voyage_rerank,
    query: str,
    course_ids: list[str],
    retrieve_k: int,
    rerank_k: int,
    apply_knee: bool,
) -> tuple[list[dict], list[str]]:
    """Retrieve + rerank for a single (query, course_ids) pair. Returns (reranked, errors)."""
    all_chunks: list[dict] = []
    errors: list[str] = []

    if len(course_ids) <= 1:
        if course_ids:
            all_chunks = hybrid_search(query, course_ids[0], retrieve_k)
    else:
        original_ctx = contextvars.copy_context()
        worker_ctxs = [original_ctx.run(contextvars.copy_context) for _ in course_ids]
        futures = {}
        with ThreadPoolExecutor(max_workers=min(len(course_ids), 8)) as executor:
            for wctx, cid in zip(worker_ctxs, course_ids):
                futures[executor.submit(wctx.run, hybrid_search, query, cid, retrieve_k)] = cid
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    all_chunks.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    msg = f"hybrid_search failed for {cid}: {exc}"
                    errors.append(msg)
                    logging.warning("retrieval_node: %s", msg)

    reranked = voyage_rerank(query, all_chunks, top_k=rerank_k, apply_knee=apply_knee)
    return reranked, errors


@observe(name="node.retrieval")
@timing_middleware
@error_guard_middleware
def retrieval_node(state: PipelineState) -> dict:
    """Execute retrieval based on function type."""
    from tamubot.rag.tools.mongo import hybrid_search, semantic_search
    from tamubot.rag.tools.voyage import rerank as voyage_rerank

    function = state.get("function", "out_of_scope")
    course_ids = state.get("course_ids", [])
    rewritten_query = state.get("rewritten_query") or state.get("query", "")
    subqueries: list[str] = list(state.get("subqueries") or [])
    if not subqueries:
        subqueries = [rewritten_query]
    node_trace = list(state.get("node_trace", []))
    node_trace.append("retrieval")

    dk = compute_dynamic_k(function, len(course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]
    apply_knee = not state.get("recursive_search", False)
    n_subqueries = len(subqueries)
    fanout_active = n_subqueries > 1 and function in {"hybrid_course", "semantic_general"}
    per_variant_k = max(1, math.ceil(rerank_k * 1.5 / n_subqueries)) if fanout_active else rerank_k

    # Cache check — skip retrieval on exact-match hit
    if config.SESSION_CACHE_ENABLED:
        cache_key = make_cache_key("retrieval", course_ids, rewritten_query)
        cached_chunks = state.get("retrieval_cache", {}).get(cache_key)
        if cached_chunks is not None:
            node_trace.append("retrieval_cache_hit")
            try:
                from langfuse import get_client

                get_client().update_current_span(metadata={"cache_hit": True})
            except Exception:
                pass
            cache_out: dict = {"retrieved_chunks": cached_chunks, "node_trace": node_trace}
            if function == "hybrid_course":
                primer = _build_context_primer(course_ids)
                if primer is not None:
                    cache_out["context_primer"] = primer
            return cache_out

    try:
        if function == "hybrid_course":
            all_errors: list[str] = []
            per_variant_results: list[list[dict]] = []
            per_variant_chunk_counts: list[int] = []

            for sq in subqueries:
                try:
                    reranked_variant, variant_errors = _hybrid_course_for_query(
                        hybrid_search,
                        voyage_rerank,
                        sq,
                        course_ids,
                        retrieve_k,
                        per_variant_k,
                        apply_knee,
                    )
                    per_variant_results.append(reranked_variant)
                    per_variant_chunk_counts.append(len(reranked_variant))
                    all_errors.extend(variant_errors)
                except Exception as exc:  # noqa: BLE001
                    msg = f"subquery retrieval failed: {sq!r}: {exc}"
                    logging.warning("retrieval_node: %s", msg)
                    all_errors.append(msg)
                    per_variant_chunk_counts.append(0)

            if not any(per_variant_results):
                if fanout_active and not per_variant_results:
                    raise RuntimeError("all fanout variants failed")
                reranked: list[dict] = []
            elif fanout_active:
                reranked = rrf_fuse(
                    [r for r in per_variant_results if r],
                    k=60,
                    final_k=rerank_k,
                    id_key=_chunk_rrf_key,
                )
            else:
                # Single-query path — no fusion, but still dedup so a course
                # with several near-duplicate chunks doesn't dominate the output.
                reranked = deduplicate_chunks(per_variant_results[0])

            retrieval_cache_update = {}
            if config.SESSION_CACHE_ENABLED:
                cache_key = make_cache_key("retrieval", course_ids, rewritten_query)
                existing_cache = state.get("retrieval_cache", {})
                retrieval_cache_update = {**existing_cache, cache_key: reranked}

            result: dict = {
                "retrieved_chunks": reranked,
                "retrieval_cache": retrieval_cache_update,
                "node_trace": node_trace,
                "subqueries_chunk_counts": per_variant_chunk_counts,
            }
            primer = _build_context_primer(course_ids)
            if primer is not None:
                result["context_primer"] = primer
            if all_errors:
                result["retrieval_partial_errors"] = all_errors
            return result

        elif function == "semantic_general":
            per_variant_results = []
            per_variant_chunk_counts = []
            errors: list[str] = []
            for sq in subqueries:
                try:
                    chunks = semantic_search(sq, retrieve_k)
                    reranked_variant = voyage_rerank(
                        sq,
                        chunks,
                        top_k=per_variant_k,
                        apply_knee=apply_knee,
                    )
                    per_variant_results.append(reranked_variant)
                    per_variant_chunk_counts.append(len(reranked_variant))
                except Exception as exc:  # noqa: BLE001
                    msg = f"semantic_search failed for {sq!r}: {exc}"
                    logging.warning("retrieval_node: %s", msg)
                    errors.append(msg)
                    per_variant_chunk_counts.append(0)

            if not any(per_variant_results):
                if fanout_active and not per_variant_results:
                    raise RuntimeError("all fanout variants failed")
                reranked = []
            elif fanout_active:
                reranked = rrf_fuse(
                    [r for r in per_variant_results if r],
                    k=60,
                    final_k=rerank_k,
                    id_key=_chunk_rrf_key,
                )
                reranked = deduplicate_chunks(reranked)
                reranked = _cap_per_course(reranked, MAX_CHUNKS_PER_COURSE)
            else:
                reranked = deduplicate_chunks(per_variant_results[0])
                reranked = _cap_per_course(reranked, MAX_CHUNKS_PER_COURSE)

            retrieval_cache_update = {}
            if config.SESSION_CACHE_ENABLED:
                cache_key = make_cache_key("retrieval", course_ids, rewritten_query)
                existing_cache = state.get("retrieval_cache", {})
                retrieval_cache_update = {**existing_cache, cache_key: reranked}

            out: dict = {
                "retrieved_chunks": reranked,
                "retrieval_cache": retrieval_cache_update,
                "node_trace": node_trace,
                "subqueries_chunk_counts": per_variant_chunk_counts,
            }
            if errors:
                out["retrieval_partial_errors"] = errors
            return out

        else:
            return {"retrieved_chunks": [], "node_trace": node_trace}

    except Exception as e:
        return {
            "retrieved_chunks": [],
            "error": f"Retrieval failed: {e}",
            "node_trace": node_trace,
        }
