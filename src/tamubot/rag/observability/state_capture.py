"""Per-node state capture for eval-time introspection.

Given a final pipeline state dict (the return value of ``graph.invoke()``),
``serialize_state()`` returns a node-keyed dict suitable for:
  - writing as JSONL sidecar records (one per node, per question)
  - emitting as rows in per-node candidate golden-set xlsx files
  - setting as Langfuse span input/output

The chunk lists are truncated to ``max_chunks`` and reduced to id + score + a
few identifying fields, because the full chunk payloads can be many KB each
and the captures are for human/diff inspection, not replay.
"""

from __future__ import annotations

from typing import Any

_MAX_CHUNKS = 20


def _summarize_chunk(chunk: dict, idx: int) -> dict:
    return {
        "rank": idx,
        "chunk_id": chunk.get("chunk_id") or chunk.get("_id") or chunk.get("id"),
        "course_id": chunk.get("course_id"),
        "chunk_tag": chunk.get("chunk_tag"),
        "score": chunk.get("score") or chunk.get("rerank_score"),
    }


def serialize_state(state: dict[str, Any], max_chunks: int = _MAX_CHUNKS) -> dict[str, dict]:
    """Return a {node_name: {fields}} dict.

    Keys present in the output depend on which nodes ran. Always-present:
      - router
    Conditionally present (only when their fields are non-empty):
      - retrieval, generator, history
    """
    query = state.get("query", "")
    rewritten = state.get("rewritten_query", "")
    chunks = list(state.get("retrieved_chunks", []) or [])
    anchor_chunks = list(state.get("recursive_chunks", []) or [])
    all_chunks = (anchor_chunks + chunks) if anchor_chunks else chunks

    dump: dict[str, dict] = {}

    dump["router"] = {
        "query": query,
        "function": state.get("function", ""),
        "course_ids": list(state.get("course_ids", []) or []),
        "rewritten_query": rewritten,
        "subqueries": list(state.get("subqueries", []) or []),
        "intent_type": state.get("intent_type"),
        "recursive_search": bool(state.get("recursive_search", False)),
        "retrieval_mode": state.get("retrieval_mode", ""),
        "section": state.get("section"),
        "dropped_course_ids": list(state.get("dropped_course_ids", []) or []),
        "repairs_applied": list(state.get("repairs_applied", []) or []),
    }

    context_primer = state.get("context_primer")
    if all_chunks or context_primer:
        dump["retrieval"] = {
            "rewritten_query": rewritten or query,
            "course_ids": list(state.get("course_ids", []) or []),
            "n_anchor": len(anchor_chunks),
            "n_followup": len(chunks),
            "chunks": [_summarize_chunk(c, i + 1) for i, c in enumerate(all_chunks[:max_chunks])],
            "subqueries_chunk_counts": list(state.get("subqueries_chunk_counts", []) or []),
            # Non-citable primer (course-overview text). Captured separately from chunks
            # so Ragas precision/recall stays computed against retrieved_chunks only.
            "context_primer": context_primer,
            "context_primer_len": len(context_primer) if context_primer else 0,
        }

    answer = state.get("answer", "") or ""
    if answer:
        dump["generator"] = {
            "query": query,
            "n_chunks": len(all_chunks),
            "answer": answer,
        }

    summary = state.get("history_summary", "") or ""
    if summary:
        dump["history"] = {
            "summary": summary,
            "turn_number": state.get("turn_number", 0),
        }

    return dump
