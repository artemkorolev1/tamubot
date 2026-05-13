"""Router node — classifies query and populates routing fields in state."""

from __future__ import annotations

from typing import Optional

from tamubot.core import config
from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware
from tamubot.rag.state.pipeline_state import PipelineState
from tamubot.rag.utils import normalize_query


def _build_prior_context(history: list) -> Optional[str]:
    """Build a context string from the most recent turn for pronoun resolution.

    Scans backwards to find the most recent user query and its adjacent assistant
    router_result. Stops at the first assistant message seen — never pulls context
    from an earlier, unrelated turn.
    """
    if not history:
        return None

    prior_query = ""
    prior_course_ids: list[str] = []
    seen_assistant = False

    for msg in reversed(history):
        role = msg.get("role", "")
        if not prior_query and role == "user":
            prior_query = msg.get("content", "")[:150]
        if role == "assistant" and not seen_assistant:
            seen_assistant = True
            rr = msg.get("router_result") or {}
            prior_course_ids = rr.get("course_ids", [])
        if seen_assistant and prior_query:
            break  # have both query and the one adjacent assistant turn — stop

    if not prior_query:
        return None

    parts = [f'previous query: "{prior_query}"']
    if prior_course_ids:
        parts.append(f"courses: {', '.join(prior_course_ids)}")
    return ", ".join(parts)


@timing_middleware
@error_guard_middleware
def router_node(state: PipelineState) -> dict:
    """Classify query using classify_query, write all RouterResult fields to state.

    Falls back to out_of_scope on any error.
    """
    from tamubot.rag.router import classify_query

    raw_query = state.get("query", "")
    query = state.get("rewritten_query") or raw_query
    node_trace = list(state.get("node_trace", []))
    node_trace.append("router")

    # Cache check — skip LLM call on exact-match hit (key on raw query)
    if config.SESSION_CACHE_ENABLED:
        cache_key = normalize_query(raw_query)
        cached = state.get("router_cache", {}).get(cache_key)
        if cached:
            node_trace.append("router_cache_hit")
            try:
                from langfuse import get_client

                get_client().update_current_span(metadata={"cache_hit": True})
            except Exception:
                pass
            return {
                "router_result": None,  # RouterResult object not available from cache dict
                "function": cached["function"],
                "course_ids": cached["course_ids"],
                "rewritten_query": cached.get("rewritten_query") or raw_query,
                "intent_type": cached.get("intent_type"),
                "recursive_search": cached.get("recursive_search", False),
                "requires_retrieval": cached.get("requires_retrieval", True),
                "node_trace": node_trace,
            }

    # history_inject_node bakes history_context into rewritten_query; skip prior_context
    # when the query is already augmented, otherwise fall back to compact context.
    prior_context = None if state.get("rewritten_query") else _build_prior_context(state.get("history", []))

    try:
        router_result = classify_query(query, prior_context=prior_context)

        # Cache write — store serialized result for future identical queries
        router_cache_update = {}
        if config.SESSION_CACHE_ENABLED:
            cache_key = normalize_query(raw_query)
            existing_cache = state.get("router_cache", {})
            router_cache_update = {
                **existing_cache,
                cache_key: {
                    "function": router_result.function,
                    "course_ids": router_result.course_ids,
                    "rewritten_query": router_result.rewritten_query or query,
                    "intent_type": router_result.intent_type,
                    "recursive_search": router_result.recursive_search,
                    "requires_retrieval": router_result.requires_retrieval,
                },
            }

        return {
            "router_result": router_result,
            "function": router_result.function,
            "course_ids": router_result.course_ids,
            "rewritten_query": router_result.rewritten_query or query,
            "intent_type": router_result.intent_type,
            "recursive_search": router_result.recursive_search,
            "requires_retrieval": router_result.requires_retrieval,
            "router_cache": router_cache_update,
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "function": "out_of_scope",
            "course_ids": [],
            "rewritten_query": query,
            "requires_retrieval": False,
            "error": f"Router failed: {e}",
            "node_trace": node_trace,
        }
