"""RAG tool node — invokes the headless RAG graph with SCP-enriched input."""

from __future__ import annotations

import logging
from typing import Any

from tamubot.advisory.rag_input_builder import build_rag_input
from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware

_logger = logging.getLogger("tamubot")

_headless_graph = None


def _get_headless_graph():
    global _headless_graph
    if _headless_graph is None:
        from tamubot.rag.graph.builder import build_graph_headless

        _headless_graph = build_graph_headless()
    return _headless_graph


@timing_middleware
@error_guard_middleware
def rag_tool_node(state: Any) -> dict:
    rag_input = build_rag_input(state)
    _logger.info(
        "advisory.rag_tool: function=%s, course_ids=%s, query_len=%d",
        rag_input.get("function"),
        rag_input.get("course_ids"),
        len(rag_input.get("rewritten_query", "")),
    )

    result = _get_headless_graph().invoke(rag_input)
    answer = result.get("answer", "")

    return {"rag_answer": answer}
