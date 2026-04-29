"""Response passthrough node — copies rag_answer to final_answer.

Future: when policy_context is not None, this node will synthesize
the RAG answer with policy information into a combined advisory response.
"""

from __future__ import annotations

from typing import Any

from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware


@timing_middleware
@error_guard_middleware
def response_passthrough_node(state: Any) -> dict:
    policy = state.get("policy_context")
    rag_answer = state.get("rag_answer", "")

    if policy is not None:
        # Future: synthesize RAG answer + policy context
        pass

    return {"final_answer": rag_answer}
