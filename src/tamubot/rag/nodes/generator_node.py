"""Generator node — streams the final answer."""

from __future__ import annotations

from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware
from tamubot.rag.state.pipeline_state import PipelineState


@timing_middleware
@error_guard_middleware
def generator_node(state: PipelineState) -> dict:
    """Generate the answer. Stores answer_stream as list[str] (picklable for LangGraph)."""
    from tamubot.rag.generator import generate_stream

    node_trace = list(state.get("node_trace", []))
    node_trace.append("generator")

    function = state.get("function", "semantic_general")
    if state.get("recursive_search"):
        function = "recursive"

    anchor_chunks = state.get("recursive_chunks") or []
    anchor_ids = {c.get("course_id") for c in anchor_chunks}
    follow_up_chunks = [c for c in (state.get("retrieved_chunks") or []) if c.get("course_id") not in anchor_ids]
    results = anchor_chunks + follow_up_chunks

    try:
        tokens = list(
            generate_stream(
                results=results,
                question=state.get("query", ""),
                function=function,
                course_ids=state.get("course_ids", []),
                intent_type=state.get("intent_type"),
                data_gaps=state.get("data_gaps", []),
                data_integrity=state.get("data_integrity", True),
                conflicted_course_ids=[],
                history_context=state.get("history_context"),
                context_primer=state.get("context_primer"),
            )
        )
        return {
            "answer": "".join(tokens),
            "answer_stream": tokens,
            "node_trace": node_trace,
        }
    except Exception as e:
        err_msg = f"Generation failed: {e}"
        return {
            "answer": err_msg,
            "answer_stream": [err_msg],
            "error": err_msg,
            "node_trace": node_trace,
        }
