"""Advisory pipeline entry point.

Convenience wrapper around the advisory orchestrator graph,
analogous to ``rag.graph.pipeline.run_pipeline()``.
"""

from __future__ import annotations

import logging
from typing import Optional

from tamubot.advisory.graph import build_advisory_graph
from tamubot.advisory.state import OrchestratorState

_logger = logging.getLogger("tamubot")

_advisory_graph = None


def _get_advisory_graph():
    global _advisory_graph
    if _advisory_graph is None:
        _advisory_graph = build_advisory_graph()
    return _advisory_graph


def run_advisory_pipeline(
    query: str,
    scp: dict,
    router_result: dict,
    trace=None,
    session_id: str = "",
    history: Optional[list[dict]] = None,
) -> tuple[str, Optional[str]]:
    """Run the advisory orchestrator pipeline.

    Args:
        query: Raw user query.
        scp: Student Context Profile dict with keys:
             scp_program, scp_completed_courses, scp_target_semester, scp_goal.
        router_result: RouterResult fields as dict (from the existing RAG router).
        trace: Langfuse trace (when called from Streamlit, OTEL context is
               already active; when None, this function creates its own trace).
        session_id: Session identifier.
        history: Conversation history.

    Returns:
        (answer, error) — error is None on success.
    """
    initial_state: OrchestratorState = {
        "query": query,
        "session_id": session_id,
        "history": history or [],
        "scp_program": scp.get("scp_program"),
        "scp_completed_courses": scp.get("scp_completed_courses", []),
        "scp_target_semester": scp.get("scp_target_semester"),
        "scp_goal": scp.get("scp_goal"),
        "scp_validated": True,
        "router_result": router_result,
        "rag_answer": "",
        "final_answer": "",
        "error": None,
    }

    _logger.info(
        "advisory pipeline: program=%s, target=%s, courses=%d",
        scp.get("scp_program"),
        scp.get("scp_target_semester"),
        len(scp.get("scp_completed_courses", [])),
    )

    # Create a standalone trace when no caller-provided OTEL context exists.
    # When Streamlit calls us, `trace` is set and OTEL context is already
    # active from create_trace() — skip to avoid double-tracing.
    own_trace = None
    if trace is None:
        from tamubot.rag.observability import create_trace
        from tamubot.rag.observability.config import ObservabilityConfig

        obs = ObservabilityConfig(
            trace_name="tamubot.advisory",
            tags=["advisory"],
            session_id=session_id or None,
        )
        own_trace, _ = create_trace(obs, query)

    try:
        result = _get_advisory_graph().invoke(initial_state)
        answer = result.get("final_answer", "")
        error = result.get("error")
    finally:
        # Finalize only the trace we created (not the caller's)
        if own_trace is not None:
            from tamubot.rag.observability import finalize_trace

            finalize_trace(own_trace, output=answer if not error else f"[error] {error}")

    return answer, error
