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
        trace: Langfuse trace (accepted for API compat, tracing via OTEL context).
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

    result = _get_advisory_graph().invoke(initial_state)

    answer = result.get("final_answer", "")
    error = result.get("error")

    return answer, error
