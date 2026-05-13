"""Translate OrchestratorState → PipelineState dict for the headless RAG graph.

This is the **only** place the two state schemas touch.  Fails loudly
via Pydantic validation when required fields are missing — never silently
passes None into the RAG graph.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator

from tamubot.advisory.state import OrchestratorState


class _RagInputValidator(BaseModel):
    """Validate that the translated PipelineState has all required fields."""

    query: str
    rewritten_query: str
    function: str
    retrieval_mode: str
    course_ids: list[str]
    intent_type: Optional[str] = None

    @field_validator("function")
    @classmethod
    def function_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("function must not be empty")
        return v

    @field_validator("retrieval_mode")
    @classmethod
    def retrieval_mode_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("retrieval_mode must not be empty")
        return v


def _format_scp_block(state: OrchestratorState) -> str:
    """Build a concise [Student Context: ...] string from SCP fields."""
    parts: list[str] = []
    if state.get("scp_program"):
        parts.append(state["scp_program"])
    courses = state.get("scp_completed_courses", [])
    if courses:
        parts.append(f"completed {'/'.join(courses)}")
    if state.get("scp_target_semester"):
        parts.append(f"targeting {state['scp_target_semester']}")
    if state.get("scp_goal"):
        parts.append(f"goal: {state['scp_goal']}")
    return f"[Student Context: {', '.join(parts)}]" if parts else ""


def build_rag_input(state: OrchestratorState) -> dict:
    """Convert OrchestratorState to a dict matching PipelineState structure.

    Raises ``pydantic.ValidationError`` if required fields are missing or empty.
    """
    router = state.get("router_result", {})

    # Enrich the rewritten_query with SCP context
    base_query = router.get("rewritten_query") or state.get("query", "")
    scp_block = _format_scp_block(state)
    enriched_query = f"{base_query} {scp_block}".strip() if scp_block else base_query

    # Build the PipelineState-compatible dict
    rag_input: dict = {
        "query": state.get("query", ""),
        "rewritten_query": enriched_query,
        "function": router.get("function", ""),
        "course_ids": router.get("course_ids", []),
        "intent_type": router.get("intent_type"),
        "recursive_search": router.get("recursive_search", False),
        "retrieval_mode": router.get("retrieval_mode", ""),
        "requires_retrieval": router.get("requires_retrieval", True),
        "section": router.get("section"),
        # Defaults matching _INITIAL_STATE in rag/graph/pipeline.py
        "node_trace": [],
        "timing_ms": {},
        "data_gaps": [],
        "data_integrity": True,
        "recursive_chunks": [],
        "retrieved_chunks": [],
        "answer": "",
        "history_context": "",
        "answer_stream": None,
    }

    # Copy session fields if present
    if state.get("session_id"):
        rag_input["session_id"] = state["session_id"]
    if state.get("history"):
        rag_input["history"] = state["history"]

    # Validate — fails loudly on missing/empty required fields
    _RagInputValidator(
        query=rag_input["query"],
        rewritten_query=rag_input["rewritten_query"],
        function=rag_input["function"],
        retrieval_mode=rag_input["retrieval_mode"],
        course_ids=rag_input["course_ids"],
        intent_type=rag_input["intent_type"],
    )

    return rag_input
