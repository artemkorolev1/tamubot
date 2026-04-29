"""Orchestrator state for the advisory pipeline.

Separate from PipelineState — the two schemas touch only in
``rag_input_builder.build_rag_input()`` which translates between them.
"""

from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class OrchestratorState(TypedDict, total=False):
    # --- Input (from Streamlit session) ---
    query: str
    session_id: str
    history: list[dict]

    # --- Student Context Profile (from Streamlit form) ---
    scp_program: Optional[str]
    scp_completed_courses: list[str]
    scp_target_semester: Optional[str]
    scp_goal: Optional[str]
    scp_validated: bool

    # --- Router result (from existing router, before orchestrator) ---
    router_result: dict  # full RouterResult fields as dict

    # --- Tool outputs ---
    policy_context: Optional[str]
    rag_answer: str
    final_answer: str
    error: Optional[str]
