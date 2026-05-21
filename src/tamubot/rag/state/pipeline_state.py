"""Pipeline state TypedDict — single data contract for the RAG graph.

All data flowing between nodes is defined here. Nodes only read/write
these typed fields; no other shared state exists.
"""

from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class ConversationMessage(TypedDict, total=False):
    role: str  # "user" | "assistant"
    content: str
    router_result: Optional[dict]  # lightweight {function, course_ids} for coreference


class PipelineState(TypedDict, total=False):
    # --- Core query ---
    query: str  # raw user input, never overwritten
    rewritten_query: str  # router lookup query; overwritten by recursive_router

    # --- Router fields ---
    function: str  # "hybrid_course"|"recursive"|"semantic_general"|"course_summary"|"out_of_scope"
    course_ids: list[str]
    intent_type: Optional[str]
    recursive_search: bool  # True when recursive path was triggered
    retrieval_mode: str
    requires_retrieval: bool
    section: Optional[str]
    subqueries: list[str]  # 1–4 retrieval rewrites; len>1 triggers RRF fanout in retrieval
    dropped_course_ids: list[str]  # course_ids the validator pruned (not in COURSE_INDEX)
    repairs_applied: list[str]  # validator repair tags (dropped_unreal_course_ids, etc.)

    # --- Retrieval ---
    recursive_chunks: list[dict]  # first-pass anchor chunks (recursive path only)
    retrieved_chunks: list[dict]  # second-pass or standard retrieval chunks
    subqueries_chunk_counts: list[int]  # post-rerank count per subquery variant
    data_gaps: list[tuple[str, str]]
    data_integrity: bool

    # --- Generation ---
    answer: str
    answer_stream: Optional[list]  # list[str] tokens — picklable, checkpointed by LangGraph

    # --- Session / history (merged from former ConversationState) ---
    session_id: str
    history: list[ConversationMessage]
    history_summary: str
    history_context: str
    turn_number: int
    router_cache: dict
    retrieval_cache: dict
    answer_cache: dict
    history_compressed: bool

    # --- Diagnostics ---
    timing_ms: dict[str, float]
    error: Optional[str]
    node_trace: list[str]
    retrieval_partial_errors: list[str]  # per-course failures from parallel hybrid_course path


# Backward-compat alias — history_inject_node and history_update_node use this name
ConversationState = PipelineState
