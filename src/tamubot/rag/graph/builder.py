"""Build the RAG LangGraph state machine."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from tamubot.rag.nodes.generator_node import generator_node
from tamubot.rag.nodes.out_of_scope_node import out_of_scope_node
from tamubot.rag.nodes.recursive_retrieval_node import recursive_retrieval_node
from tamubot.rag.nodes.recursive_router_node import recursive_router_node
from tamubot.rag.nodes.retrieval_node import retrieval_node
from tamubot.rag.nodes.router_node import router_node
from tamubot.rag.state.pipeline_state import PipelineState

# ---------------------------------------------------------------------------
# Conditional edge dispatcher (inlined from former edges/routing.py)
# ---------------------------------------------------------------------------


def route_after_router(state: PipelineState) -> str:
    """Dispatch to retrieval path based on function type."""
    function = state.get("function", "out_of_scope")
    if function == "out_of_scope":
        return "out_of_scope"
    elif function == "recursive":
        return "recursive_retrieval"
    else:
        return "retrieval"


# ---------------------------------------------------------------------------
# Routing matrix — reference documentation only (not wired to graph routing)
# ---------------------------------------------------------------------------


class RoutingEntry(TypedDict):
    requires_retrieval: bool
    retrieval_passes: list[str]
    generation_mode: str


ROUTING_MATRIX: dict[str, RoutingEntry] = {
    "out_of_scope": {
        "requires_retrieval": False,
        "retrieval_passes": [],
        "generation_mode": "canned",
    },
    "recursive": {
        "requires_retrieval": True,
        "retrieval_passes": ["recursive_retrieval", "retrieval"],
        "generation_mode": "stream",
    },
    "hybrid_course": {
        "requires_retrieval": True,
        "retrieval_passes": ["hybrid_course"],
        "generation_mode": "stream",
    },
    "semantic_general": {
        "requires_retrieval": True,
        "retrieval_passes": ["semantic"],
        "generation_mode": "stream",
    },
}


def build_graph():
    """Build and compile the RAG pipeline graph (stateless, no conversation memory)."""
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_router", recursive_router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_router")
    graph.add_edge("recursive_router", "retrieval")
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def build_graph_eval():
    """Build eval-only graph: router + retrieval, no generator.

    Identical edge structure to build_graph() but terminates at retrieval.
    Use with run_pipeline_eval() to get the same node-level Langfuse traces
    as production runs without running the generator.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_router", recursive_router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_router")
    graph.add_edge("recursive_router", "retrieval")
    graph.add_edge("retrieval", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def _route_headless(state: PipelineState) -> str:
    """Dispatch for headless graph — no out_of_scope path."""
    function = state.get("function", "semantic_general")
    if function == "recursive":
        return "recursive_retrieval"
    return "retrieval"


def build_graph_headless():
    """Build a router-less RAG graph: caller provides routing fields directly.

    The caller must populate ``function``, ``course_ids``, ``retrieval_mode``,
    ``rewritten_query``, and ``intent_type`` in the initial PipelineState.
    The graph starts at retrieval (or recursive_retrieval for recursive queries).
    """
    graph = StateGraph(PipelineState)

    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_router", recursive_router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generator", generator_node)

    graph.set_conditional_entry_point(
        _route_headless,
        {
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_router")
    graph.add_edge("recursive_router", "retrieval")
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", END)

    return graph.compile()


def build_graph_with_memory(checkpointer=None):
    """Build the RAG pipeline graph with conversation memory support."""
    from tamubot.rag.nodes.history_inject_node import history_inject_node
    from tamubot.rag.nodes.history_update_node import history_update_node

    graph = StateGraph(PipelineState)

    graph.add_node("history_inject", history_inject_node)
    graph.add_node("router", router_node)
    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_router", recursive_router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("history_update", history_update_node)

    graph.set_entry_point("history_inject")
    graph.add_edge("history_inject", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_router")
    graph.add_edge("recursive_router", "retrieval")
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", "history_update")
    graph.add_edge("out_of_scope", "history_update")
    graph.add_edge("history_update", END)

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)
