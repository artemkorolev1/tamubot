"""Advisory orchestrator graph — deterministic 3-node pipeline.

Nodes execute in fixed order:
    policy_tool → rag_tool → response_passthrough

SCP is already populated in state via the Streamlit form before this
graph is invoked — there is no profile_collector node.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from tamubot.advisory.nodes.policy_tool import policy_tool_node
from tamubot.advisory.nodes.rag_tool import rag_tool_node
from tamubot.advisory.nodes.response_passthrough import response_passthrough_node
from tamubot.advisory.state import OrchestratorState


def build_advisory_graph():
    """Build and compile the advisory orchestrator graph."""
    graph = StateGraph(OrchestratorState)

    graph.add_node("policy_tool", policy_tool_node)
    graph.add_node("rag_tool", rag_tool_node)
    graph.add_node("response_passthrough", response_passthrough_node)

    graph.set_entry_point("policy_tool")
    graph.add_edge("policy_tool", "rag_tool")
    graph.add_edge("rag_tool", "response_passthrough")
    graph.add_edge("response_passthrough", END)

    return graph.compile()
