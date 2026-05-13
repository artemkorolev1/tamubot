"""Policy tool node — stub for future policy retrieval.

Always returns ``policy_context = None``.  When policy retrieval is
implemented, this node will query a policy knowledge base and populate
the field with relevant policy text.
"""

from __future__ import annotations

from typing import Any

from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware


@timing_middleware
@error_guard_middleware
def policy_tool_node(state: Any) -> dict:
    return {"policy_context": None}
