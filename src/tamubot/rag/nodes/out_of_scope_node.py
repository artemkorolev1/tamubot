"""Out-of-scope node — generates a dynamic LLM response for off-topic queries.

Uses a lightweight LLM call to acknowledge the user's specific topic before
politely redirecting them to TAMU academics, rather than returning a generic
canned string.
"""

from __future__ import annotations

from langfuse import get_client as _lf_get_client
from langfuse import observe

from tamubot.core import config
from tamubot.rag.graph.middleware import error_guard_middleware, timing_middleware
from tamubot.rag.prompts import OUT_OF_SCOPE_SYSTEM
from tamubot.rag.state.pipeline_state import PipelineState
from tamubot.rag.tools.llm import stream_llm
from tamubot.rag.utils import OOS_FALLBACK as _OOS_FALLBACK


@observe(as_type="generation", name="node.out_of_scope")
def _generate_oos_response(query: str) -> list[str]:
    """Call the LLM to generate a polite, query-aware out-of-scope reply."""
    messages = [
        {"role": "system", "content": OUT_OF_SCOPE_SYSTEM},
        {"role": "user", "content": query},
    ]
    oos_model = config.OUT_OF_SCOPE_MODEL or (config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL)
    _lf_get_client().update_current_generation(
        model=oos_model,
        input=messages,
    )
    usage_out: list = []
    tokens = list(
        stream_llm(
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            usage_out=usage_out,
            model=config.OUT_OF_SCOPE_MODEL,
        )
    )
    if usage_out:
        _lf_get_client().update_current_generation(
            usage_details={"input": usage_out[0] or 0, "output": usage_out[1] or 0},
        )
    return tokens


@timing_middleware
@error_guard_middleware
def out_of_scope_node(state: PipelineState) -> dict:
    """Generate a dynamic out-of-scope response. Falls back to canned string on error."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("out_of_scope")
    query = state.get("query", "")

    try:
        tokens = _generate_oos_response(query)
        answer = "".join(tokens)
        if not answer.strip():
            raise ValueError("empty LLM response")
    except Exception:
        tokens = [_OOS_FALLBACK]
        answer = _OOS_FALLBACK

    return {
        "answer": answer,
        "answer_stream": tokens,
        "node_trace": node_trace,
    }
