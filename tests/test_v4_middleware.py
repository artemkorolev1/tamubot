"""Unit tests for v4 middleware decorators."""

import pytest

from tamubot.rag.graph.middleware import V4PipelineError, error_guard_middleware, timing_middleware


def test_error_guard_preserves_node_trace_on_v4_pipeline_error():
    """When a V4PipelineError is raised, node_trace from state must be in the error dict."""

    @error_guard_middleware
    def failing_node(state, **kwargs):
        raise V4PipelineError("boom")

    state = {"node_trace": ["router", "anchor"]}
    result = failing_node(state)

    assert "error" in result
    assert result["node_trace"] == ["router", "anchor"]


def test_error_guard_empty_node_trace_on_v4_pipeline_error():
    """Empty node_trace from state is preserved (not missing key)."""

    @error_guard_middleware
    def failing_node(state, **kwargs):
        raise V4PipelineError("missing")

    result = failing_node({})
    assert result["node_trace"] == []


def test_error_guard_reraises_non_v4_exceptions():
    """Non-V4PipelineError exceptions propagate up (middleware contract)."""

    @error_guard_middleware
    def buggy_node(state, **kwargs):
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        buggy_node({})


def test_timing_middleware_records_elapsed():
    """timing_middleware merges node elapsed time into state timing_ms."""

    @timing_middleware
    def fast_node(state, **kwargs):
        return {"answer": "hi"}

    result = fast_node({"timing_ms": {"router": 5.0}})
    assert "fast_node" in result["timing_ms"]
    assert result["timing_ms"]["router"] == 5.0
    assert isinstance(result["timing_ms"]["fast_node"], float)
