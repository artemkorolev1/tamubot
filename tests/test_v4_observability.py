"""Tests for graph middleware and pipeline observability wiring."""

from unittest.mock import MagicMock

from tamubot.rag.graph.middleware import V4PipelineError, error_guard_middleware, timing_middleware

# ── timing_middleware tests ──────────────────────────────────────────────────


def test_timing_middleware_populates_timing_ms():
    """timing_middleware must add node_name key to state['timing_ms']."""

    @timing_middleware
    def mock_node(state, registry=None):
        return {"answer": "hello"}

    result = mock_node({"timing_ms": {}}, registry=None)
    assert "mock_node" in result["timing_ms"]
    assert isinstance(result["timing_ms"]["mock_node"], float)
    assert result["timing_ms"]["mock_node"] >= 0.0


def test_timing_middleware_merges_existing_timing():
    """timing_middleware must preserve existing timing_ms entries."""

    @timing_middleware
    def another_node(state, registry=None):
        return {}

    result = another_node({"timing_ms": {"prev_node": 42.0}}, registry=None)
    assert "prev_node" in result["timing_ms"]
    assert "another_node" in result["timing_ms"]


# ── error_guard_middleware tests ─────────────────────────────────────────────


def test_error_guard_catches_v4_pipeline_error():
    """error_guard_middleware catches V4PipelineError and writes state['error']."""

    @error_guard_middleware
    def failing_node(state, registry=None):
        raise V4PipelineError("retrieval timeout")

    result = failing_node({}, registry=None)
    assert "error" in result
    assert "retrieval timeout" in result["error"]


def test_error_guard_does_not_catch_non_v4_errors():
    """error_guard_middleware re-raises non-V4PipelineError exceptions."""

    @error_guard_middleware
    def buggy_node(state, registry=None):
        raise ValueError("unexpected bug")

    try:
        buggy_node({}, registry=None)
        assert False, "Should have raised"
    except ValueError:
        pass  # expected


def test_error_guard_passes_through_successful_result():
    """error_guard_middleware returns result unchanged on success."""

    @error_guard_middleware
    def good_node(state, registry=None):
        return {"answer": "ok", "node_trace": ["good_node"]}

    result = good_node({})
    assert result["answer"] == "ok"


# ── Pipeline tracing wiring ────────────────────────────────────────────────


def test_pipeline_does_not_inject_callbacks():
    """run_pipeline_with_memory no longer injects a CallbackHandler.

    Tracing is handled by the OTEL context set in create_trace(), so the
    pipeline should pass config through without adding LangChain callbacks.
    """
    mock_trace = MagicMock()
    mock_trace.trace_id = "trace-id-123"

    mock_graph_result = {
        "retrieved_chunks": [],
        "function": "out_of_scope",
        "course_ids": [],
        "intent_type": None,
        "recursive_search": False,
        "rewritten_query": "",
        "retrieval_mode": "",
        "data_gaps": [],
        "data_integrity": True,
        "answer_stream": [],
        "answer": "",
    }

    import tamubot.rag.graph.pipeline as pipeline_mod
    from tamubot.rag.graph.pipeline import run_pipeline_with_memory

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = mock_graph_result
    original = pipeline_mod._memory_graph
    pipeline_mod._memory_graph = mock_graph

    try:
        run_pipeline_with_memory(
            "hello",
            trace=mock_trace,
            thread_config={"configurable": {"thread_id": "t-test"}},
        )
    finally:
        pipeline_mod._memory_graph = original

    # graph.invoke was called — no callbacks should be injected
    call_kwargs = mock_graph.invoke.call_args
    config_arg = call_kwargs[1].get("config", {}) if call_kwargs[1] else {}
    callbacks = config_arg.get("callbacks", [])
    assert callbacks == []


def test_pipeline_no_callback_when_trace_is_none():
    """run_pipeline_with_memory passes no callbacks when trace=None."""
    mock_graph_result = {
        "retrieved_chunks": [],
        "function": "out_of_scope",
        "course_ids": [],
        "intent_type": None,
        "recursive_search": False,
        "rewritten_query": "",
        "retrieval_mode": "",
        "data_gaps": [],
        "data_integrity": True,
        "answer_stream": [],
        "answer": "",
    }

    import tamubot.rag.graph.pipeline as pipeline_mod
    from tamubot.rag.graph.pipeline import run_pipeline_with_memory

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = mock_graph_result
    original = pipeline_mod._memory_graph
    pipeline_mod._memory_graph = mock_graph

    try:
        run_pipeline_with_memory(
            "hello",
            trace=None,
            thread_config={"configurable": {"thread_id": "t-no-trace"}},
        )
    finally:
        pipeline_mod._memory_graph = original

    # graph.invoke called — check no "callbacks" key or empty callbacks
    call_kwargs = mock_graph.invoke.call_args
    # Either no config, empty config, or callbacks list is empty
    config_arg = call_kwargs[1].get("config", {}) if call_kwargs[1] else {}
    callbacks = config_arg.get("callbacks", [])
    assert callbacks == []
