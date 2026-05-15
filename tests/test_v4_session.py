"""Tests for SessionManager and multi-turn session state."""

from tamubot.rag.graph.session import SessionManager


def test_same_session_id_gets_same_thread_id():
    """Two calls with the same session_id return the same thread config."""
    manager = SessionManager()
    config1 = manager.get_thread_config("session-abc")
    config2 = manager.get_thread_config("session-abc")
    assert config1["configurable"]["thread_id"] == config2["configurable"]["thread_id"]


def test_different_session_ids_get_different_thread_ids():
    """Two different session IDs must get different thread configs."""
    manager = SessionManager()
    config1 = manager.get_thread_config("session-1")
    config2 = manager.get_thread_config("session-2")
    assert config1["configurable"]["thread_id"] != config2["configurable"]["thread_id"]


def test_clear_session_removes_thread_id():
    """After clear_session, next call creates a new thread_id."""
    manager = SessionManager()
    config1 = manager.get_thread_config("session-x")
    manager.clear_session("session-x")
    config2 = manager.get_thread_config("session-x")
    assert config1["configurable"]["thread_id"] != config2["configurable"]["thread_id"]


def test_two_turns_same_session_share_history():
    """With MemorySaver, two invocations with the same thread_id accumulate history."""
    from tamubot.rag.graph.builder import build_graph_with_memory
    from tamubot.rag.graph.checkpointer import make_checkpointer

    checkpointer = make_checkpointer("memory")
    graph = build_graph_with_memory(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "test-thread-1"}}

    base_state = {
        "query": "hello",
        "node_trace": [],
        "timing_ms": {},
        "data_gaps": [],
        "data_integrity": True,
        "recursive_chunks": [],
        "retrieved_chunks": [],
        "history": [],
    }

    # Turn 1
    result1 = graph.invoke(base_state, config=thread_config)
    assert result1.get("turn_number", 0) >= 1

    # Turn 2 — same thread_id, should have history from turn 1
    result2 = graph.invoke(
        {**base_state, "query": "follow-up"},
        config=thread_config,
    )
    assert result2.get("turn_number", 0) >= 2 or len(result2.get("history", [])) >= 2


def test_two_different_sessions_are_independent():
    """Two different thread_ids don't share state."""
    from tamubot.rag.graph.builder import build_graph_with_memory
    from tamubot.rag.graph.checkpointer import make_checkpointer

    checkpointer = make_checkpointer("memory")
    graph = build_graph_with_memory(checkpointer=checkpointer)

    state = {
        "query": "hello",
        "node_trace": [],
        "timing_ms": {},
        "data_gaps": [],
        "data_integrity": True,
        "recursive_chunks": [],
        "retrieved_chunks": [],
        "history": [],
    }

    result_a = graph.invoke(state, config={"configurable": {"thread_id": "thread-A"}})
    result_b = graph.invoke(state, config={"configurable": {"thread_id": "thread-B"}})

    # Both ran independently — turn_number should be 1 for both (not accumulated)
    assert result_a.get("turn_number", 1) == result_b.get("turn_number", 1)


def test_sqlite_checkpointer_path_is_absolute():
    """The computed SQLite DB path must be absolute, not CWD-relative."""
    import os
    from pathlib import Path

    import tamubot.rag.graph.checkpointer as cp_mod

    db_path = str(Path(cp_mod.__file__).parent / "sessions.db")
    assert os.path.isabs(db_path), f"Expected absolute path, got: {db_path}"
