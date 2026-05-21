"""Tests for tamubot.rag.observability.state_capture.serialize_state."""


def test_serialize_state_router_only():
    from tamubot.rag.observability.state_capture import serialize_state

    state = {
        "query": "what is CSCE 670 about?",
        "function": "hybrid_course",
        "course_ids": ["CSCE 670"],
        "rewritten_query": "CSCE 670 course content",
        "intent_type": "ACADEMIC",
        "recursive_search": False,
        "retrieval_mode": "hybrid",
        "section": None,
    }
    dump = serialize_state(state)
    assert "router" in dump
    assert dump["router"]["function"] == "hybrid_course"
    assert dump["router"]["course_ids"] == ["CSCE 670"]
    assert dump["router"]["rewritten_query"] == "CSCE 670 course content"
    # No retrieval or generator captured when their fields are empty
    assert "retrieval" not in dump
    assert "generator" not in dump


def test_serialize_state_with_retrieval_and_generator():
    from tamubot.rag.observability.state_capture import serialize_state

    chunks = [
        {"chunk_id": "abc", "course_id": "CSCE 670", "chunk_tag": "300t_50o", "score": 0.92},
        {"chunk_id": "def", "course_id": "CSCE 670", "score": 0.81},
    ]
    state = {
        "query": "what is CSCE 670 about?",
        "function": "hybrid_course",
        "course_ids": ["CSCE 670"],
        "rewritten_query": "CSCE 670 course content",
        "retrieved_chunks": chunks,
        "recursive_chunks": [],
        "answer": "CSCE 670 is about information retrieval [Source 1].",
    }
    dump = serialize_state(state)
    assert "retrieval" in dump
    assert dump["retrieval"]["n_anchor"] == 0
    assert dump["retrieval"]["n_followup"] == 2
    assert len(dump["retrieval"]["chunks"]) == 2
    assert dump["retrieval"]["chunks"][0]["chunk_id"] == "abc"
    assert dump["retrieval"]["chunks"][0]["score"] == 0.92

    assert "generator" in dump
    assert dump["generator"]["n_chunks"] == 2
    assert "[Source 1]" in dump["generator"]["answer"]


def test_serialize_state_truncates_chunks():
    from tamubot.rag.observability.state_capture import serialize_state

    chunks = [{"chunk_id": f"c{i}", "score": 0.5} for i in range(50)]
    state = {
        "query": "q",
        "function": "semantic_general",
        "course_ids": [],
        "rewritten_query": "q",
        "retrieved_chunks": chunks,
        "recursive_chunks": [],
        "answer": "",
    }
    dump = serialize_state(state, max_chunks=5)
    assert len(dump["retrieval"]["chunks"]) == 5
    assert dump["retrieval"]["n_followup"] == 50  # original count preserved


def test_serialize_state_history_summary():
    from tamubot.rag.observability.state_capture import serialize_state

    state = {
        "query": "q",
        "function": "hybrid_course",
        "course_ids": [],
        "rewritten_query": "q",
        "history_summary": "User asked about CSCE 670 in prior turn.",
        "turn_number": 3,
    }
    dump = serialize_state(state)
    assert "history" in dump
    assert dump["history"]["turn_number"] == 3
    assert "CSCE 670" in dump["history"]["summary"]
