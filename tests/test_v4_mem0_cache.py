"""Tests for mem0 integration and session cache (router, retrieval, answer caches)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# cache_utils
# ---------------------------------------------------------------------------


def test_normalize_query_lowercases_and_strips_punctuation():
    from tamubot.rag.utils import normalize_query

    assert normalize_query("CSCE 221??") == "csce 221"


def test_normalize_query_collapses_whitespace():
    from tamubot.rag.utils import normalize_query

    assert normalize_query("  what   are  the  prereqs?  ") == "what are the prereqs"


def test_normalize_query_identical_paraphrases_differ():
    """Two differently-phrased queries should NOT produce the same key (exact-match only)."""
    from tamubot.rag.utils import normalize_query

    assert normalize_query("CSCE 221 grading") != normalize_query("CSCE 221 grading policy")


def test_normalize_query_same_text_same_key():
    from tamubot.rag.utils import normalize_query

    q = "What is the final exam date?"
    assert normalize_query(q) == normalize_query(q)


# ---------------------------------------------------------------------------
# router_node — cache hit
# ---------------------------------------------------------------------------


def _base_router_state(**extra):
    return {"query": "CSCE 221 grading?", "node_trace": [], "timing_ms": {}, **extra}


def _make_router_result(function="hybrid_course"):
    from tamubot.rag.router import RouterResult

    return RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="CSCE 221 grading policy",
        function=function,
        intent_type="ACADEMIC",
    )


def test_router_cache_hit_skips_llm():
    """When router_cache contains the query key, classify_query() is never called."""
    from tamubot.rag.nodes.router_node import router_node
    from tamubot.rag.utils import normalize_query

    cached_entry = {
        "function": "hybrid_course",
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "CSCE 221 grading policy",
        "intent_type": "ACADEMIC",
        "specific_categories": [],
        "recursive_search": False,
        "requires_retrieval": True,
    }
    query = "CSCE 221 grading?"
    state = _base_router_state(router_cache={normalize_query(query): cached_entry})

    with (
        patch("tamubot.core.config.SESSION_CACHE_ENABLED", True),
        patch("tamubot.rag.router.classify_query") as mock_classify,
    ):
        result = router_node(state)

    mock_classify.assert_not_called()
    assert "router_cache_hit" in result["node_trace"]
    assert result["function"] == "hybrid_course"
    assert result["course_ids"] == ["202611_CSCE_221_500"]


def test_router_cache_miss_calls_llm_and_writes_cache():
    """On cache miss, classify_query() is called and the result is written to router_cache."""
    from tamubot.rag.nodes.router_node import router_node
    from tamubot.rag.utils import normalize_query

    rr = _make_router_result()
    state = _base_router_state(router_cache={})

    with (
        patch("tamubot.core.config.SESSION_CACHE_ENABLED", True),
        patch("tamubot.rag.router.classify_query", return_value=rr) as mock_classify,
    ):
        result = router_node(state)

    mock_classify.assert_called_once()
    assert "router_cache_hit" not in result["node_trace"]
    assert "router" in result["node_trace"]

    written_cache = result.get("router_cache", {})
    cache_key = normalize_query("CSCE 221 grading?")
    assert cache_key in written_cache
    assert written_cache[cache_key]["function"] == "hybrid_course"


def test_router_cache_disabled_does_not_use_cache():
    """When SESSION_CACHE_ENABLED=False, no cache check or write happens."""
    from tamubot.rag.nodes.router_node import router_node

    rr = _make_router_result()
    # Pre-populate cache — should be ignored
    state = _base_router_state(router_cache={"csce 221 grading": {"function": "out_of_scope", "course_ids": []}})

    with (
        patch("tamubot.core.config.SESSION_CACHE_ENABLED", False),
        patch("tamubot.rag.router.classify_query", return_value=rr) as mock_classify,
    ):
        result = router_node(state)

    # LLM must have been called because cache was disabled
    mock_classify.assert_called_once()
    assert "router_cache_hit" not in result["node_trace"]
    # No router_cache written to result
    assert not result.get("router_cache")


# ---------------------------------------------------------------------------
# retrieval_node — cache
# ---------------------------------------------------------------------------


def _base_retrieval_state(function="hybrid_course", **extra):
    return {
        "function": function,
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "CSCE 221 grading policy",
        "specific_categories": [],
        "node_trace": [],
        "timing_ms": {},
        **extra,
    }


def _default_chunks():
    return [{"content": "chunk text", "score": 0.9}]


def test_retrieval_cache_hit_skips_retrieval_hybrid():
    """On cache hit for hybrid_course, retriever and reranker are never called."""
    from tamubot.rag.nodes.retrieval_node import _make_retrieval_cache_key, retrieval_node

    fake_chunks = [{"content": "cached chunk", "score": 0.95}]
    cache_key = _make_retrieval_cache_key("hybrid_course", ["202611_CSCE_221_500"], "CSCE 221 grading policy", "")
    state = _base_retrieval_state(retrieval_cache={cache_key: fake_chunks})

    with (
        patch("tamubot.core.config.SESSION_CACHE_ENABLED", True),
        patch("tamubot.rag.tools.mongo.hybrid_search") as mock_hs,
        patch("tamubot.rag.tools.voyage.rerank") as mock_rr,
    ):
        result = retrieval_node(state)

    mock_hs.assert_not_called()
    mock_rr.assert_not_called()
    assert "retrieval_cache_hit" in result["node_trace"]
    assert result["retrieved_chunks"] == fake_chunks


def test_retrieval_cache_hit_skips_retrieval_semantic():
    """On cache hit for semantic_general, retriever and reranker are never called."""
    from tamubot.rag.nodes.retrieval_node import _make_retrieval_cache_key, retrieval_node

    fake_chunks = [{"content": "semantic cached", "score": 0.8}]
    cache_key = _make_retrieval_cache_key("semantic_general", [], "CSCE 221 grading policy", "")
    state = _base_retrieval_state(
        function="semantic_general",
        course_ids=[],
        retrieval_cache={cache_key: fake_chunks},
    )

    with (
        patch("tamubot.core.config.SESSION_CACHE_ENABLED", True),
        patch("tamubot.rag.tools.mongo.semantic_search") as mock_ss,
        patch("tamubot.rag.tools.voyage.rerank") as mock_rr,
    ):
        result = retrieval_node(state)

    mock_ss.assert_not_called()
    mock_rr.assert_not_called()
    assert "retrieval_cache_hit" in result["node_trace"]


def test_retrieval_cache_miss_calls_retrieval_and_writes_cache():
    """On cache miss for hybrid_course, retrieval is executed and written to retrieval_cache."""
    from tamubot.rag.nodes.retrieval_node import _make_retrieval_cache_key, retrieval_node

    chunks = _default_chunks()
    state = _base_retrieval_state(retrieval_cache={})

    with (
        patch("tamubot.core.config.SESSION_CACHE_ENABLED", True),
        patch("tamubot.rag.tools.mongo.hybrid_search", return_value=chunks) as mock_hs,
        patch("tamubot.rag.tools.voyage.rerank", side_effect=lambda q, c, top_k, **kwargs: c),
    ):
        result = retrieval_node(state)

    mock_hs.assert_called_once()
    assert "retrieval_cache_hit" not in result["node_trace"]

    written_cache = result.get("retrieval_cache", {})
    cache_key = _make_retrieval_cache_key("hybrid_course", ["202611_CSCE_221_500"], "CSCE 221 grading policy", "")
    assert cache_key in written_cache


def test_retrieval_cache_key_same_query_same_key():
    """Same function/course_ids/rewritten_query produces identical cache keys."""
    from tamubot.rag.nodes.retrieval_node import _make_retrieval_cache_key

    key1 = _make_retrieval_cache_key("hybrid_course", ["cid1"], "grading policy")
    key2 = _make_retrieval_cache_key("hybrid_course", ["cid1"], "grading policy")
    assert key1 == key2

    # Different rewritten_query → different key
    key3 = _make_retrieval_cache_key("hybrid_course", ["cid1"], "office hours")
    assert key1 != key3


# ---------------------------------------------------------------------------
# history_update_node — answer cache
# ---------------------------------------------------------------------------


def _base_update_state(**extra):
    return {
        "query": "CSCE 221 grading?",
        "answer": "The course uses 40% exams.",
        "history": [],
        "router_result": None,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
        **extra,
    }


def test_history_update_writes_answer_cache():
    """history_update_node should write normalize(query) → answer to answer_cache."""
    from tamubot.rag.nodes.history_update_node import history_update_node
    from tamubot.rag.utils import normalize_query

    state = _base_update_state()

    with patch("tamubot.core.config.SESSION_CACHE_ENABLED", True), patch("tamubot.core.config.MEM0_ENABLED", False):
        result = history_update_node(state)

    cache = result.get("answer_cache", {})
    key = normalize_query("CSCE 221 grading?")
    assert key in cache
    assert cache[key] == "The course uses 40% exams."


def test_history_update_answer_cache_merges_with_existing():
    """answer_cache should merge new entry with any pre-existing entries."""
    from tamubot.rag.nodes.history_update_node import history_update_node
    from tamubot.rag.utils import normalize_query

    existing = {normalize_query("old question"): "old answer"}
    state = _base_update_state(answer_cache=existing)

    with patch("tamubot.core.config.SESSION_CACHE_ENABLED", True), patch("tamubot.core.config.MEM0_ENABLED", False):
        result = history_update_node(state)

    cache = result.get("answer_cache", {})
    assert normalize_query("old question") in cache
    assert normalize_query("CSCE 221 grading?") in cache


def test_history_update_no_answer_cache_when_disabled():
    """When SESSION_CACHE_ENABLED=False, no answer_cache entry is written."""
    from tamubot.rag.nodes.history_update_node import history_update_node

    state = _base_update_state()

    with patch("tamubot.core.config.SESSION_CACHE_ENABLED", False), patch("tamubot.core.config.MEM0_ENABLED", False):
        result = history_update_node(state)

    cache = result.get("answer_cache", {})
    assert len(cache) == 0


def test_history_update_no_answer_cache_when_query_empty():
    """No cache write when query is empty."""
    from tamubot.rag.nodes.history_update_node import history_update_node

    state = _base_update_state(query="", answer="some answer")

    with patch("tamubot.core.config.SESSION_CACHE_ENABLED", True), patch("tamubot.core.config.MEM0_ENABLED", False):
        result = history_update_node(state)

    assert len(result.get("answer_cache", {})) == 0


# ---------------------------------------------------------------------------
# history_inject_node — mem0 context path
# ---------------------------------------------------------------------------


def test_history_inject_skips_mem0_when_no_session_id():
    """When session_id is empty, mem0 path is skipped even if MEM0_ENABLED=True."""
    from tamubot.rag.nodes.history_inject_node import history_inject_node

    mock_manager = MagicMock()

    state = {
        "query": "office hours?",
        "rewritten_query": "office hours?",
        "session_id": "",  # empty → skip mem0
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "Data structures course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    with (
        patch("tamubot.core.config.MEM0_ENABLED", True),
        patch("tamubot.rag.tools.mem0.get_mem0_manager", return_value=mock_manager),
    ):
        history_inject_node(state)

    mock_manager.search_context.assert_not_called()


# ---------------------------------------------------------------------------
# mem0_registry
# ---------------------------------------------------------------------------


def test_mem0_registry_register_and_get():
    import tamubot.rag.tools.mem0 as mem0_registry

    mock_mgr = MagicMock()
    mem0_registry.register("session-abc", mock_mgr)
    assert mem0_registry.get("session-abc") is mock_mgr
    # cleanup
    mem0_registry.unregister("session-abc")


def test_mem0_registry_get_missing_returns_none():
    import tamubot.rag.tools.mem0 as mem0_registry

    assert mem0_registry.get("nonexistent-session") is None


def test_mem0_registry_unregister_removes_entry():
    import tamubot.rag.tools.mem0 as mem0_registry

    mock_mgr = MagicMock()
    mem0_registry.register("to-remove", mock_mgr)
    mem0_registry.unregister("to-remove")
    assert mem0_registry.get("to-remove") is None
