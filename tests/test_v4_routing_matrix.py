"""Tests for the routing matrix data contract."""

from tamubot.rag.graph.routing_matrix import ROUTING_MATRIX


def test_all_five_functions_present():
    assert "out_of_scope" in ROUTING_MATRIX
    assert "recursive" in ROUTING_MATRIX
    assert "hybrid_course" in ROUTING_MATRIX
    assert "semantic_general" in ROUTING_MATRIX
    assert "course_summary" in ROUTING_MATRIX


def test_out_of_scope_no_retrieval():
    assert ROUTING_MATRIX["out_of_scope"]["requires_retrieval"] is False
    assert ROUTING_MATRIX["out_of_scope"]["retrieval_passes"] == []
    assert ROUTING_MATRIX["out_of_scope"]["generation_mode"] == "canned"


def test_recursive_has_two_passes():
    entry = ROUTING_MATRIX["recursive"]
    assert entry["requires_retrieval"] is True
    assert "recursive_retrieval" in entry["retrieval_passes"]
    assert "retrieval" in entry["retrieval_passes"]


def test_all_entries_have_required_keys():
    required = {"requires_retrieval", "retrieval_passes", "generation_mode"}
    for fn, entry in ROUTING_MATRIX.items():
        assert required <= entry.keys(), f"Missing keys in {fn}: {required - entry.keys()}"
