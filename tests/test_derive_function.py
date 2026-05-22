import inspect

from tamubot.rag.router import _derive_function


def test_derive_function_signature_has_no_use_summary():
    sig = inspect.signature(_derive_function)
    assert "use_summary" not in sig.parameters


def test_derive_function_course_bound_non_recursive_is_hybrid_course():
    assert _derive_function(["ISEN 625"], False, "ACADEMIC") == "hybrid_course"
    assert _derive_function(["ISEN 625"], False, None) == "hybrid_course"


def test_derive_function_empty_course_ids_unchanged():
    assert _derive_function([], False, "ACADEMIC") == "semantic_general"
    assert _derive_function([], False, None) == "out_of_scope"


def test_derive_function_recursive_unchanged():
    assert _derive_function(["ISEN 625"], True, "ACADEMIC") == "recursive"
