"""Tests for router._validate_and_repair — deterministic post-LLM repairs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tamubot.rag.router import _validate_and_repair

KNOWN_IDS = {"ISEN 625", "ISEN 601", "CSCE 638", "MATH 151"}
TITLE_MAP = {
    "management of engineering systems": "ISEN 601",
    "human factors in healthcare": "ISEN 689",
}


def _is_known(cid: str) -> bool:
    return cid in KNOWN_IDS


def _match_title(text: str):
    haystack = (text or "").lower()
    # Longest-first match
    for title in sorted(TITLE_MAP, key=len, reverse=True):
        if title in haystack:
            return TITLE_MAP[title]
    return None


@pytest.fixture(autouse=True)
def _patch_course_index():
    with (
        patch("tamubot.rag.router.course_index.is_known_course_id", side_effect=_is_known),
        patch("tamubot.rag.router.course_index.match_title", side_effect=_match_title),
    ):
        yield


def _make(**kwargs):
    """Minimal raw LLM dict with sensible defaults."""
    base = {
        "course_ids": [],
        "intent_type": None,
        "recursive_search": False,
        "rewritten_query": "rewritten",
        "section": None,
        "subqueries": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# course_ids grounding
# ---------------------------------------------------------------------------


def test_hallucinated_course_id_dropped():
    cleaned, tele = _validate_and_repair(_make(course_ids=["ISEN 999"]), query="what is ISEN 999")
    assert cleaned["course_ids"] == []
    assert "ISEN 999" in tele["dropped_course_ids"]
    assert "dropped_unreal_course_ids" in tele["repairs_applied"]


def test_real_course_id_kept_even_if_query_paraphrases_title():
    # "Management of Engineering Systems" is the title for ISEN 601 (real).
    cleaned, tele = _validate_and_repair(
        _make(course_ids=["ISEN 601"]),
        query="What's the prereq for the Management of Engineering Systems course?",
    )
    assert cleaned["course_ids"] == ["ISEN 601"]
    assert tele["dropped_course_ids"] == []


def test_title_recovery_when_all_dropped():
    # LLM returned an unreal ID but query mentions a real title — recover.
    cleaned, tele = _validate_and_repair(
        _make(course_ids=["ISEN 999"]),
        query="What's in the Management of Engineering Systems course?",
    )
    assert cleaned["course_ids"] == ["ISEN 601"]
    assert "course_id_recovered_from_title" in tele["repairs_applied"]


def test_title_recovery_when_llm_returns_no_ids():
    cleaned, _ = _validate_and_repair(
        _make(course_ids=[]),
        query="Tell me about Management of Engineering Systems",
    )
    assert cleaned["course_ids"] == ["ISEN 601"]


def test_no_recovery_when_query_has_no_title_match():
    cleaned, tele = _validate_and_repair(
        _make(course_ids=["FAKE 999"]),
        query="completely unrelated question",
    )
    assert cleaned["course_ids"] == []
    assert "course_id_recovered_from_title" not in tele["repairs_applied"]


def test_no_recovery_on_discovery_question_even_with_title_match():
    # Query mentions a real course title as a TOPIC ("which course teaches X").
    # Recovery should NOT fire — the user is asking the system to find a course,
    # not naming one. semantic_general handles this via course_id metadata on chunks.
    cleaned, tele = _validate_and_repair(
        _make(course_ids=[]),
        query="which course teaches Human Factors in Healthcare?",
    )
    assert cleaned["course_ids"] == []
    assert "course_id_recovered_from_title" not in tele["repairs_applied"]
    assert "recovery_skipped_discovery_query" in tele["repairs_applied"]


def test_no_recovery_on_what_course_phrasing():
    cleaned, tele = _validate_and_repair(
        _make(course_ids=[]),
        query="What course covers Management of Engineering Systems topics?",
    )
    assert cleaned["course_ids"] == []
    assert "course_id_recovered_from_title" not in tele["repairs_applied"]


def test_no_recovery_on_which_class_phrasing():
    cleaned, tele = _validate_and_repair(
        _make(course_ids=[]),
        query="which class would teach Human Factors in Healthcare?",
    )
    assert cleaned["course_ids"] == []
    assert "course_id_recovered_from_title" not in tele["repairs_applied"]


def test_course_id_normalization():
    cleaned, _ = _validate_and_repair(_make(course_ids=["isen625"]), query="grading in isen 625")
    assert cleaned["course_ids"] == ["ISEN 625"]


def test_course_ids_string_coerced_to_list():
    cleaned, _ = _validate_and_repair(_make(course_ids="ISEN 625"), query="ISEN 625 schedule")
    assert cleaned["course_ids"] == ["ISEN 625"]


def test_duplicate_course_ids_collapsed():
    cleaned, _ = _validate_and_repair(_make(course_ids=["ISEN 625", "ISEN 625"]), query="ISEN 625")
    assert cleaned["course_ids"] == ["ISEN 625"]


# ---------------------------------------------------------------------------
# recursive_search flag combos
# ---------------------------------------------------------------------------


def test_rs_flipped_when_no_course_ids():
    cleaned, tele = _validate_and_repair(
        _make(course_ids=["FAKE 999"], recursive_search=True),
        query="random query no title",
    )
    assert cleaned["recursive_search"] is False
    assert "rs_flipped_to_false" in tele["repairs_applied"]


def test_rs_kept_when_course_ids_present():
    cleaned, tele = _validate_and_repair(
        _make(course_ids=["ISEN 625"], recursive_search=True),
        query="what comes after ISEN 625",
    )
    assert cleaned["recursive_search"] is True
    assert "rs_flipped_to_false" not in tele["repairs_applied"]


# ---------------------------------------------------------------------------
# subqueries
# ---------------------------------------------------------------------------


def test_subqueries_passthrough():
    cleaned, _ = _validate_and_repair(
        _make(rewritten_query="best rewrite", subqueries=["best rewrite", "alt 1", "alt 2"]),
        query="orig",
    )
    assert cleaned["subqueries"] == ["best rewrite", "alt 1", "alt 2"]
    assert cleaned["rewritten_query"] == "best rewrite"


def test_subqueries_dedupe_by_normalized_form():
    cleaned, tele = _validate_and_repair(
        _make(
            rewritten_query="grading for CSCE 638",
            subqueries=["grading for CSCE 638", "Grading for CSCE 638!", "schedule for CSCE 638"],
        ),
        query="orig",
    )
    assert len(cleaned["subqueries"]) == 2
    assert "subqueries_deduped" in tele["repairs_applied"]


def test_subqueries_capped_at_4():
    cleaned, tele = _validate_and_repair(
        _make(
            rewritten_query="a",
            subqueries=["a", "b", "c", "d", "e", "f"],
        ),
        query="orig",
    )
    assert len(cleaned["subqueries"]) == 4
    assert "subqueries_capped" in tele["repairs_applied"]


def test_subqueries_fallback_to_rewritten():
    cleaned, tele = _validate_and_repair(
        _make(rewritten_query="rewritten only", subqueries=[]),
        query="orig query",
    )
    assert cleaned["subqueries"] == ["rewritten only"]
    # rewritten_query injected → not a fallback (no repair flag)
    assert cleaned["rewritten_query"] == "rewritten only"


def test_subqueries_fallback_to_original_when_all_empty():
    cleaned, tele = _validate_and_repair(
        _make(rewritten_query="", subqueries=[]),
        query="original user query",
    )
    assert cleaned["subqueries"] == ["original user query"]
    assert "subqueries_fallback" in tele["repairs_applied"]


def test_rewritten_query_always_first_subquery():
    cleaned, _ = _validate_and_repair(
        _make(
            rewritten_query="canonical rewrite",
            subqueries=["variant A", "variant B"],  # rewritten missing from list
        ),
        query="orig",
    )
    assert cleaned["subqueries"][0] == "canonical rewrite"
    assert "variant A" in cleaned["subqueries"]


def test_subqueries_string_coerced_to_list():
    cleaned, _ = _validate_and_repair(
        _make(rewritten_query="r", subqueries="only one as string"),
        query="orig",
    )
    assert cleaned["subqueries"] == ["r", "only one as string"]


def test_subqueries_drops_non_strings_and_blanks():
    cleaned, _ = _validate_and_repair(
        _make(rewritten_query="r", subqueries=["r", "", "  ", None, 123, "valid"]),
        query="orig",
    )
    assert cleaned["subqueries"] == ["r", "valid"]


# ---------------------------------------------------------------------------
# intent_type whitelist
# ---------------------------------------------------------------------------


def test_intent_type_valid_kept():
    cleaned, _ = _validate_and_repair(_make(intent_type="ACADEMIC"), query="q")
    assert cleaned["intent_type"] == "ACADEMIC"


def test_intent_type_invalid_nulled():
    cleaned, tele = _validate_and_repair(_make(intent_type="UNICORN"), query="q")
    assert cleaned["intent_type"] is None
    assert "intent_type_rejected" in tele["repairs_applied"]


def test_intent_type_none_passes_silently():
    cleaned, tele = _validate_and_repair(_make(intent_type=None), query="q")
    assert cleaned["intent_type"] is None
    assert "intent_type_rejected" not in tele["repairs_applied"]


def test_no_use_summary_reset_multifacet_when_flag_set():
    cleaned, tele = _validate_and_repair(
        {
            "course_ids": ["ISEN 625"],
            "subqueries": ["a", "b"],
            "rewritten_query": "a",
            "use_summary": True,
            "intent_type": "ACADEMIC",
            "recursive_search": False,
            "section": None,
        },
        query="ISEN 625 a + b",
    )
    assert "use_summary_reset_multifacet" not in tele["repairs_applied"]


def test_use_summary_absent_from_cleaned_output():
    cleaned, _ = _validate_and_repair(_make(), query="orig")
    assert "use_summary" not in cleaned
