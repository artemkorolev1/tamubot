"""Tests for the course_index module — title parsing, lookup, mocked-Mongo load."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tamubot.rag import course_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SAMPLE_DOCS = [
    {
        "course_id": "ISEN 625",
        "course_summary": "ISEN 625 | Simulation Methods and Applications | Fall 2026\nrest...",
    },
    {
        "course_id": "ISEN 601",
        "course_summary": "ISEN 601 | Management of Engineering Systems | Spring 2026",
    },
    {
        "course_id": "ISEN 689",
        "course_summary": "ISEN 689 / ISEN 489 | Human Factors in Healthcare | Fall 2026",
    },
    {
        "course_id": "ISEN 660",
        "course_summary": "ISEN 660 / CHEN 660 / SENG 660 | Risk Engineering | Fall 2026",
    },
    {
        "course_id": "ISEN 999",
        "course_summary": "",  # no summary at all
    },
]


@pytest.fixture(autouse=True)
def _reset_module_cache():
    course_index._reset_for_tests()
    yield
    course_index._reset_for_tests()


def _patched_load():
    return patch.object(course_index, "_load", lambda: course_index._build_from_docs(iter(SAMPLE_DOCS)))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_simple():
    ids, title = course_index._parse_summary_first_line("ISEN 625 | Simulation Methods | Fall 2026")
    assert ids == ["ISEN 625"]
    assert title == "Simulation Methods"


def test_parse_crosslisted_with_slash():
    ids, title = course_index._parse_summary_first_line("ISEN 660 / CHEN 660 / SENG 660 | Risk Engineering | Fall 2026")
    assert ids == ["ISEN 660", "CHEN 660", "SENG 660"]
    assert title == "Risk Engineering"


def test_parse_no_title_pipe():
    ids, title = course_index._parse_summary_first_line("ISEN 625 only")
    assert ids == []
    assert title is None


def test_parse_empty_title():
    ids, title = course_index._parse_summary_first_line("ISEN 625 |  | Fall 2026")
    assert ids == ["ISEN 625"]
    assert title is None


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def test_known_course_id_real():
    with _patched_load():
        assert course_index.is_known_course_id("ISEN 625")
        assert course_index.is_known_course_id("isen 625")  # normalize handles case


def test_known_course_id_unreal():
    with _patched_load():
        assert not course_index.is_known_course_id("ISEN 998")


def test_crosslisted_ids_all_registered():
    with _patched_load():
        assert course_index.is_known_course_id("ISEN 660")
        assert course_index.is_known_course_id("CHEN 660")
        assert course_index.is_known_course_id("SENG 660")


def test_doc_with_empty_summary_still_registers_id():
    with _patched_load():
        # ISEN 999 has no summary — title is "" but the id is known.
        assert course_index.is_known_course_id("ISEN 999")


# ---------------------------------------------------------------------------
# match_title
# ---------------------------------------------------------------------------


def test_match_title_exact():
    with _patched_load():
        assert course_index.match_title("Management of Engineering Systems") == "ISEN 601"


def test_match_title_case_insensitive():
    with _patched_load():
        assert course_index.match_title("the management of engineering systems course") == "ISEN 601"


def test_match_title_longest_overlap_wins():
    # Add a course whose title is a prefix of another to verify longest-match.
    docs = list(SAMPLE_DOCS) + [
        {"course_id": "ISEN 700", "course_summary": "ISEN 700 | Engineering | Fall 2026"},
        {"course_id": "ISEN 701", "course_summary": "ISEN 701 | Engineering Systems | Fall 2026"},
    ]
    with patch.object(course_index, "_load", lambda: course_index._build_from_docs(iter(docs))):
        # "engineering systems" should beat the shorter "engineering" hit.
        assert course_index.match_title("studying engineering systems") == "ISEN 701"


def test_match_title_none():
    with _patched_load():
        assert course_index.match_title("zzz no relevant title here") is None


def test_match_title_empty():
    with _patched_load():
        assert course_index.match_title("") is None


# ---------------------------------------------------------------------------
# Lazy single-load
# ---------------------------------------------------------------------------


def test_lazy_load_called_once():
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return course_index._build_from_docs(iter(SAMPLE_DOCS))

    with patch.object(course_index, "_load", fake_load):
        course_index.get_index()
        course_index.get_index()
        course_index.is_known_course_id("ISEN 625")
        course_index.match_title("Risk Engineering")
    assert calls["n"] == 1
