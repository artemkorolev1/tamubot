"""Schema contract for SyllabusExtract + its NuExtract template."""

from __future__ import annotations

from tamubot.ingestion.clients.nuextract_client import SYLLABUS_TEMPLATE
from tamubot.rag.models_v4 import SyllabusExtract


def test_schema_has_split_grading_fields():
    """Grading must be split into the two distinct concepts; the conflated
    'grading_scale' field must be gone."""
    fields = set(SyllabusExtract.model_fields)
    assert {"assessment_weights", "letter_grade_cutoffs"} <= fields
    assert "grading_scale" not in fields


def test_template_keys_match_model_fields():
    """The NuExtract template and the Pydantic model must stay in sync."""
    assert set(SYLLABUS_TEMPLATE) == set(SyllabusExtract.model_fields)


def test_extract_validates_nested_records():
    ex = SyllabusExtract.model_validate(
        {
            "course_code": "STAT 608",
            "assessment_weights": [{"component": "Final", "weight_pct": 50}],
            "letter_grade_cutoffs": [{"grade": "A", "min_percent": 90}],
            "meeting_schedule": [{"day": "MWF", "time": "10-11", "location": "BLOC 457"}],
        }
    )
    assert ex.assessment_weights[0].weight_pct == 50
    assert ex.letter_grade_cutoffs[0].grade == "A"
    assert ex.meeting_schedule[0].location == "BLOC 457"


def test_extract_ignores_unknown_fields():
    ex = SyllabusExtract.model_validate({"course_code": "X", "bogus_field": 1})
    assert ex.course_code == "X"
    assert not hasattr(ex, "bogus_field")
