"""parse_extract output-parsing logic (no GPU / model required)."""

from __future__ import annotations

import pytest

from tamubot.ingestion.clients.nuextract_client import parse_extract


def test_parse_clean_json():
    ex = parse_extract('{"course_code": "STAT 608", "credit_hours": 3}')
    assert ex.course_code == "STAT 608"
    assert ex.credit_hours == 3


def test_parse_strips_json_fence():
    ex = parse_extract('```json\n{"course_code": "STAT 615"}\n```')
    assert ex.course_code == "STAT 615"


def test_parse_recovers_from_surrounding_prose():
    ex = parse_extract('Here you go: {"course_code": "CSCE 608"} hope that helps')
    assert ex.course_code == "CSCE 608"


def test_parse_ignores_extra_fields():
    ex = parse_extract('{"course_code": "X", "extra": "junk"}')
    assert ex.course_code == "X"


def test_parse_handles_null_subfields():
    ex = parse_extract('{"assessment_weights": [{"component": "Final", "weight_pct": null}]}')
    assert ex.assessment_weights[0].component == "Final"
    assert ex.assessment_weights[0].weight_pct is None


def test_parse_raises_on_no_json():
    with pytest.raises(ValueError):
        parse_extract("no json object here at all")
