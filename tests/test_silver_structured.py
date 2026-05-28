"""silver_structured pure helpers + the compute function (mocked extractor)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from tamubot.ingestion.pipeline_v6b.assets import silver_structured as mod
from tamubot.rag.models_v4 import AssessmentWeight, MeetingSlot, SyllabusExtract


def test_needs_vision_on_empty_or_tiny():
    assert mod.needs_vision("")
    assert mod.needs_vision("   \n  ")
    assert mod.needs_vision("tiny")
    assert not mod.needs_vision("x" * 300)


def test_apply_course_code_fix_prepends_dept_to_bare_number():
    ex = SyllabusExtract(course_code="615")
    mod.apply_course_code_fix(ex, "202611_STAT_615_600_30302_HP")
    assert ex.course_code == "STAT 615"


def test_apply_course_code_fix_leaves_prefixed_code_alone():
    ex = SyllabusExtract(course_code="STAT 608")
    mod.apply_course_code_fix(ex, "202611_STAT_608_600_12115_HP")
    assert ex.course_code == "STAT 608"


def test_merge_extracts_takes_first_populated_per_field():
    a = SyllabusExtract(course_code="STAT 608", meeting_schedule=[])
    b = SyllabusExtract(
        course_code=None,
        meeting_schedule=[MeetingSlot(day="MWF")],
        assessment_weights=[AssessmentWeight(component="Final", weight_pct=50)],
    )
    merged = mod.merge_extracts([a, b])
    assert merged.course_code == "STAT 608"  # from a (first populated)
    assert merged.meeting_schedule[0].day == "MWF"  # from b
    assert merged.assessment_weights[0].component == "Final"  # from b


def test_compute_writes_structured_json_via_text_path(tmp_path, monkeypatch):
    from tamubot.ingestion.pipeline_v6b import paths

    stem = "202611_STAT_624_600_34058"
    md_path = tmp_path / f"{stem}.md"
    md_path.write_text("# Syllabus\n" + "real content " * 50, encoding="utf-8")
    out_path = tmp_path / f"{stem}.json"

    monkeypatch.setattr(paths, "bronze_md_path", lambda s: md_path)
    monkeypatch.setattr(paths, "silver_structured_path", lambda s: out_path)

    fake_extract = SyllabusExtract(
        course_code="STAT 624",
        instructor_name="Toryn Schafer",
        assessment_weights=[AssessmentWeight(component="PP", weight_pct=100)],
    )
    fake_extractor = MagicMock()
    fake_extractor.extract_text.return_value = fake_extract

    ctx = MagicMock()
    ctx.partition_key = stem
    ctx.resources.nuextract.get_extractor.return_value = fake_extractor

    mod._compute_silver_structured(ctx)

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["course_code"] == "STAT 624"
    assert data["assessment_weights"][0]["component"] == "PP"
    fake_extractor.extract_text.assert_called_once()
    fake_extractor.extract_image.assert_not_called()
