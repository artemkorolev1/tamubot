"""silver_structured pure helpers + the compute function (mocked LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from tamubot.ingestion.pipeline_v6b.assets import silver_structured as mod
from tamubot.rag.models_v4 import (
    AssessmentWeight,
    LetterGradeCutoff,
    MeetingSlot,
    SyllabusExtract,
)


# ── needs_vision / merge_extracts ────────────────────────────────────────────


def test_needs_vision_on_empty_or_tiny():
    assert mod.needs_vision("")
    assert mod.needs_vision("   \n  ")
    assert mod.needs_vision("tiny")
    assert not mod.needs_vision("x" * 300)


def test_merge_extracts_takes_first_populated_per_field():
    a = SyllabusExtract(course_code="STAT 608", meeting_schedule=[])
    b = SyllabusExtract(
        course_code=None,
        meeting_schedule=[MeetingSlot(day="MWF")],
        assessment_weights=[AssessmentWeight(component="Final", weight_pct=50)],
    )
    merged = mod.merge_extracts([a, b])
    assert merged.course_code == "STAT 608"
    assert merged.meeting_schedule[0].day == "MWF"
    assert merged.assessment_weights[0].component == "Final"


def test_merge_extracts_unions_list_fields_across_pages():
    # List fields must accumulate across pages (not truncate to page 1), with
    # order-preserving dedupe of repeats seen on multiple pages.
    p1 = SyllabusExtract(
        course_code="STAT 608",
        learning_outcomes=["LO1", "LO2"],
        assessment_weights=[AssessmentWeight(component="HW", weight_pct=40)],
    )
    p2 = SyllabusExtract(
        learning_outcomes=["LO2", "LO3"],  # LO2 repeats across the page boundary
        assessment_weights=[AssessmentWeight(component="Final", weight_pct=60)],
    )
    merged = mod.merge_extracts([p1, p2])
    assert merged.course_code == "STAT 608"  # scalar: first populated
    assert merged.learning_outcomes == ["LO1", "LO2", "LO3"]  # unioned, deduped
    assert [w.component for w in merged.assessment_weights] == ["HW", "Final"]


# ── parse_syllabus_json ──────────────────────────────────────────────────────


def test_parse_syllabus_json_plain_object():
    raw = '{"course_code": "STAT 600", "credit_hours": 3}'
    ex = mod.parse_syllabus_json(raw)
    assert ex.course_code == "STAT 600"
    assert ex.credit_hours == 3


def test_parse_syllabus_json_strips_json_fence():
    raw = '```json\n{"course_code": "STAT 615"}\n```'
    ex = mod.parse_syllabus_json(raw)
    assert ex.course_code == "STAT 615"


def test_parse_syllabus_json_recovers_from_surrounding_prose():
    raw = 'Here is the JSON: {"course_code": "STAT 612"} hope this helps!'
    ex = mod.parse_syllabus_json(raw)
    assert ex.course_code == "STAT 612"


# ── Tier 1 cleaners ──────────────────────────────────────────────────────────


def test_fix_course_code_prepends_dept_for_bare_number():
    assert mod._fix_course_code("615", "202611_STAT_615_600_30302_HP") == "STAT 615"


def test_fix_course_code_inserts_missing_space():
    assert mod._fix_course_code("STAT692", "202621_STAT_692_701_31935") == "STAT 692"


def test_fix_course_code_leaves_well_formed_codes_alone():
    assert mod._fix_course_code("STAT 608", "202611_STAT_608_600_12115_HP") == "STAT 608"


def test_fix_course_code_handles_none_and_empty():
    assert mod._fix_course_code(None, "any") is None
    assert mod._fix_course_code("", "any") == ""


def test_fix_f_cutoff_normalizes_F_to_zero_when_ladder_present():
    cutoffs = [
        LetterGradeCutoff(grade="A", min_percent=90),
        LetterGradeCutoff(grade="B", min_percent=80),
        LetterGradeCutoff(grade="C", min_percent=70),
        LetterGradeCutoff(grade="D", min_percent=60),
        LetterGradeCutoff(grade="F", min_percent=59),  # wrong — F is the ceiling
    ]
    fixed = mod._fix_f_cutoff(cutoffs)
    f = next(c for c in fixed if c.grade == "F")
    assert f.min_percent == 0


def test_fix_f_cutoff_leaves_zero_alone():
    cutoffs = [
        LetterGradeCutoff(grade="A", min_percent=90),
        LetterGradeCutoff(grade="F", min_percent=0),
    ]
    fixed = mod._fix_f_cutoff(cutoffs)
    f = next(c for c in fixed if c.grade == "F")
    assert f.min_percent == 0


def test_fix_f_cutoff_skips_when_no_letter_ladder():
    # S/U grading — no A/B/C/D context, so any F-like value should be left alone.
    cutoffs = [LetterGradeCutoff(grade="S", min_percent=70), LetterGradeCutoff(grade="U", min_percent=None)]
    fixed = mod._fix_f_cutoff(cutoffs)
    assert fixed[0].min_percent == 70


def test_clean_policy_strips_trailing_json_artifacts():
    cleaned = mod._clean_policy("An Aggie does not lie...case'}\"")
    assert cleaned == "An Aggie does not lie...case"


def test_clean_policy_replaces_apostrophe_newline_glue():
    text = "First sentence ends.'\n'Second sentence starts here."
    cleaned = mod._clean_policy(text)
    # Replace the glue with a paragraph break, no stray apostrophes left.
    assert "'\n'" not in cleaned
    assert "Second sentence starts here." in cleaned


def test_clean_policy_passes_through_clean_text():
    text = "Texas A&M University students are responsible for class attendance."
    assert mod._clean_policy(text) == text


def test_clean_policy_handles_none_and_empty():
    assert mod._clean_policy(None) is None
    # Pure whitespace cleans to None (drop the field rather than carry whitespace).
    assert mod._clean_policy("   ") is None


def test_clean_extract_is_idempotent():
    ex = SyllabusExtract(
        course_code="STAT615",
        letter_grade_cutoffs=[
            LetterGradeCutoff(grade="A", min_percent=90),
            LetterGradeCutoff(grade="F", min_percent=59),
        ],
        attendance_policy="rule.'\n'Boilerplate continues.",
    )
    once = mod.clean_extract(ex, "202611_STAT_615_600_30302_HP")
    snapshot = once.model_dump()
    twice = mod.clean_extract(once, "202611_STAT_615_600_30302_HP")
    assert twice.model_dump() == snapshot
    assert twice.course_code == "STAT 615"
    assert next(c for c in twice.letter_grade_cutoffs if c.grade == "F").min_percent == 0


# ── process_stems (with Gemini mocked at module level) ───────────────────────


def _redirect_paths(monkeypatch, tmp_path):
    """Point the v6b path helpers at a temp tree; return (bronze_dir, out_dir)."""
    from tamubot.ingestion.pipeline_v6b import paths

    bronze = tmp_path / "bronze"
    bronze.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(paths, "bronze_md_path", lambda s: bronze / f"{s}.md")
    monkeypatch.setattr(paths, "silver_structured_path", lambda s: out / f"{s}.json")
    # No raw PDFs in the temp tree → never routes to the vision fallback.
    monkeypatch.setattr(paths, "raw_path", lambda s: tmp_path / "missing" / f"{s}.pdf")
    return bronze, out


def test_process_stems_skips_done_and_collects_failures(tmp_path, monkeypatch):
    bronze, out = _redirect_paths(monkeypatch, tmp_path)
    good = "202611_STAT_624_600_34058"
    bad = "202611_STAT_625_600_00001"
    done = "202611_STAT_608_600_12115_HP"
    (bronze / f"{good}.md").write_text("good content " * 60, encoding="utf-8")
    (bronze / f"{bad}.md").write_text("BOOM content " * 60, encoding="utf-8")
    (bronze / f"{done}.md").write_text("done content " * 60, encoding="utf-8")
    (out / f"{done}.json").write_text("{}", encoding="utf-8")  # already extracted

    def fake_gemini(md: str) -> SyllabusExtract:
        if "BOOM" in md:
            raise RuntimeError("model parse error")
        return SyllabusExtract(course_code="STAT 624")

    monkeypatch.setattr(mod, "extract_text_via_gemini", fake_gemini)

    recorded: list[str] = []
    summary = mod.process_stems(
        [good, bad, done],
        log=lambda *_: None,
        on_done=lambda stem, ex, vis: recorded.append(stem),
    )

    assert summary.ok == 1
    assert summary.errors == 1
    assert summary.failed == [bad]
    assert (out / f"{good}.json").exists()
    assert not (out / f"{bad}.json").exists()
    assert recorded == [good]


def test_process_stems_applies_clean_extract(tmp_path, monkeypatch):
    bronze, out = _redirect_paths(monkeypatch, tmp_path)
    stem = "202611_STAT_615_600_30302_HP"
    (bronze / f"{stem}.md").write_text("real content " * 60, encoding="utf-8")

    # Gemini returns a bare-number course_code AND an F=59 (both Tier-1 fixable).
    def fake_gemini(_md: str) -> SyllabusExtract:
        return SyllabusExtract(
            course_code="615",
            letter_grade_cutoffs=[
                LetterGradeCutoff(grade="A", min_percent=90),
                LetterGradeCutoff(grade="F", min_percent=59),
            ],
        )

    monkeypatch.setattr(mod, "extract_text_via_gemini", fake_gemini)

    mod.process_stems([stem], log=lambda *_: None)

    data = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
    assert data["course_code"] == "STAT 615"  # dept prepended by clean_extract
    f_cutoff = next(c for c in data["letter_grade_cutoffs"] if c["grade"] == "F")
    assert f_cutoff["min_percent"] == 0


def test_compute_writes_structured_json_via_gemini_path(tmp_path, monkeypatch):
    from tamubot.ingestion.pipeline_v6b import paths

    bronze, out = _redirect_paths(monkeypatch, tmp_path)
    stem = "202611_STAT_624_600_34058"
    (bronze / f"{stem}.md").write_text("# Syllabus\n" + "real content " * 50, encoding="utf-8")
    out_path = paths.silver_structured_path(stem)

    fake_extract = SyllabusExtract(
        course_code="STAT 624",
        instructor_name="Toryn Schafer",
        assessment_weights=[AssessmentWeight(component="PP", weight_pct=100)],
    )

    monkeypatch.setattr(mod, "extract_text_via_gemini", lambda _md: fake_extract)

    ctx = MagicMock()
    ctx.partition_keys = [stem]
    # NuExtract resource is required by the asset but should not be touched in
    # the text path — give the test a sentinel that explodes if called.
    sentinel = MagicMock(side_effect=AssertionError("nuextract loaded on text-only run"))
    ctx.resources.nuextract.get_extractor = sentinel

    mod._compute_silver_structured(ctx)

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["course_code"] == "STAT 624"
    assert data["assessment_weights"][0]["component"] == "PP"
    sentinel.assert_not_called()
    ctx.add_asset_metadata.assert_called_once()
    assert ctx.add_asset_metadata.call_args.kwargs["partition_key"] == stem
