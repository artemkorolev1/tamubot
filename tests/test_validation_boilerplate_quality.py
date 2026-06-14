"""Tests for the boilerplate over-hide guard (ECEN_749 class).

Pure — no Dagster/Docling. A boilerplate-flagged chunk carrying a course-specific
signal (grade weight / penalty percentage or concrete due date) is a suspected
over-hide; genuine standard policies (Makeup / Academic Integrity / ADA) must not fire.
"""

from tamubot.ingestion.validation.boilerplate_quality import (
    check_no_boilerplate_overhide,
    check_no_duplicate_overhide,
    find_course_specific_signal,
    find_dedup_protected_signal,
)

# Genuine university-standard policy wording — no course-specific numbers/dates.
_STANDARD_MAKEUP = (
    "## Makeup Work Policy\n\n"
    "Students will be excused from attending class on the day of a graded activity or "
    "an exam for a university-approved absence per Student Rule 7. The student is "
    "responsible for making arrangements with the instructor."
)
_STANDARD_ADA = (
    "## Americans with Disabilities Act\n\n"
    "Texas A&M University is committed to providing equitable access to learning "
    "opportunities for all students. Please contact Disability Resources."
)


def test_standard_makeup_policy_no_signal():
    assert find_course_specific_signal(_STANDARD_MAKEUP) is None


def test_standard_ada_policy_no_signal():
    assert find_course_specific_signal(_STANDARD_ADA) is None


def test_course_specific_penalty_percent_detected():
    body = (
        "## Late Work Policy\n\n"
        "Assignments submitted after the deadline incur a penalty of 20% per day."
    )
    sig = find_course_specific_signal(body)
    assert sig is not None
    assert sig.startswith("percent:")


def test_concrete_due_date_detected():
    sig = find_course_specific_signal("The project proposal is due March 3 by midnight.")
    assert sig is not None
    assert sig.startswith("due_date:")


def test_weekday_time_off_by_default():
    """A support-hours weekday+time (the live false-positive) does NOT fire by default."""
    body = "IT Help Desk hours: Monday - Friday 7:30am to 5:00pm."
    assert find_course_specific_signal(body) is None
    assert find_course_specific_signal(body, include_weekday_time=True) is not None


def test_overhide_flags_customized_late_policy():
    """ECEN_749 class: a header-anchored boilerplate hiding a course-specific late policy."""
    chunks = [
        {
            "is_boilerplate": True,
            "boilerplate_match_source": "header_anchored",
            "header_path": "Late Work Policy",
            "content": "## Late Work Policy\n\nWork submitted late incurs a penalty of 20% per day.",
        },
        {
            "is_boilerplate": True,
            "boilerplate_match_source": "body_jaccard",
            "header_path": "Academic Integrity",
            "content": _STANDARD_ADA,
        },
    ]
    out = check_no_boilerplate_overhide(chunks)
    assert out.passed is False
    assert out.metadata["overhide_count"] == 1
    assert out.metadata["header_anchored_overhide_count"] == 1
    assert "Late Work Policy" in out.metadata["offending_header_paths"]


def test_overhide_clean_for_standard_policies():
    """Genuine STAT_651 standard policies (Makeup/ADA) must NOT trigger."""
    chunks = [
        {
            "is_boilerplate": True,
            "boilerplate_match_source": "header_anchored",
            "header_path": "Makeup Work Policy",
            "content": _STANDARD_MAKEUP,
        },
        {
            "is_boilerplate": True,
            "boilerplate_match_source": "header_anchored",
            "header_path": "Americans with Disabilities Act",
            "content": _STANDARD_ADA,
        },
    ]
    out = check_no_boilerplate_overhide(chunks)
    assert out.passed
    assert out.metadata["overhide_count"] == 0


def test_non_boilerplate_chunk_ignored():
    """A non-boilerplate chunk with a percentage is fine — it wasn't hidden."""
    chunks = [
        {
            "is_boilerplate": False,
            "header_path": "Grading",
            "content": "Homework is 40% of the grade.",
        }
    ]
    out = check_no_boilerplate_overhide(chunks)
    assert out.passed
    assert out.metadata["overhide_count"] == 0


# ---------- dedup over-hide guard (CSCE_629 class) ----------

def test_dedup_protected_signal_detects_url():
    sig = find_dedup_protected_signal("Also, check daily: https://canvas.tamu.edu/ for updates.")
    assert sig is not None
    assert sig.startswith("url:")


def test_dedup_protected_signal_none_for_plain_policy():
    assert find_dedup_protected_signal(_STANDARD_ADA) is None


def test_duplicate_overhide_flags_hidden_url_line():
    """A cross-doc duplicate carrying a course URL should be surfaced as an over-hide."""
    chunks = [
        {
            "is_duplicate": True,
            "duplicate_of_chunk_id": "OTHER_STEM#3",
            "header_path": "Course Website",
            "content": "Check daily: https://canvas.tamu.edu/ is the authoritative source.",
        },
        {
            "is_duplicate": True,
            "duplicate_of_chunk_id": "OTHER_STEM#9",
            "header_path": "Academic Integrity",
            "content": _STANDARD_ADA,  # plain policy → correctly collapsible
        },
    ]
    out = check_no_duplicate_overhide(chunks)
    assert out.passed is False
    assert out.metadata["overhide_count"] == 1
    assert "Course Website" in out.metadata["offending_header_paths"]


def test_duplicate_overhide_clean_for_plain_duplicates():
    chunks = [
        {"is_duplicate": True, "duplicate_of_chunk_id": "X#1", "content": _STANDARD_ADA},
        {"is_duplicate": False, "content": "Visit https://example.com/ — not a duplicate."},
    ]
    out = check_no_duplicate_overhide(chunks)
    assert out.passed
    assert out.metadata["overhide_count"] == 0
