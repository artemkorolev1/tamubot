"""Tests for _apply_hierarchy_safety_net header-level calibration."""

from __future__ import annotations

from tamubot.ingestion.converters.docling_converter import (
    _apply_hierarchy_safety_net,
)


def test_demote_inline_label_h1_to_h3():
    md = "\n".join(
        [
            "# Course Description",
            "Real content here.",
            "# This material Is: Required",  # inline label, NOT a top-level section
            "Some details.",
        ]
    )
    out = _apply_hierarchy_safety_net(
        md,
        original_header_texts_lower={
            "course description",
            "this material is: required",
        },
    )
    lines = out.splitlines()
    # The inline label should be demoted from H1 to H3
    assert lines[0] == "# Course Description"
    assert lines[2] == "### This material Is: Required", f"expected H3 demotion, got {lines[2]!r}"


def test_real_section_headers_unchanged():
    md = "# Course Description\nbody\n# Grading Policy\nmore body"
    out = _apply_hierarchy_safety_net(
        md,
        original_header_texts_lower={"course description", "grading policy"},
    )
    assert out == md
