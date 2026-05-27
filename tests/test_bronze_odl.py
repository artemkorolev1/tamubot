"""Unit tests for tamubot.ingestion.pipeline_v6c.assets.bronze_odl.

Tests cover the pure-Python pieces — letter-drop repair, vocabulary
building, JSON heading walk — without invoking opendataloader-pdf
itself (which needs Java and a real PDF).
"""

from __future__ import annotations

from tamubot.ingestion.pipeline_v5.schemas import HeaderEntry
from tamubot.ingestion.pipeline_v6c.assets.bronze_odl import (
    _build_vocab,
    _collapse_dups,
    _repair_letter_drops,
    _walk_headings,
)


class TestCollapseDups:
    def test_simple(self):
        assert _collapse_dups("College") == "Colege"
        assert _collapse_dups("Meeting") == "Meting"

    def test_empty(self):
        assert _collapse_dups("") == ""

    def test_no_dups(self):
        assert _collapse_dups("abc") == "abc"


class TestRepairLetterDrops:
    def test_word_level_fix(self):
        vocab = _build_vocab("College Syllabus Meeting")
        out, n_word, n_merge = _repair_letter_drops("Colege Sylabus Meting", vocab)
        assert out == "College Syllabus Meeting"
        assert n_word == 3
        assert n_merge == 0

    def test_merge_fix(self):
        vocab = _build_vocab("common TOOLS parallel")
        out, n_word, n_merge = _repair_letter_drops("co mon T OLS paralel", vocab)
        # 'paralel' → 'parallel' is a word-level fix (single-letter drop, no space)
        # 'co mon' → 'common' and 'T OLS' → 'TOOLS' are merge fixes
        assert "common" in out
        assert "TOOLS" in out
        assert "parallel" in out
        assert n_merge >= 2


class TestWalkHeadings:
    def test_extracts_heading_nodes(self):
        raw = {
            "children": [
                {"type": "heading", "content": "Section A", "heading level": 1, "page number": 1},
                {"type": "paragraph", "content": "body text"},
                {"type": "heading", "content": "Section B", "heading level": 2, "page number": 2},
            ],
        }
        out: list[HeaderEntry] = []
        _walk_headings(raw, out)
        assert len(out) == 2
        assert out[0].text == "Section A"
        assert out[0].level == 1
        assert out[1].text == "Section B"
        assert out[1].level == 2

    def test_skips_unlabeled_or_empty(self):
        raw = [
            {"type": "heading", "content": "", "heading level": 1},
            {"type": "heading", "content": "  ", "heading level": 2},
        ]
        out: list[HeaderEntry] = []
        _walk_headings(raw, out)
        assert out == []


from tamubot.ingestion.pipeline_v6c.assets.bronze_odl import _promote_label_headings


class TestPromoteLabelHeadings:
    def test_promotes_orphaned_label_line(self):
        md = "\n".join(
            [
                "Some intro paragraph.",
                "",
                "Prerequisites:",
                "",
                "STAT 601 or equivalent.",
            ]
        )
        out = _promote_label_headings(md)
        lines = out.splitlines()
        assert "## Prerequisites:" in lines, f"expected H2 promotion, got {lines!r}"

    def test_does_not_promote_inline_label(self):
        """'Office: BLOC 417' is an inline metadata line, not a section header."""
        md = "Instructor: Dr. Ghosh\nOffice: BLOC 417\nEmail: foo@bar"
        out = _promote_label_headings(md)
        assert "##" not in out

    def test_does_not_promote_existing_heading(self):
        md = "## Prerequisites:\n\nSTAT 601."
        out = _promote_label_headings(md)
        # Must not double-promote
        assert out.count("## Prerequisites:") == 1
        assert "### Prerequisites:" not in out
        assert "#### Prerequisites:" not in out

    def test_requires_blank_lines_around(self):
        """A 'Label:' line glued to its predecessor isn't a heading."""
        md = "Notes follow.\nPrerequisites:\nSTAT 601."
        out = _promote_label_headings(md)
        assert "## Prerequisites:" not in out

    def test_skips_long_lines(self):
        """A 60+ char line ending in ':' is prose, not a label."""
        long_label = "When you arrive at the building, please find the third floor reception desk and check in:"
        md = f"Intro.\n\n{long_label}\n\nbody."
        out = _promote_label_headings(md)
        assert f"## {long_label}" not in out


class TestRepairMergeBoundaries:
    def test_no_merge_across_colon(self):
        vocab = _build_vocab("Notes Interactive")
        # 'N' is the orphan; 'Interactiveotes:' starts a new label after a colon.
        # The current merge pass would produce 'NInteractiveotes' — that is the bug.
        out, _, _ = _repair_letter_drops("Foo:\nN Interactive notes", vocab)
        assert "NInteractive" not in out, f"merge should not cross newline/colon boundary, got {out!r}"

    def test_no_merge_across_newline(self):
        vocab = _build_vocab("common common common")
        out, _, _ = _repair_letter_drops("co\nmon", vocab)
        assert "common" not in out, "merge should not span newline"


class TestRepairDigitDrops:
    def test_digit_drop_in_section_number(self):
        # Section "600" got collapsed to "60" by veraPDF.
        vocab = _build_vocab("Section 600 Section 600")
        out, n_word, _ = _repair_letter_drops("Section 60 meets here", vocab)
        assert "Section 600" in out, f"expected digit restore, got {out!r}"
        assert n_word == 1

    def test_alphanum_token_repair(self):
        # Duplicate-digit drop: vocab "STAT600" → collapse key "stat60";
        # input "STAT60" has same collapse key. Repair restores the dropped '0'.
        vocab = _build_vocab("STAT600 STAT600")
        out, _, _ = _repair_letter_drops("STAT60", vocab)  # one '0' dropped
        assert "STAT600" in out, f"got {out!r}"


class TestWalkHeadingsUrlFragmentGuard:
    def test_rejects_single_lowercase_word_heading(self):
        raw = {
            "children": [
                {"type": "heading", "content": "statement", "heading level": 1, "page number": 5},
                {"type": "heading", "content": "Course Description", "heading level": 1, "page number": 1},
            ],
        }
        out: list[HeaderEntry] = []
        _walk_headings(raw, out)
        assert [h.text for h in out] == ["Course Description"], (
            f"expected 'statement' rejected as URL fragment, got {[h.text for h in out]!r}"
        )

    def test_accepts_capitalized_short_heading(self):
        raw = {"type": "heading", "content": "Notes", "heading level": 2, "page number": 1}
        out: list[HeaderEntry] = []
        _walk_headings(raw, out)
        assert [h.text for h in out] == ["Notes"]


class TestRepairProtectsAmpersand:
    def test_amp_letter_sequence_not_extended(self):
        # 'A&M' appears in the PDF; an unrelated 'iss' is in the body. The
        # merge pass MUST NOT combine 'A&M' + 'iss' → 'A&Miss'.
        vocab = _build_vocab("Texas A&M Mississippi Mississippi miss")
        out, _, _ = _repair_letter_drops("Texas A&M is committed", vocab)
        assert "A&Miss" not in out, f"merge wrongly extended 'A&M' with 'is', got {out!r}"
