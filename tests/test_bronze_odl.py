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
