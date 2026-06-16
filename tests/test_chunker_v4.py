"""Tests for tamubot.ingestion.chunker_v4 — covers merge guard and oversized paragraph split."""

from __future__ import annotations

import pytest

from tamubot.ingestion.chunker_v4 import (
    chunk_markdown,
    chunk_semantic,
    chunk_with_log,
)

MAX_TOK = 600
MIN_TOK = 50


# ── Helpers ──────────────────────────────────────────────────────────────────


def _word_block(n_tokens: int) -> str:
    """Generate a block of text with approximately n_tokens tokens."""
    # 1 token ≈ 4 chars; "word " = 5 chars ≈ 1.25 tokens
    words = ["word"] * int(n_tokens * 0.8)
    return " ".join(words)


# ── Merge guard tests ────────────────────────────────────────────────────────


class TestMergeRespectsMaxTokens:
    """_merge_small must never produce a chunk exceeding max_chunk_tokens."""

    def test_many_small_sections_same_header(self):
        """Many tiny sections under one header should not merge into oversized chunk."""
        # Create 20 small sections (~40 tok each) under the same parent
        sections = []
        for i in range(20):
            sections.append(f"## Section {i}\n\n{_word_block(40)}")
        md = "\n\n".join(sections)

        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        for chunk in chunks:
            assert chunk["token_count"] <= MAX_TOK, (
                f"Chunk {chunk['chunk_index']} exceeds max: "
                f"{chunk['token_count']} > {MAX_TOK} (reason={chunk['split_reason']})"
            )

    def test_two_chunks_just_under_limit(self):
        """Two chunks each ~400 tokens should NOT merge (combined > 600)."""
        md = f"## A\n\n{_word_block(400)}\n\n## B\n\n{_word_block(400)}"
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        for chunk in chunks:
            assert chunk["token_count"] <= MAX_TOK

    def test_small_chunks_merge_when_safe(self):
        """Two tiny chunks should still merge when combined <= max_tok."""
        md = "## A\n\nHello.\n\n## B\n\nWorld."
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        # They're both tiny, should merge into 1
        assert len(chunks) == 1
        assert chunks[0]["split_reason"] == "merged"


# ── Oversized paragraph split tests ─────────────────────────────────────────


class TestOversizedParagraphSplit:
    """_paragraph_fallback must split single paragraphs exceeding max_tok."""

    def test_single_huge_paragraph(self):
        """A single 2000-token paragraph must be split into multiple chunks."""
        big_para = _word_block(2000)
        md = f"## Big Section\n\n{big_para}"
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert len(chunks) > 1, "Single huge paragraph should be split"
        for chunk in chunks:
            assert chunk["token_count"] <= MAX_TOK * 1.1, (
                f"Chunk {chunk['chunk_index']} too large: {chunk['token_count']}"
            )

    def test_huge_paragraph_no_headers(self):
        """Huge text with no headers at all should still be split."""
        md = _word_block(2000)
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["token_count"] <= MAX_TOK * 1.1

    def test_long_paragraph_split_logged(self):
        """chunk_with_log should track long_paragraph_split count."""
        md = f"## Section\n\n{_word_block(2000)}"
        chunks, log = chunk_with_log(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert log["long_paragraph_split"] >= 1


# ── Edge case tests ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_header_file(self):
        """Pure table content with no markdown headers should produce chunks."""
        md = "| Col A | Col B |\n|---|---|\n| val1 | val2 |\n| val3 | val4 |"
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert len(chunks) >= 1
        assert chunks[0]["header_path"] == ""

    def test_tiny_file(self):
        """Small file should produce at least one chunk with content preserved."""
        md = "## Title\n\nShort content here.\n\n## Info\n\nMore short text."
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert len(chunks) >= 1
        total_content = " ".join(c["content"] for c in chunks)
        assert "Short content" in total_content
        assert "More short text" in total_content

    def test_empty_input(self):
        """Empty markdown should produce no chunks."""
        chunks = chunk_markdown("", max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert chunks == []

    def test_whitespace_only(self):
        """Whitespace-only input should produce no chunks."""
        chunks = chunk_markdown("   \n\n   ", max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert chunks == []


# ── Regression test with real file ───────────────────────────────────────────


class TestRealFileRegression:
    """Run on actual ISEN files to verify no regressions."""

    @pytest.fixture
    def hierarchy_dir(self):
        from pathlib import Path

        d = Path("data/syllabi/silver/04_hierarchy")
        if not d.exists():
            pytest.skip("hierarchy data not available")
        return d

    def test_normal_file_produces_reasonable_chunks(self, hierarchy_dir):
        """ISEN 613 (normal course) should produce 4-8 chunks, all within limits."""
        path = hierarchy_dir / "202641_ISEN_613_601_34731.md"
        if not path.exists():
            pytest.skip("test file not available")
        md = path.read_text()
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        assert 3 <= len(chunks) <= 10
        for chunk in chunks:
            assert chunk["token_count"] <= MAX_TOK

    def test_previously_oversized_file_now_within_limits(self, hierarchy_dir):
        """ISEN 689-602 had a 1014-tok merged chunk; verify it's fixed."""
        path = hierarchy_dir / "202641_ISEN_689_602_66862.md"
        if not path.exists():
            pytest.skip("test file not available")
        md = path.read_text()
        chunks = chunk_markdown(md, max_chunk_tokens=MAX_TOK, min_chunk_tokens=MIN_TOK)
        for chunk in chunks:
            assert chunk["token_count"] <= MAX_TOK, (
                f"Chunk {chunk['chunk_index']} still oversized: "
                f"{chunk['token_count']} tok (header={chunk['header_path']})"
            )


class TestSemanticCrossSectionMergeGuard:
    """BP_OVERREACH regression: a sub-floor section must NOT backward-merge into
    a previous chunk from a *different* section (gluing course-specific content
    onto an adjacent generic policy and inheriting its header). See ECEN_688."""

    def _ecen688_like(self) -> str:
        # Oversized H1 forces recursion into the H2 children. "Late Work Policy"
        # is generic boilerplate; the sibling "Course Specific Late Work Policy"
        # is tiny (<50 tok) course-specific content that previously merged
        # backward into the generic one.
        return (
            "# College of Engineering\n\n"
            "## Grading Policy\n\n" + ("Grading detail prose with many words. " * 30) + "\n\n"
            "## Late Work Policy\n\n"
            "Work submitted by a student as makeup work for an excused absence is not "
            "considered late work and is exempt from the late work policy.\n\n"
            "## Course Specific Late Work Policy\n\n"
            "Unexcused late work will not be accepted.\n\n"
            "## Course Schedule\n\n" + ("Schedule prose week by week with content. " * 30)
        )

    def test_tiny_course_specific_section_not_glued_onto_generic(self):
        chunks, _ = chunk_semantic(
            self._ecen688_like(), flag_threshold=600, min_chunk_tokens=50
        )
        by_leaf = {c["header_path"].split(" > ")[-1]: c for c in chunks}
        # The course-specific section survives as its OWN chunk with its OWN header
        assert "Course Specific Late Work Policy" in by_leaf, (
            f"course-specific section was absorbed; headers={list(by_leaf)}"
        )
        csl = by_leaf["Course Specific Late Work Policy"]
        assert "Unexcused late work will not be accepted." in csl["content"]
        # …and the generic Late Work Policy chunk does NOT swallow it
        glwp = by_leaf.get("Late Work Policy")
        assert glwp is not None
        assert "Unexcused late work" not in glwp["content"]

    def test_no_content_dropped(self):
        md = self._ecen688_like()
        chunks, _ = chunk_semantic(md, flag_threshold=600, min_chunk_tokens=50)
        joined = "\n".join(c["content"] for c in chunks)
        assert "Unexcused late work will not be accepted." in joined

    def test_same_section_tiny_still_merges(self):
        """Guard must not block within-section merges: a tiny trailing fragment
        in the SAME section still merges (no new fragmentation there)."""
        md = (
            "# Root\n\n"
            "## Only Section\n\n" + ("Body sentence with several words here. " * 40) + "\n\n"
            "Tiny trailing line.\n\n"
            "## Other\n\n" + ("Other body sentence with several words. " * 40)
        )
        chunks, log = chunk_semantic(md, flag_threshold=600, min_chunk_tokens=50)
        # "Tiny trailing line." belongs to "Only Section" and should ride with it
        only = [c for c in chunks if c["header_path"].endswith("Only Section")]
        assert only and "Tiny trailing line." in only[0]["content"]


def test_chunk_semantic_root_body_flagged_no_header():
    """Root body (text before the first header) is headerless, so it must carry
    the NO_HEADER flag the blocking low_no_header_rate check counts (M5)."""
    md = (
        "This is introductory prose that appears before any heading at all. " * 8
        + "\n\n## Real Section\n\n"
        + "Section body content goes here with enough words to be kept. " * 8
    )
    chunks, _ = chunk_semantic(md, flag_threshold=600, min_chunk_tokens=15)
    root = chunks[0]
    assert root["header_path"] == ""
    assert "NO_HEADER" in root["flags"]
