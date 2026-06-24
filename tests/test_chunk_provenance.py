"""Unit tests for the chunk→block alignment bridge (pure, no PDF needed)."""

from tamubot.ingestion.pipeline_v6b.chunk_provenance import (
    align_chunks_to_blocks,
    block_match_key,
    normalize,
)


def _heading(text, page=1, bid="h"):
    return {"type": "heading", "text": text, "page_idx": page, "block_id": bid}


def _text(text, page=1, bid="t"):
    return {"type": "text", "text": text, "page_idx": page, "block_id": bid}


def _chunk(idx, content, hp=""):
    return {"chunk_index": idx, "content": content, "header_path": hp}


def test_normalize_collapses_nbsp_and_whitespace():
    assert normalize("Section\xa0600   Database\n\nSystems") == "section 600 database systems"


def test_block_match_key_strips_markdown_noise_from_headers():
    assert block_match_key(_heading("Grading Policy")) == "grading policy"


def test_block_match_key_table_uses_caption_then_first_cell():
    assert block_match_key({"type": "table", "table_caption": "Grade Scale"}) == "grade scale"
    assert (
        block_match_key({"type": "table", "table_caption": "", "table_body": [["", "Weight"], ["Exam", "40"]]})
        == "weight"
    )


def test_alignment_maps_consecutive_blocks_into_one_chunk():
    blocks = [
        _heading("Grading Policy", bid="a"),
        _text("Your course grade is based on exams.", bid="b"),
        _text("Late work loses 10 percent per day.", bid="c"),
    ]
    chunks = [
        _chunk(
            0,
            "## Grading Policy\n\nYour course grade is based on exams.\n\nLate work loses 10 percent per day.",
        )
    ]
    aligned = align_chunks_to_blocks(blocks, chunks)
    assert [b["block_id"] for b in aligned[0]] == ["a", "b", "c"]


def test_alignment_splits_blocks_across_two_chunks_in_order():
    blocks = [
        _heading("Outcomes", bid="a"),
        _text("Understand databases.", bid="b"),
        _heading("Schedule", bid="c"),
        _text("Week one covers intros.", bid="d"),
    ]
    chunks = [
        _chunk(0, "## Outcomes\n\nUnderstand databases."),
        _chunk(1, "## Schedule\n\nWeek one covers intros."),
    ]
    aligned = align_chunks_to_blocks(blocks, chunks)
    assert [b["block_id"] for b in aligned[0]] == ["a", "b"]
    assert [b["block_id"] for b in aligned[1]] == ["c", "d"]


def test_alignment_skips_blocks_present_in_no_chunk():
    # A deduped table cell whose text never made it into any chunk must not
    # attach to an unrelated chunk.
    blocks = [
        _text("Real paragraph content here.", bid="keep"),
        _text("orphan cell xyzzy", bid="drop"),
    ]
    chunks = [_chunk(0, "Real paragraph content here.")]
    aligned = align_chunks_to_blocks(blocks, chunks)
    assert [b["block_id"] for b in aligned[0]] == ["keep"]


def test_alignment_cursor_is_monotonic_no_backward_match():
    # The word "policy" appears in both chunks; a later block must not bind to an
    # earlier chunk once the cursor has advanced.
    blocks = [
        _heading("Attendance Policy", page=1, bid="a"),
        _heading("Integrity Policy", page=2, bid="b"),
    ]
    chunks = [
        _chunk(0, "## Attendance Policy\n\nShow up."),
        _chunk(1, "## Integrity Policy\n\nNo cheating."),
    ]
    aligned = align_chunks_to_blocks(blocks, chunks)
    assert [b["block_id"] for b in aligned[0]] == ["a"]
    assert [b["block_id"] for b in aligned[1]] == ["b"]


def test_alignment_returns_list_parallel_to_chunks():
    aligned = align_chunks_to_blocks([], [_chunk(0, "x"), _chunk(1, "y")])
    assert aligned == [[], []]
