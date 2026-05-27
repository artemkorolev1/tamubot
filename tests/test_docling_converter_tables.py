"""Integration test: Docling extracts table cell content when enabled."""

from __future__ import annotations

from pathlib import Path

import pytest

from tamubot.ingestion.converters.docling_converter import convert, create_converter

FIXTURE = Path("data/syllabi/STAT/v5/raw/202611_STAT_615_600_30302_HP.pdf")


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF missing")
def test_table_structure_extracted(tmp_path):
    converter = create_converter()
    result = convert(FIXTURE, tmp_path, converter=converter)
    md = result.markdown
    # The 3-page PDF has tables (schedule, topics, policies). When
    # do_table_structure=True, these survive in the markdown as
    # pipe-delimited rows. Assert on a cell that is consistently present.
    assert "| Homework" in md or "|Homework" in md, (
        "Expected policy table to contain Homework cell after table-structure extraction"
    )
    assert md.count("|") >= 20, "Expected multiple table rows (≥20 pipe chars total)"
