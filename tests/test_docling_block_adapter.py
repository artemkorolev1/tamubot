"""Tests for tamubot.ingestion.converters.docling_block_adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tamubot.ingestion.converters import docling_block_adapter as dba


def _fake_text_item(text: str, page: int = 1, level: int = 0):
    item = MagicMock(spec=["text", "prov"])
    item.text = text
    prov = MagicMock()
    prov.page_no = page
    prov.bbox = MagicMock(l=0.0, t=0.0, r=100.0, b=20.0)
    item.prov = [prov]
    return item


def _fake_doc(items):
    doc = MagicMock()
    doc.iterate_items = MagicMock(return_value=[(it, 0) for it in items])
    return doc


def _fake_convert_result(items):
    res = MagicMock()
    res.document = _fake_doc(items)
    return res


class TestDoclingBlockAdapter:
    def test_emits_text_blocks(self, tmp_path):
        from docling_core.types.doc.document import TextItem

        text_item = _fake_text_item("hello world")
        text_item.__class__ = TextItem

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([text_item]))

        with patch.object(dba, "convert"):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "hello world"
        assert blocks[0]["page_idx"] == 1
        assert isinstance(blocks[0]["block_id"], str)
        assert len(blocks[0]["block_id"]) == 16

    def test_heading_levels(self, tmp_path):
        from docling_core.types.doc.document import SectionHeaderItem, TitleItem

        title = _fake_text_item("Doc Title")
        title.__class__ = TitleItem
        h2 = _fake_text_item("Section A")
        h2.level = 2
        h2.__class__ = SectionHeaderItem

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([title, h2]))

        with patch.object(dba, "convert"):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        types = [b["type"] for b in blocks]
        assert types == ["heading", "heading"]
        assert blocks[0]["level"] == 1
        assert blocks[1]["level"] == 2

    def test_image_and_table_blocks(self, tmp_path):
        from docling_core.types.doc.document import PictureItem, TableItem

        pic = _fake_text_item("")
        pic.caption_text = "Figure 1"
        pic.__class__ = PictureItem
        tbl = _fake_text_item("")
        tbl.caption_text = "Grade scheme"
        tbl.__class__ = TableItem

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([pic, tbl]))

        with patch.object(dba, "convert"):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        types = [b["type"] for b in blocks]
        assert types == ["image", "table"]
        assert blocks[0]["image_caption"] == "Figure 1"
        assert blocks[1]["table_caption"] == "Grade scheme"

    def test_block_ids_stable_per_position(self, tmp_path):
        """Same input produces same block_ids across runs (stem+page+bbox+idx hash)."""
        from docling_core.types.doc.document import TextItem

        def make_items():
            a = _fake_text_item("para A")
            a.__class__ = TextItem
            b = _fake_text_item("para B", page=2)
            b.__class__ = TextItem
            return [a, b]

        fake_converter_1 = MagicMock()
        fake_converter_1.convert = MagicMock(return_value=_fake_convert_result(make_items()))
        fake_converter_2 = MagicMock()
        fake_converter_2.convert = MagicMock(return_value=_fake_convert_result(make_items()))

        with patch.object(dba, "convert"):
            blocks_1 = dba.docling_to_blocks(
                pdf_path=tmp_path / "stable.pdf",
                output_dir=tmp_path / "out1",
                converter=fake_converter_1,
            )
            blocks_2 = dba.docling_to_blocks(
                pdf_path=tmp_path / "stable.pdf",
                output_dir=tmp_path / "out2",
                converter=fake_converter_2,
            )

        assert [b["block_id"] for b in blocks_1] == [b["block_id"] for b in blocks_2]

    def test_parser_registered(self):
        from tamubot.vendor.raganything import get_parser

        p = get_parser("docling")
        assert p.__class__.__name__ == "DoclingBlockParser"

    def test_consecutive_duplicate_text_deduped(self, tmp_path):
        """Two TextItems with identical normalized text on the same page should
        collapse to one block."""
        from docling_core.types.doc.document import TextItem

        a = _fake_text_item("approved online distribution applets")
        a.__class__ = TextItem
        b = _fake_text_item("approved online distribution applets")
        b.__class__ = TextItem
        c = _fake_text_item("different sentence")
        c.__class__ = TextItem

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([a, b, c]))

        with patch.object(dba, "convert"):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        texts = [b["text"] for b in blocks if b["type"] == "text"]
        assert texts == [
            "approved online distribution applets",
            "different sentence",
        ], f"expected one of the duplicates dropped, got {texts!r}"

    def test_non_adjacent_duplicates_preserved(self, tmp_path):
        """Same text repeated across sections (with a heading between) must NOT be deduped."""
        from docling_core.types.doc.document import SectionHeaderItem, TextItem

        a = _fake_text_item("see the syllabus")
        a.__class__ = TextItem
        h = _fake_text_item("Notes")
        h.level = 2
        h.__class__ = SectionHeaderItem
        c = _fake_text_item("see the syllabus")
        c.__class__ = TextItem

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([a, h, c]))

        with patch.object(dba, "convert"):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        texts = [b["text"] for b in blocks if b["type"] == "text"]
        assert texts == ["see the syllabus", "see the syllabus"]

    def test_picture_img_path_emitted(self, tmp_path):
        from docling_core.types.doc.document import PictureItem

        pic = _fake_text_item("")
        pic.caption_text = "Logo"
        pic.__class__ = PictureItem

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([pic]))

        with patch.object(dba, "convert"), patch.object(dba, "_render_picture_png", return_value=True):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        img_blocks = [b for b in blocks if b["type"] == "image"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["img_path"].endswith(".png"), (
            f"Expected non-empty .png path, got {img_blocks[0]['img_path']!r}"
        )

    def test_table_body_populated_from_cells(self, tmp_path):
        """TableItem with parsed cells should produce non-empty table_body rows."""
        from docling_core.types.doc.document import TableItem

        tbl = _fake_text_item("")
        tbl.caption_text = "Schedule"
        tbl.__class__ = TableItem

        # Mock Docling's table data shape: rows of cells with .text
        def cell(t):
            return MagicMock(text=t)

        tbl.data = MagicMock()
        tbl.data.grid = [
            [cell("Day"), cell("Time")],
            [cell("MON"), cell("10am")],
            [cell("WED"), cell("11am")],
        ]

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock(return_value=_fake_convert_result([tbl]))

        with patch.object(dba, "convert"), patch.object(dba, "_render_table_png", return_value=False):
            blocks = dba.docling_to_blocks(
                pdf_path=tmp_path / "stub.pdf",
                output_dir=tmp_path / "out",
                converter=fake_converter,
            )

        table_blocks = [b for b in blocks if b["type"] == "table"]
        assert len(table_blocks) == 1
        body = table_blocks[0]["table_body"]
        assert body == [["Day", "Time"], ["MON", "10am"], ["WED", "11am"]]


class TestUrlRecovery:
    """_inject_urls_into_blocks: wrap in place, else append the dropped URL line."""

    def test_wrap_in_place_when_link_text_present(self):
        blocks = [{"type": "text", "text": "See cesg.tamu.edu for details", "page_idx": 1}]
        urls = {1: [("cesg.tamu.edu", "https://cesg.tamu.edu/")]}
        wrapped, appended = dba._inject_urls_into_blocks(blocks, urls, stem="S")
        assert (wrapped, appended) == (1, 0)
        assert blocks[0]["text"] == "See [cesg.tamu.edu](https://cesg.tamu.edu/) for details"
        assert len(blocks) == 1  # nothing appended

    def test_append_orphan_url_when_line_dropped(self):
        """Docling dropped the whole `Webpage: <url>` line -> no host block to wrap;
        the URL must be appended, not discarded (the FID_CONTENT_DROPPED fix)."""
        blocks = [
            {"type": "text", "text": "Instructor: Jane Doe", "page_idx": 2},
            {"type": "text", "text": "Catalog Description", "page_idx": 2},
        ]
        urls = {2: [("https://cesg.tamu.edu/faculty/jiang-hu/", "https://cesg.tamu.edu/faculty/jiang-hu/")]}
        wrapped, appended = dba._inject_urls_into_blocks(blocks, urls, stem="S")
        assert (wrapped, appended) == (0, 1)
        recovered = [b for b in blocks if b.get("recovered_url")]
        assert len(recovered) == 1
        assert recovered[0]["text"] == "https://cesg.tamu.edu/faculty/jiang-hu/"
        assert recovered[0]["page_idx"] == 2
        assert recovered[0]["type"] == "text"

    def test_orphan_url_inserted_in_page_order(self):
        """Recovered line lands after the last block of its page, before later pages."""
        blocks = [
            {"type": "text", "text": "p1 body", "page_idx": 1},
            {"type": "text", "text": "p2 body", "page_idx": 2},
        ]
        urls = {1: [("http://x.tamu.edu", "http://x.tamu.edu")]}
        dba._inject_urls_into_blocks(blocks, urls, stem="S")
        texts = [b["text"] for b in blocks]
        # recovered p1 URL sits between p1 body and p2 body, not at the end
        assert texts == ["p1 body", "http://x.tamu.edu", "p2 body"]

    def test_no_duplicate_when_url_already_present(self):
        """If Docling kept the URL somewhere, don't append a second copy."""
        blocks = [{"type": "text", "text": "visit https://canvas.tamu.edu/ daily", "page_idx": 1}]
        urls = {1: [("https://canvas.tamu.edu/", "https://canvas.tamu.edu/")]}
        wrapped, appended = dba._inject_urls_into_blocks(blocks, urls, stem="S")
        # wrapped in place; never appended a duplicate
        assert appended == 0
        assert sum(1 for b in blocks if b.get("recovered_url")) == 0

    def test_url_link_text_emitted_bare_even_if_uri_malformed(self):
        """When the visible link text is a clean URL but the annotation's uri
        target is garbled (source-PDF typo), emit the clean visible URL bare."""
        blocks = [{"type": "text", "text": "unrelated", "page_idx": 1}]
        urls = {1: [("https://canvas.tamu.edu/", "http://h_ps//canvas.tamu.edu/%20(check")]}
        _, appended = dba._inject_urls_into_blocks(blocks, urls, stem="S")
        assert appended == 1
        assert blocks[-1]["text"] == "https://canvas.tamu.edu/"

    def test_named_link_text_rendered_as_markdown(self):
        blocks = [{"type": "text", "text": "unrelated", "page_idx": 1}]
        urls = {1: [("Course Webpage", "https://example.edu/c")]}
        _, appended = dba._inject_urls_into_blocks(blocks, urls, stem="S")
        assert appended == 1
        assert blocks[-1]["text"] == "[Course Webpage](https://example.edu/c)"

    def test_empty_urls_noop(self):
        blocks = [{"type": "text", "text": "x", "page_idx": 1}]
        assert dba._inject_urls_into_blocks(blocks, {}, stem="S") == (0, 0)
        assert len(blocks) == 1

    def test_geometry_places_url_above_all_blocks_at_page_top(self):
        """ECEN_719 case: instructor URL at the page top must NOT land under the
        `ISBN:` block at the page bottom. Docling t is bottom-origin (H - t = from
        top); link_cy is from top. Link above both blocks -> insert before first."""
        blocks = [
            {"type": "text", "text": "Catalog Description", "page_idx": 2, "block_id": "A"},
            {"type": "text", "text": "ISBN:", "page_idx": 2, "block_id": "B"},
        ]
        tops = {"A": 687.2, "B": 97.8}  # H-687=105 (top), H-98=694 (bottom)
        heights = {2: 792.0}
        urls = {2: [("https://cesg.tamu.edu/x", "https://cesg.tamu.edu/x", 79.5)]}  # very top
        _, appended = dba._inject_urls_into_blocks(blocks, urls, tops, heights, stem="S")
        assert appended == 1
        assert [b["text"] for b in blocks] == [
            "https://cesg.tamu.edu/x",
            "Catalog Description",
            "ISBN:",
        ]

    def test_geometry_inserts_after_block_just_above_link(self):
        """A mid-page link lands right after the block immediately above it."""
        blocks = [
            {"type": "text", "text": "Catalog Description", "page_idx": 2, "block_id": "A"},
            {"type": "text", "text": "ISBN:", "page_idx": 2, "block_id": "B"},
        ]
        tops = {"A": 687.2, "B": 97.8}
        heights = {2: 792.0}
        urls = {2: [("http://mid.edu", "http://mid.edu", 400.0)]}  # between A(105) and B(694)
        dba._inject_urls_into_blocks(blocks, urls, tops, heights, stem="S")
        assert [b["text"] for b in blocks] == [
            "Catalog Description",
            "http://mid.edu",
            "ISBN:",
        ]


class TestLabelValueRecovery:
    """_extend_labels_with_values: re-attach dropped values to surviving labels."""

    def test_recovers_dropped_isbn_value(self):
        blocks = [{"type": "text", "text": "ISBN:", "page_idx": 2, "block_id": "A"}]
        lines = {2: ["SystemC: From the Ground Up", "ISBN: 978-0-387-69957-8", "Optional"]}
        bronze_tokens = {"isbn", "systemc", "optional"}  # value NOT present
        n = dba._extend_labels_with_values(blocks, lines, bronze_tokens)
        assert n == 1
        assert blocks[0]["text"] == "ISBN: 978-0-387-69957-8"
        assert blocks[0]["recovered_value"] is True

    def test_skips_when_value_already_present(self):
        """Docling kept the value (e.g. in a table) -> don't duplicate it."""
        blocks = [{"type": "text", "text": "ISBN:", "page_idx": 2, "block_id": "A"}]
        lines = {2: ["ISBN: 978-0-387-69957-8"]}
        bronze_tokens = {"isbn", "978-0-387-69957-8"}  # value already in bronze
        n = dba._extend_labels_with_values(blocks, lines, bronze_tokens)
        assert n == 0
        assert blocks[0]["text"] == "ISBN:"

    def test_ignores_non_label_blocks(self):
        """Only blocks that are a bare `…:` label are eligible."""
        blocks = [{"type": "text", "text": "This course covers", "page_idx": 1, "block_id": "A"}]
        lines = {1: ["This course covers a lot of extra dropped material here"]}
        n = dba._extend_labels_with_values(blocks, lines, set())
        assert n == 0
        assert blocks[0]["text"] == "This course covers"

    def test_only_matches_same_page(self):
        blocks = [{"type": "text", "text": "Authors:", "page_idx": 1, "block_id": "A"}]
        lines = {2: ["Authors: Black and Donovan"]}  # value is on a different page
        n = dba._extend_labels_with_values(blocks, lines, set())
        assert n == 0

    def test_requires_substantial_tail_token(self):
        """A tail of only short noise (no >=4-char missing token) is not recovered."""
        blocks = [{"type": "text", "text": "Room:", "page_idx": 1, "block_id": "A"}]
        lines = {1: ["Room: 3 A"]}  # tail tokens too short
        n = dba._extend_labels_with_values(blocks, lines, set())
        assert n == 0


class TestTwoColumnRecovery:
    """_repair_orphaned_label_values: re-pair labels with their orphaned values."""

    def test_repairs_orphaned_value(self):
        """ECEN_671: `Location:` at the top, `ETB 1037` orphaned far below after
        intervening sections -> merge in and delete the orphan."""
        blocks = [
            {"type": "text", "text": "Location:", "page_idx": 1},
            {"type": "heading", "text": "Course Description", "page_idx": 1},
            {"type": "text", "text": "Long prose about the course.", "page_idx": 1},
            {"type": "text", "text": "ETB 1037", "page_idx": 1},
        ]
        pairs = {1: {"Location:": "ETB 1037"}}
        n = dba._repair_orphaned_label_values(blocks, pairs)
        assert n == 1
        assert blocks[0]["text"] == "Location: ETB 1037"
        assert blocks[0]["recovered_two_column"] is True
        assert all(b.get("text") != "ETB 1037" for b in blocks)  # orphan removed

    def test_leaves_adjacent_value_untouched(self):
        """The correctly-ordered case (value in the next block, e.g. the instructor
        block) must not be merged or duplicated."""
        blocks = [
            {"type": "text", "text": "Instructor:", "page_idx": 1},
            {"type": "text", "text": "Prof. H.R. Harris", "page_idx": 1},
        ]
        pairs = {1: {"Instructor:": "Prof. H.R. Harris"}}
        n = dba._repair_orphaned_label_values(blocks, pairs)
        assert n == 0
        assert [b["text"] for b in blocks] == ["Instructor:", "Prof. H.R. Harris"]

    def test_no_pair_is_noop(self):
        blocks = [{"type": "text", "text": "Section:", "page_idx": 1}]
        assert dba._repair_orphaned_label_values(blocks, {1: {}}) == 0
        assert len(blocks) == 1

    def test_missing_orphan_block_not_attached(self):
        """A paired value with no matching orphan block on the page is skipped
        (don't hallucinate a value into the label)."""
        blocks = [{"type": "text", "text": "Section:", "page_idx": 1}]
        pairs = {1: {"Section:": "501"}}
        n = dba._repair_orphaned_label_values(blocks, pairs)
        assert n == 0
        assert blocks[0]["text"] == "Section:"

    def test_only_same_page(self):
        blocks = [
            {"type": "text", "text": "Location:", "page_idx": 1},
            {"type": "text", "text": "x", "page_idx": 1},
            {"type": "text", "text": "ETB 1037", "page_idx": 2},
        ]
        pairs = {1: {"Location:": "ETB 1037"}}
        n = dba._repair_orphaned_label_values(blocks, pairs)
        assert n == 0


class TestTableCellRecovery:
    """_merge_table_grids: add-only recovery of TableFormer-dropped rows/cells."""

    def test_inserts_whole_dropped_row(self):
        """ECEN_721: TableFormer dropped the `1st MIDTERM | Mar. 3` row between the
        header and the first content row. PyMuPDF kept it -> insert in place."""
        docling = [
            ["Schedule", ""],
            ["V. Transmitter Analysis", "Week 9-14"],
            ["Project Report Due", "Apr. 28"],
        ]
        pdf = [
            ["Schedule", ""],
            ["1st MIDTERM", "Mar. 3"],
            ["V. Transmitter Analysis", "Week 9-14"],
            ["Project Report Due", "Apr. 28"],
        ]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert (inserted, upgraded) == (1, 0)
        assert merged[1] == ["1st MIDTERM", "Mar. 3"]
        assert [r[0] for r in merged] == ["Schedule", "1st MIDTERM", "V. Transmitter Analysis", "Project Report Due"]

    def test_upgrades_cell_with_dropped_trailing_line(self):
        """ISEN_625: a multi-line cell lost its last line (`…plots,` ->
        `…plots, interpreting results`). Replace the cell with the fuller text."""
        docling = [
            ["Week", "Topics", "Events"],
            ["3", "Lab3: counting, branching, plots,", "Lab 3 due"],
        ]
        pdf = [
            ["Week", "Topics", "Events"],
            ["3", "Lab3: counting, branching, plots, interpreting results", "Lab 3 due"],
        ]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert (inserted, upgraded) == (0, 1)
        assert merged[1][1] == "Lab3: counting, branching, plots, interpreting results"

    def test_keeps_docling_filled_merged_cells(self):
        """Docling fills row-spanning cells (`Week 1-8`) that PyMuPDF leaves blank;
        an empty PyMuPDF cell must never overwrite a populated Docling cell."""
        docling = [
            ["Topic", "Week"],
            ["1. Optical Devices", "Week 1-8"],
        ]
        pdf = [
            ["Topic", "Week"],
            ["1. Optical Devices", ""],
        ]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert (inserted, upgraded) == (0, 0)
        assert merged == docling

    def test_does_not_munge_header_split_differently(self):
        """Docling merged `Week`+`Topics` into one header cell with an empty col0;
        the differently-split PyMuPDF header must not duplicate `Week` into col0."""
        docling = [["", "Week Topics", "Important Events"], ["1", "intro to modeling", "Lab 1 due"]]
        pdf = [["Week", "Topics", "Important Events"], ["1", "intro to modeling", "Lab 1 due"]]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert (inserted, upgraded) == (0, 0)
        assert merged[0] == ["", "Week Topics", "Important Events"]

    def test_bails_on_unequal_column_width(self):
        """Disagreeing table structure (different widths) -> keep Docling untouched."""
        docling = [["A", "B"], ["1", "2"]]
        pdf = [["A", "B", "C"], ["1", "2", "3"]]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert (inserted, upgraded) == (0, 0)
        assert merged == docling

    def test_keeps_docling_row_pymupdf_missed(self):
        """A row only Docling has must survive — recovery is add-only."""
        docling = [["H1", "H2"], ["keep me", "x"], ["common", "y"]]
        pdf = [["H1", "H2"], ["common", "y"]]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert (inserted, upgraded) == (0, 0)
        assert ["keep me", "x"] in merged

    def test_does_not_insert_row_whose_tokens_all_present(self):
        """A PyMuPDF row whose tokens already live in the grid is not re-inserted
        (no duplication), even if it failed to align positionally."""
        docling = [["H1", "H2"], ["alpha beta", "gamma"]]
        pdf = [["H1", "H2"], ["beta alpha", "gamma"]]
        merged, inserted, upgraded = dba._merge_table_grids(docling, pdf)
        assert inserted == 0

    def test_empty_pdf_is_noop(self):
        docling = [["A", "B"], ["1", "2"]]
        merged, inserted, upgraded = dba._merge_table_grids(docling, [])
        assert (merged, inserted, upgraded) == (docling, 0, 0)


@pytest.mark.slow
class TestDoclingBlockAdapterIntegration:
    """End-to-end on a real PDF. Skipped if no STAT pilot PDF is present."""

    STAT_PDF = Path("/workspace/data/syllabi/STAT/v5/raw/202611_STAT_608_600_12115_HP.pdf")

    def test_real_pdf_emits_text_and_heading(self, tmp_path):
        if not self.STAT_PDF.exists():
            pytest.skip(f"STAT pilot PDF not available at {self.STAT_PDF}")

        blocks = dba.docling_to_blocks(
            pdf_path=self.STAT_PDF,
            output_dir=tmp_path,
        )

        assert len(blocks) > 0, "expected at least one block"
        types = {b["type"] for b in blocks}
        assert "text" in types, f"no text blocks found; got types={types}"
