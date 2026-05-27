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
        cell = lambda t: MagicMock(text=t)
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
