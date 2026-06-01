from docling.datamodel.pipeline_options import TableFormerMode

from tamubot.ingestion.converters.docling_converter import _build_pipeline_options


def test_table_structure_is_accurate_and_no_cell_matching():
    opts = _build_pipeline_options()
    assert opts.do_table_structure is True
    assert opts.table_structure_options.mode == TableFormerMode.ACCURATE
    assert opts.table_structure_options.do_cell_matching is False
