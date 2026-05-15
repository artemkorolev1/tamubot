from tamubot.app.citations import rewrite_citations


def test_rewrites_with_page_to_pdf_link():
    answer = "Late work loses 10% per day [Source 1, p.3]."
    docs = [{"source_file": "202541_CSCE_670_600_12345"}]
    out = rewrite_citations(answer, docs)
    assert out == (
        "Late work loses 10% per day [\\[S1, p.3\\]](/app/static/syllabi/202541_CSCE_670_600_12345.pdf#page=3)."
    )


def test_rewrites_without_page_omits_fragment():
    answer = "It is a graduate course [Source 1]."
    docs = [{"source_file": "202541_CSCE_670_600_12345"}]
    out = rewrite_citations(answer, docs)
    assert out == ("It is a graduate course [\\[S1\\]](/app/static/syllabi/202541_CSCE_670_600_12345.pdf).")


def test_missing_source_file_falls_back_to_plain_label():
    answer = "Office hours are Wed 2-4pm [Source 1, p.2]."
    docs = [{"source_file": None}]
    out = rewrite_citations(answer, docs)
    assert "[\\[S1, p.2\\]]" not in out  # no markdown link wrap
    assert "\\[S1, p.2\\]" in out  # plain bracket-escaped label survives


def test_out_of_range_source_left_untouched():
    answer = "Mystery fact [Source 7, p.10]."
    docs = [{"source_file": "stem"}]
    out = rewrite_citations(answer, docs)
    assert out == answer


def test_multiple_citations_per_sentence():
    answer = "Topic A [Source 1, p.3] and Topic B [Source 2, p.7]."
    docs = [
        {"source_file": "stemA"},
        {"source_file": "stemB"},
    ]
    out = rewrite_citations(answer, docs)
    assert "/app/static/syllabi/stemA.pdf#page=3" in out
    assert "/app/static/syllabi/stemB.pdf#page=7" in out


def test_strips_pdf_suffix_on_source_file():
    answer = "Fact [Source 1, p.4]."
    docs = [{"source_file": "stem.pdf"}]
    out = rewrite_citations(answer, docs)
    assert "/app/static/syllabi/stem.pdf#page=4" in out


def test_strips_json_suffix_on_source_file():
    answer = "Fact [Source 1, p.4]."
    docs = [{"source_file": "202611_ISEN_665_600_59566.json"}]
    out = rewrite_citations(answer, docs)
    assert "/app/static/syllabi/202611_ISEN_665_600_59566.pdf#page=4" in out
