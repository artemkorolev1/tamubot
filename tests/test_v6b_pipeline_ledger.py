"""Unit tests for the v6b pipeline-ledger builder.

Pure — exercises build_ledger / _cell on plain dicts, no Dagster instance.
"""

from __future__ import annotations

from tamubot.ingestion.pipeline_v6b.assets.pipeline_ledger import (
    GLYPH_FAIL,
    GLYPH_MISSING,
    GLYPH_PASS,
    GLYPH_WARN,
    STAGES,
    _cell,
    build_ledger,
)

ALL_STAGES = [name for name, _label in STAGES]


def _ok(name: str) -> dict:
    return {"name": name, "status": "SUCCEEDED", "severity": "ERROR"}


def _failed(name: str, severity: str) -> dict:
    return {"name": name, "status": "FAILED", "severity": severity}


# --- _cell -----------------------------------------------------------------


def test_cell_missing_when_not_materialized():
    # Even with a failing check on record, a non-materialized stage is missing.
    assert _cell(False, [_failed("x", "ERROR")]) == GLYPH_MISSING


def test_cell_pass_when_materialized_no_checks():
    assert _cell(True, []) == GLYPH_PASS


def test_cell_pass_when_all_checks_succeed():
    assert _cell(True, [_ok("a"), _ok("b")]) == GLYPH_PASS


def test_cell_warn_on_warn_failure():
    assert _cell(True, [_ok("a"), _failed("b", "WARN")]) == GLYPH_WARN


def test_cell_fail_on_error_failure():
    assert _cell(True, [_failed("a", "ERROR"), _failed("b", "WARN")]) == GLYPH_FAIL


# --- build_ledger ----------------------------------------------------------


def _materialized_everywhere(stems: list[str]) -> dict[str, set[str]]:
    return {name: set(stems) for name in ALL_STAGES}


def test_full_pass_row():
    stems = ["202611_CSCE_601_600_111"]
    ledger = build_ledger(stems, _materialized_everywhere(stems), {})
    assert ledger["summary"] == {
        "files": 1,
        "fully_passed": 1,
        "with_warnings": 0,
        "with_failures": 0,
        "not_started": 0,
    }
    row = ledger["rows"][0]
    assert row["status"] == "pass"
    assert set(row["cells"].values()) == {GLYPH_PASS}


def test_not_started_row():
    stems = ["202611_CSCE_602_600_222"]
    ledger = build_ledger(stems, {name: set() for name in ALL_STAGES}, {})
    assert ledger["summary"]["not_started"] == 1
    row = ledger["rows"][0]
    assert row["status"] == "not_started"
    assert set(row["cells"].values()) == {GLYPH_MISSING}


def test_error_failure_classifies_row_as_fail():
    stem = "202611_CSCE_603_600_333"
    materialized = _materialized_everywhere([stem])
    checks = {"v6b_silver_tag_semantic": {stem: [_failed("v6b_silver_tag_chunk_count_preserved", "ERROR")]}}
    ledger = build_ledger([stem], materialized, checks)
    assert ledger["summary"]["with_failures"] == 1
    row = ledger["rows"][0]
    assert row["status"] == "fail"
    assert row["cells"]["v6b_silver_tag_semantic"] == GLYPH_FAIL
    assert row["failed_checks"] == [
        {
            "stage": "v6b_silver_tag_semantic",
            "name": "v6b_silver_tag_chunk_count_preserved",
            "status": "FAILED",
            "severity": "ERROR",
        }
    ]


def test_warn_failure_classifies_row_as_warn():
    stem = "202611_CSCE_604_600_444"
    materialized = _materialized_everywhere([stem])
    checks = {"v6b_silver_tag_semantic": {stem: [_failed("v6b_silver_tag_duplicate_rate_in_band", "WARN")]}}
    ledger = build_ledger([stem], materialized, checks)
    assert ledger["summary"]["with_warnings"] == 1
    assert ledger["rows"][0]["status"] == "warn"
    assert ledger["rows"][0]["cells"]["v6b_silver_tag_semantic"] == GLYPH_WARN


def test_partial_materialization_marks_missing_downstream():
    # bronze+chunk done, nothing else -> downstream cells missing, row not "pass".
    stem = "202611_CSCE_605_600_555"
    materialized = {name: set() for name in ALL_STAGES}
    materialized["v6b_bronze_blocks"] = {stem}
    materialized["v6b_silver_chunk_semantic"] = {stem}
    ledger = build_ledger([stem], materialized, {})
    row = ledger["rows"][0]
    assert row["cells"]["v6b_bronze_blocks"] == GLYPH_PASS
    assert row["cells"]["v6b_silver_embed"] == GLYPH_MISSING
    # some stage materialized -> not "not_started"; no failures -> "pass"
    assert row["status"] == "pass"
    assert ledger["summary"]["not_started"] == 0


def test_rows_sorted_and_markdown_has_all_rows():
    stems = ["202611_ZZZZ_999_600_2", "202611_AAAA_100_600_1"]
    ledger = build_ledger(stems, _materialized_everywhere(stems), {})
    assert [r["stem"] for r in ledger["rows"]] == sorted(stems)
    md = ledger["markdown"]
    for s in stems:
        assert s in md
    # header carries the short stage labels
    for _name, label in STAGES:
        assert label in md
