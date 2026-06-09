"""Unit tests for the v6b pipeline-ledger builder.

Pure — exercises build_ledger / _cell on plain dicts, no Dagster instance.
"""

from __future__ import annotations

from tamubot.ingestion.pipeline_v6b.assets.pipeline_ledger import (
    GLYPH_FAIL,
    GLYPH_MISSING,
    GLYPH_PASS,
    GLYPH_UNVERIFIED,
    GLYPH_WARN,
    STAGES,
    _cell,
    _check_histogram,
    _select_recent_stems,
    build_ledger,
)

ALL_STAGES = [name for name, _label in STAGES]
# Every stage defines at least one check in the live pipeline.
HAS_CHECKS = {name: True for name in ALL_STAGES}


def _ok(name: str) -> dict:
    return {"name": name, "status": "SUCCEEDED", "severity": "ERROR"}


def _failed(name: str, severity: str) -> dict:
    return {"name": name, "status": "FAILED", "severity": severity}


def _planned(name: str) -> dict:
    return {"name": name, "status": "PLANNED", "severity": "ERROR"}


# --- _cell -----------------------------------------------------------------


def test_cell_missing_when_not_materialized():
    # Even with a failing check on record, a non-materialized stage is missing.
    assert _cell(False, [_failed("x", "ERROR")], True) == GLYPH_MISSING


def test_cell_unverified_when_materialized_but_checks_not_run():
    # Stage has checks but none have a success/fail record yet.
    assert _cell(True, [], True) == GLYPH_UNVERIFIED
    assert _cell(True, [_planned("x")], True) == GLYPH_UNVERIFIED


def test_cell_pass_when_stage_has_no_checks_defined():
    assert _cell(True, [], False) == GLYPH_PASS


def test_cell_pass_when_a_check_succeeded():
    assert _cell(True, [_ok("a"), _ok("b")], True) == GLYPH_PASS


def test_cell_warn_on_warn_failure():
    assert _cell(True, [_ok("a"), _failed("b", "WARN")], True) == GLYPH_WARN


def test_cell_fail_on_error_failure():
    assert _cell(True, [_failed("a", "ERROR"), _failed("b", "WARN")], True) == GLYPH_FAIL


# --- build_ledger ----------------------------------------------------------


def _materialized_everywhere(stems: list[str]) -> dict[str, set[str]]:
    return {name: set(stems) for name in ALL_STAGES}


def _all_green(stems: list[str]) -> dict[str, dict[str, list[dict]]]:
    return {name: {s: [_ok(f"{name}_chk")] for s in stems} for name in ALL_STAGES}


def test_full_pass_row():
    stems = ["202611_CSCE_601_600_111"]
    ledger = build_ledger(stems, _materialized_everywhere(stems), _all_green(stems), HAS_CHECKS)
    assert ledger["summary"] == {
        "files": 1,
        "passed": 1,
        "in_progress": 0,
        "with_warnings": 0,
        "with_failures": 0,
        "not_started": 0,
        "by_check": [],
    }
    row = ledger["rows"][0]
    assert row["status"] == "passed"
    assert set(row["cells"].values()) == {GLYPH_PASS}


def test_materialized_everywhere_but_no_checks_run_is_unverified_pass():
    # All stages done but no check records -> cells are '~', row still counts as passed
    # (nothing missing, nothing failing) but the cells make the unverified state visible.
    stems = ["202611_CSCE_601_600_111"]
    ledger = build_ledger(stems, _materialized_everywhere(stems), {}, HAS_CHECKS)
    row = ledger["rows"][0]
    assert set(row["cells"].values()) == {GLYPH_UNVERIFIED}
    assert row["status"] == "passed"


def test_not_started_row():
    stems = ["202611_CSCE_602_600_222"]
    ledger = build_ledger(stems, {name: set() for name in ALL_STAGES}, {}, HAS_CHECKS)
    assert ledger["summary"]["not_started"] == 1
    row = ledger["rows"][0]
    assert row["status"] == "not_started"
    assert set(row["cells"].values()) == {GLYPH_MISSING}


def test_partial_materialization_is_in_progress_not_passed():
    # bronze+chunk done (green), rest not started -> 'in_progress', NOT 'passed'.
    stem = "202611_CSCE_605_600_555"
    materialized = {name: set() for name in ALL_STAGES}
    materialized["v6b_bronze_blocks"] = {stem}
    materialized["v6b_silver_chunk_semantic"] = {stem}
    checks = {
        "v6b_bronze_blocks": {stem: [_ok("v6b_bronze_blocks_nonempty")]},
        "v6b_silver_chunk_semantic": {stem: [_ok("v6b_silver_chunk_count_nonzero")]},
    }
    ledger = build_ledger([stem], materialized, checks, HAS_CHECKS)
    row = ledger["rows"][0]
    assert row["cells"]["v6b_bronze_blocks"] == GLYPH_PASS
    assert row["cells"]["v6b_silver_embed"] == GLYPH_MISSING
    assert row["status"] == "in_progress"
    assert ledger["summary"] == {
        "files": 1,
        "passed": 0,
        "in_progress": 1,
        "with_warnings": 0,
        "with_failures": 0,
        "not_started": 0,
        "by_check": [],
    }


def test_error_failure_classifies_row_as_failed():
    stem = "202611_CSCE_603_600_333"
    materialized = _materialized_everywhere([stem])
    checks = _all_green([stem])
    checks["v6b_silver_tag_semantic"][stem] = [_failed("v6b_silver_tag_chunk_count_preserved", "ERROR")]
    ledger = build_ledger([stem], materialized, checks, HAS_CHECKS)
    assert ledger["summary"]["with_failures"] == 1
    row = ledger["rows"][0]
    assert row["status"] == "failed"
    assert row["cells"]["v6b_silver_tag_semantic"] == GLYPH_FAIL
    assert row["failed_checks"] == [
        {
            "stage": "v6b_silver_tag_semantic",
            "name": "v6b_silver_tag_chunk_count_preserved",
            "status": "FAILED",
            "severity": "ERROR",
        }
    ]


def test_warn_failure_classifies_row_as_warning():
    stem = "202611_CSCE_604_600_444"
    materialized = _materialized_everywhere([stem])
    checks = _all_green([stem])
    checks["v6b_silver_tag_semantic"][stem] = [_failed("v6b_silver_tag_duplicate_rate_in_band", "WARN")]
    ledger = build_ledger([stem], materialized, checks, HAS_CHECKS)
    assert ledger["summary"]["with_warnings"] == 1
    assert ledger["rows"][0]["status"] == "warning"
    assert ledger["rows"][0]["cells"]["v6b_silver_tag_semantic"] == GLYPH_WARN


def test_failure_outranks_missing_downstream():
    # A blocking failure with downstream not yet started is still 'failed', not 'in_progress'.
    stem = "202611_CSCE_606_600_666"
    materialized = {name: set() for name in ALL_STAGES}
    materialized["v6b_bronze_blocks"] = {stem}
    checks = {"v6b_bronze_blocks": {stem: [_failed("v6b_bronze_blocks_nonempty", "ERROR")]}}
    ledger = build_ledger([stem], materialized, checks, HAS_CHECKS)
    assert ledger["rows"][0]["status"] == "failed"


# --- _select_recent_stems (run-scope) --------------------------------------


def _run(partition: str | None, timestamp: float = 0.0) -> dict:
    return {"partition": partition, "timestamp": timestamp}


def test_recent_stems_none_when_no_filters():
    # No scope toggles -> None signals "whole corpus" to the caller.
    runs = [_run("a"), _run("b")]
    assert _select_recent_stems(runs, last_runs=0, since_hours=0, now=100.0) is None


def test_recent_stems_last_n_takes_newest_runs_only():
    runs = [_run("a"), _run("b"), _run("c"), _run("d")]  # newest-first
    assert _select_recent_stems(runs, last_runs=2, since_hours=0, now=100.0) == {"a", "b"}


def test_recent_stems_dedupes_partitions_across_runs():
    # 3 runs but only 2 distinct files (a re-run of "a") -> set of 2.
    runs = [_run("a"), _run("a"), _run("b")]
    assert _select_recent_stems(runs, last_runs=3, since_hours=0, now=100.0) == {"a", "b"}


def test_recent_stems_skips_runs_without_partition():
    # Non-partitioned runs (e.g. the ledger itself) carry no partition tag.
    runs = [_run(None), _run("a"), _run(None)]
    assert _select_recent_stems(runs, last_runs=3, since_hours=0, now=100.0) == {"a"}


def test_recent_stems_since_hours_windows_by_timestamp():
    now = 10_000.0
    runs = [_run("fresh", now - 1800), _run("stale", now - 7200)]  # 0.5h vs 2h ago
    assert _select_recent_stems(runs, last_runs=0, since_hours=1, now=now) == {"fresh"}


def test_recent_stems_intersects_last_n_and_window():
    now = 10_000.0
    runs = [_run("a", now - 60), _run("b", now - 7200), _run("c", now - 30)]
    # last_runs=2 -> {a,b}; window 1h drops b -> {a}.
    assert _select_recent_stems(runs, last_runs=2, since_hours=1, now=now) == {"a"}


# --- _check_histogram + markdown rollup ------------------------------------


def test_check_histogram_counts_files_per_check_sorted():
    stems = ["s1", "s2", "s3"]
    materialized = _materialized_everywhere(stems)
    checks = _all_green(stems)
    # boilerplate fails on all 3; duplicate fails on 1.
    for s in stems:
        checks["v6b_silver_tag_semantic"][s] = [_failed("v6b_silver_tag_boilerplate_rate_in_band", "WARN")]
    checks["v6b_silver_tag_semantic"]["s1"].append(_failed("v6b_silver_tag_duplicate_rate_in_band", "WARN"))
    ledger = build_ledger(stems, materialized, checks, HAS_CHECKS)
    hist = ledger["summary"]["by_check"]
    assert hist[0] == {
        "name": "v6b_silver_tag_boilerplate_rate_in_band",
        "stage": "v6b_silver_tag_semantic",
        "severity": "WARN",
        "count": 3,
    }
    assert hist[1]["name"] == "v6b_silver_tag_duplicate_rate_in_band"
    assert hist[1]["count"] == 1
    # markdown surfaces the rollup table
    assert "Failures by check" in ledger["markdown"]
    assert _check_histogram([]) == []


def test_scope_label_rendered_in_markdown():
    stems = ["202611_CSCE_601_600_111"]
    ledger = build_ledger(
        stems,
        _materialized_everywhere(stems),
        _all_green(stems),
        HAS_CHECKS,
        scope_label="Scoped to last 20 runs — 1 file(s).",
    )
    assert "Scoped to last 20 runs" in ledger["markdown"]


def test_rows_sorted_and_markdown_has_legend_and_rows():
    stems = ["202611_ZZZZ_999_600_2", "202611_AAAA_100_600_1"]
    ledger = build_ledger(stems, _materialized_everywhere(stems), _all_green(stems), HAS_CHECKS)
    assert [r["stem"] for r in ledger["rows"]] == sorted(stems)
    md = ledger["markdown"]
    for s in stems:
        assert s in md
    for _name, label in STAGES:
        assert label in md
    # legend documents every glyph
    for g in (GLYPH_PASS, GLYPH_UNVERIFIED, GLYPH_WARN, GLYPH_FAIL, GLYPH_MISSING):
        assert g in md
