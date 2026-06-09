"""v6b_pipeline_ledger: non-partitioned, report-only asset that renders ONE
table answering "which files ran through the pipeline, and how did each fare at
each step?"

Rows = files (syllabus stems, the dynamic partition keys). Columns = the seven
pipeline stages, in execution order. Each cell rolls up that file's status at
that stage:

    ✓  passed      — materialized, its asset checks ran and all are green
    ~  unverified  — materialized, but its checks haven't run yet (or are pending)
    ⚠  warning     — materialized, a WARN (non-blocking) check failed
    ✗  failed      — materialized, an ERROR (blocking) check failed
    ·  not started — stage not materialized for this file

And a per-file row status: passed (all 7 done & green) / in progress (some
stages still not started) / warning / failed / not started.

Below the grid, a "Failures by check" histogram tallies how many files tripped
each check (most-failures-first), then a failures-first "Problems" table names
every failed check — file × stage × check × *why* (the check's own description,
which carries the actual metric vs threshold) — so the one report answers "which
checks failed, in which documents, and what was wrong" without clicking into each
partition.

Scope: by default every *registered* partition is a row (the whole corpus, most
of it never run). Set ``V6B_LEDGER_LAST_RUNS=N`` (and/or ``V6B_LEDGER_SINCE_HOURS``)
to restrict the roster to the files touched by the recent runs — e.g. ``=20``
after a 20-file run renders just that cohort instead of all ~190 syllabi.

REPORT ONLY — reads the Dagster instance + event log, never asset outputs, so it
takes NO `deps` and never mutates anything (same shape as v6b_corpus_report).

The check keys per stage are derived from the live asset graph at runtime, so
the table never drifts when checks are added/removed. The status-derivation
logic lives in the pure `build_ledger()` helper (no Dagster imports) so it is
unit-testable; `_compute` only does the instance I/O.
"""

import json
import time
from datetime import timezone
from pathlib import Path

from dagster import (
    AssetCheckKey,
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)
from dagster._core.event_api import PartitionKeyFilter

from tamubot.core import config
from tamubot.ingestion.pipeline_v5.util import DATA_ROOT
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions

# Dagster stamps the partition key on every partitioned run under this tag; the
# ledger reads it back off recent runs to derive the "files we just ran" cohort.
PARTITION_TAG = "dagster/partition"

# (asset key, short column label) in execution order. struct is the parallel
# branch off bronze; the main chain is bronze->modal->chunk->tag->embed->atlas.
STAGES: list[tuple[str, str]] = [
    ("v6b_bronze_blocks", "bronze"),
    ("v6b_silver_modal", "modal"),
    ("v6b_silver_chunk_semantic", "chunk"),
    ("v6b_silver_tag_semantic", "tag"),
    ("v6b_silver_embed", "embed"),
    ("v6b_silver_atlas_upsert", "atlas"),
    ("v6b_silver_structured", "struct"),
]

GLYPH_PASS = "✓"  # materialized, checks ran & all green
GLYPH_UNVERIFIED = "~"  # materialized, checks defined but not run yet
GLYPH_WARN = "⚠"  # materialized, a WARN (non-blocking) check failed
GLYPH_FAIL = "✗"  # materialized, an ERROR (blocking) check failed
GLYPH_MISSING = "·"  # not materialized

# A check outcome is a plain dict so build_ledger stays Dagster-free:
#   {"name": str, "status": "SUCCEEDED"|"FAILED"|"PLANNED", "severity": "ERROR"|"WARN"}


def _select_recent_stems(
    runs: list[dict],
    last_runs: int,
    since_hours: float,
    now: float,
) -> set[str] | None:
    """Pure: derive the "files we just ran" cohort from recent runs.

    - runs: [{"partition": str|None, "timestamp": float}] newest-first.
    - last_runs: keep only the N newest runs (0 = no cap).
    - since_hours: keep only runs within this window (0 = no window).
    - now: current epoch seconds (injected for testability).

    Returns the set of partition keys (stems) to include, or None when neither
    filter is active (caller treats None as "whole corpus"). When both filters
    are set they intersect: the newest N runs *and* within the window.
    """
    if last_runs <= 0 and since_hours <= 0:
        return None
    selected = runs
    if last_runs > 0:
        selected = selected[:last_runs]
    if since_hours > 0:
        cutoff = now - since_hours * 3600
        selected = [r for r in selected if r.get("timestamp", 0) >= cutoff]
    return {r["partition"] for r in selected if r.get("partition")}


def _check_histogram(rows: list[dict]) -> list[dict]:
    """Pure: roll the per-file failed_checks up into a per-check tally — the
    "how many files failed each check?" view. Sorted most-failures-first."""
    agg: dict[str, dict] = {}
    for r in rows:
        for fc in r.get("failed_checks", []):
            name = fc.get("name", "")
            entry = agg.setdefault(
                name,
                {"name": name, "stage": fc.get("stage", ""), "severity": fc.get("severity", ""), "count": 0},
            )
            entry["count"] += 1
    return sorted(agg.values(), key=lambda d: (-d["count"], d["name"]))


def _cell(materialized: bool, outcomes: list[dict], stage_has_checks: bool) -> str:
    """Roll up one (stem, stage) into a single glyph.

    Failures win over successes; a stage with checks defined but no recorded
    success/failure is 'unverified' (~) rather than claiming a pass.
    """
    if not materialized:
        return GLYPH_MISSING
    failed = [o for o in outcomes if o.get("status") == "FAILED"]
    if any(o.get("severity") == "ERROR" for o in failed):
        return GLYPH_FAIL
    if failed:  # remaining failures are WARN-severity
        return GLYPH_WARN
    if any(o.get("status") == "SUCCEEDED" for o in outcomes):
        return GLYPH_PASS
    # Materialized, no failures, nothing recorded as succeeded:
    return GLYPH_UNVERIFIED if stage_has_checks else GLYPH_PASS


def _row_status(cells: dict[str, str]) -> str:
    """Per-file rollup across stages. Failures/warnings win; otherwise distinguish
    a fully-done file from one still working through the stages."""
    glyphs = set(cells.values())
    if glyphs == {GLYPH_MISSING}:
        return "not_started"
    if GLYPH_FAIL in glyphs:
        return "failed"
    if GLYPH_WARN in glyphs:
        return "warning"
    if GLYPH_MISSING in glyphs:  # some done, some not, no failures
        return "in_progress"
    return "passed"  # every stage done (✓ or ~), nothing failing


def build_ledger(
    roster: list[str],
    materialized_by_stage: dict[str, set[str]],
    checks_by_stage: dict[str, dict[str, list[dict]]],
    stage_has_checks: dict[str, bool],
    scope_label: str | None = None,
) -> dict:
    """Pure builder: turn pre-fetched instance state into the ledger.

    - roster: the stems to render (already scoped by the caller — every
      registered stem for the full-corpus view, or just the recent cohort).
    - materialized_by_stage: {stage_asset: {stem materialized, ...}}.
    - checks_by_stage: {stage_asset: {stem: [check-outcome dicts]}}.
    - stage_has_checks: {stage_asset: bool} — whether the stage defines any checks.
    - scope_label: optional one-line note (e.g. "scoped to last 20 runs") shown
      atop the report; None renders the plain full-corpus header.

    Returns {"rows", "summary", "markdown"}; absent stems/stages are treated as
    missing (never an error) so a registered-but-never-run stem renders cleanly.
    """
    rows: list[dict] = []
    counts = {"passed": 0, "in_progress": 0, "warning": 0, "failed": 0, "not_started": 0}

    for stem in sorted(roster):
        cells: dict[str, str] = {}
        failed_checks: list[dict] = []
        for asset_name, _label in STAGES:
            mat = stem in materialized_by_stage.get(asset_name, set())
            outcomes = checks_by_stage.get(asset_name, {}).get(stem, [])
            cells[asset_name] = _cell(mat, outcomes, stage_has_checks.get(asset_name, False))
            for o in outcomes:
                if o.get("status") == "FAILED":
                    failed_checks.append({"stage": asset_name, **o})

        status = _row_status(cells)
        counts[status] += 1
        rows.append({"stem": stem, "status": status, "cells": cells, "failed_checks": failed_checks})

    by_check = _check_histogram(rows)
    summary = {
        "files": len(rows),
        "passed": counts["passed"],
        "in_progress": counts["in_progress"],
        "with_warnings": counts["warning"],
        "with_failures": counts["failed"],
        "not_started": counts["not_started"],
        "by_check": by_check,
    }
    return {"rows": rows, "summary": summary, "markdown": _render_markdown(rows, summary, scope_label)}


def _render_problems(rows: list[dict]) -> str:
    """Failures-first detail table: one row per failed check, naming the file,
    the stage, the check, and *why* it failed (the check's own description, which
    carries the actual metric vs threshold). This is the "which checks failed, in
    which documents, and what was wrong" view — ERROR-severity first, then WARN."""
    label_of = {key: label for key, label in STAGES}
    lines: list[dict] = []
    for r in rows:
        for fc in r.get("failed_checks", []):
            is_error = fc.get("severity") == "ERROR"
            lines.append(
                {
                    "is_error": is_error,
                    "glyph": GLYPH_FAIL if is_error else GLYPH_WARN,
                    "stem": r["stem"],
                    "stage": label_of.get(fc.get("stage"), fc.get("stage", "")),
                    "name": fc.get("name", ""),
                    "why": fc.get("description") or "(no description)",
                }
            )
    if not lines:
        return "**Problems:** none — every check that ran passed. ✓"
    # ERROR (blocking) failures first; stable order preserves the sorted rows within.
    lines.sort(key=lambda d: 0 if d["is_error"] else 1)
    header = "| | File | Stage | Check | Problem |"
    divider = "|--|------|-------|-------|---------|"
    body = [f"| {d['glyph']} | {d['stem']} | {d['stage']} | {d['name']} | {d['why']} |" for d in lines]
    n_err = sum(1 for d in lines if d["is_error"])
    n_warn = len(lines) - n_err
    title = f"**Problems ({n_err} failing · {n_warn} warning) — failures first:**"
    return title + "\n\n" + "\n".join([header, divider, *body])


def _render_check_histogram(by_check: list[dict]) -> str:
    """Per-check tally: 'N files failed check X' — the aggregate rollup that
    turns the scattered one-per-partition warnings into a ranked problem list."""
    if not by_check:
        return ""
    label_of = {key: label for key, label in STAGES}
    n_err = sum(d["count"] for d in by_check if d.get("severity") == "ERROR")
    n_warn = sum(d["count"] for d in by_check if d.get("severity") != "ERROR")
    title = f"**Failures by check ({n_err} blocking · {n_warn} warning, files affected):**"
    header = "| Files | Check | Stage | Severity |"
    divider = "|------:|-------|-------|----------|"
    body = [
        f"| {d['count']} | {d['name']} | {label_of.get(d['stage'], d['stage'])} | {d.get('severity', '')} |"
        for d in by_check
    ]
    return title + "\n\n" + "\n".join([header, divider, *body])


def _render_markdown(rows: list[dict], summary: dict, scope_label: str | None = None) -> str:
    labels = [label for _key, label in STAGES]
    header = "| File | " + " | ".join(labels) + " |"
    divider = "|------|" + "|".join([":--:"] * len(STAGES)) + "|"
    body = ["| " + r["stem"] + " | " + " | ".join(r["cells"][key] for key, _label in STAGES) + " |" for r in rows]
    summary_line = (
        f"**{summary['files']} files** — "
        f"{summary['passed']} passed · "
        f"{summary['in_progress']} in progress · "
        f"{summary['with_warnings']} with warnings · "
        f"{summary['with_failures']} with failures · "
        f"{summary['not_started']} not started"
    )
    if scope_label:
        summary_line = f"_{scope_label}_\n\n" + summary_line
    legend = (
        "Legend: "
        f"{GLYPH_PASS} passed (checks green) · "
        f"{GLYPH_UNVERIFIED} done, checks not run · "
        f"{GLYPH_WARN} warning (non-blocking check failed) · "
        f"{GLYPH_FAIL} failed (blocking check failed) · "
        f"{GLYPH_MISSING} not started"
    )
    grid = "\n".join([header, divider, *body])
    histogram = _render_check_histogram(summary.get("by_check", []))
    problems = _render_problems(rows)
    sections = [summary_line, legend, grid]
    if histogram:
        sections.append(histogram)
    sections.append(problems)
    return "\n\n".join(sections)


def _stage_check_keys() -> dict[str, list[AssetCheckKey]]:
    """Derive each stage's asset-check keys from the live asset graph, so the
    ledger never drifts when checks are added/removed. Deferred import avoids a
    module-load cycle (definitions imports this asset)."""
    from tamubot.ingestion.pipeline_v6b.definitions import defs

    ag = defs.resolve_asset_graph()
    return {stage: sorted(ag.get(AssetKey(stage)).check_keys, key=lambda k: k.name) for stage, _label in STAGES}


def _fetch_checks_by_stage(
    context: AssetExecutionContext,
    roster: list[str],
    stage_keys: dict[str, list[AssetCheckKey]],
) -> dict[str, dict[str, list[dict]]]:
    """One event-log call per stem (all check keys batched) -> per (stage, stem)
    list of plain check-outcome dicts."""
    els = context.instance.event_log_storage
    stage_of: dict[AssetCheckKey, str] = {}
    all_keys: list[AssetCheckKey] = []
    for stage, keys in stage_keys.items():
        for key in keys:
            stage_of[key] = stage
            all_keys.append(key)

    result: dict[str, dict[str, list[dict]]] = {stage: {} for stage, _ in STAGES}
    if not all_keys:
        return result
    for stem in roster:
        records = els.get_latest_asset_check_execution_by_key(all_keys, partition_filter=PartitionKeyFilter(key=stem))
        for key, record in records.items():
            evaluation = record.evaluation
            severity = evaluation.severity.value if evaluation is not None else "ERROR"
            description = evaluation.description if evaluation is not None else None
            result[stage_of[key]].setdefault(stem, []).append(
                {
                    "name": key.name,
                    "status": record.status.value,
                    "severity": severity,
                    "description": description,
                }
            )
    return result


def _recent_stem_scope(context: AssetExecutionContext, last_runs: int, since_hours: float) -> set[str] | None:
    """Fetch recent runs and reduce them to the cohort of stems to include, or
    None when neither scope toggle is set (full-corpus view)."""
    if last_runs <= 0 and since_hours <= 0:
        return None
    limit = last_runs if last_runs > 0 else 2000
    records = context.instance.get_run_records(limit=limit)
    runs = [
        {
            "partition": rec.dagster_run.tags.get(PARTITION_TAG),
            "timestamp": rec.create_timestamp.replace(tzinfo=timezone.utc).timestamp(),
        }
        for rec in records
    ]
    return _select_recent_stems(runs, last_runs, since_hours, time.time())


def _scope_label(last_runs: int, since_hours: float, n_files: int) -> str:
    bits = []
    if last_runs > 0:
        bits.append(f"last {last_runs} runs")
    if since_hours > 0:
        bits.append(f"last {since_hours:g}h")
    return (
        f"Scoped to {' ∩ '.join(bits)} — {n_files} file(s). "
        "Unset V6B_LEDGER_LAST_RUNS / V6B_LEDGER_SINCE_HOURS for the full corpus."
    )


def _compute_pipeline_ledger(context: AssetExecutionContext) -> MaterializeResult:
    roster = list(context.instance.get_dynamic_partitions(stem_partitions.name))

    include = _recent_stem_scope(context, config.V6B_LEDGER_LAST_RUNS, config.V6B_LEDGER_SINCE_HOURS)
    scope_label: str | None = None
    if include is not None:
        roster = [s for s in roster if s in include]
        scope_label = _scope_label(config.V6B_LEDGER_LAST_RUNS, config.V6B_LEDGER_SINCE_HOURS, len(roster))

    materialized_by_stage = {
        asset_name: context.instance.get_materialized_partitions(AssetKey(asset_name)) for asset_name, _label in STAGES
    }
    stage_keys = _stage_check_keys()
    stage_has_checks = {stage: len(keys) > 0 for stage, keys in stage_keys.items()}
    checks_by_stage = _fetch_checks_by_stage(context, roster, stage_keys)

    ledger = build_ledger(roster, materialized_by_stage, checks_by_stage, stage_has_checks, scope_label)
    summary = ledger["summary"]

    out_path = Path(DATA_ROOT) / "_meta" / "v6b_pipeline_ledger.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": ledger["rows"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    context.log.info(
        "v6b_pipeline_ledger: %d files (%d passed, %d in progress, %d warn, %d fail, %d not started)",
        summary["files"],
        summary["passed"],
        summary["in_progress"],
        summary["with_warnings"],
        summary["with_failures"],
        summary["not_started"],
    )

    metadata = {
        "ledger": MetadataValue.md(ledger["markdown"]),
        "files": summary["files"],
        "passed": summary["passed"],
        "in_progress": summary["in_progress"],
        "with_warnings": summary["with_warnings"],
        "with_failures": summary["with_failures"],
        "not_started": summary["not_started"],
        "failing_checks": len(summary["by_check"]),
        "scope": scope_label or "full corpus",
        "report_path": str(out_path),
    }
    return MaterializeResult(metadata=metadata)


v6b_pipeline_ledger = asset(
    name="v6b_pipeline_ledger",
    group_name="v6b_ops",
    description=(
        "Report-only: one table of every file (syllabus stem) × pipeline stage, each cell "
        "rolling up materialization + asset-check status (✓ passed / ~ checks-not-run / "
        "⚠ warning / ✗ failed / · not started), plus a failures-first 'Problems' table "
        "naming each failed check's file, stage, and why (metric vs threshold). Reads the "
        "Dagster instance + event log; no deps, never mutates."
    ),
)(_compute_pipeline_ledger)
