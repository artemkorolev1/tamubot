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

REPORT ONLY — reads the Dagster instance + event log, never asset outputs, so it
takes NO `deps` and never mutates anything (same shape as v6b_corpus_report).

The check keys per stage are derived from the live asset graph at runtime, so
the table never drifts when checks are added/removed. The status-derivation
logic lives in the pure `build_ledger()` helper (no Dagster imports) so it is
unit-testable; `_compute` only does the instance I/O.
"""

import json
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

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions

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
) -> dict:
    """Pure builder: turn pre-fetched instance state into the ledger.

    - roster: every registered stem (dynamic partition keys).
    - materialized_by_stage: {stage_asset: {stem materialized, ...}}.
    - checks_by_stage: {stage_asset: {stem: [check-outcome dicts]}}.
    - stage_has_checks: {stage_asset: bool} — whether the stage defines any checks.

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

    summary = {
        "files": len(rows),
        "passed": counts["passed"],
        "in_progress": counts["in_progress"],
        "with_warnings": counts["warning"],
        "with_failures": counts["failed"],
        "not_started": counts["not_started"],
    }
    return {"rows": rows, "summary": summary, "markdown": _render_markdown(rows, summary)}


def _render_markdown(rows: list[dict], summary: dict) -> str:
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
    legend = (
        "Legend: "
        f"{GLYPH_PASS} passed (checks green) · "
        f"{GLYPH_UNVERIFIED} done, checks not run · "
        f"{GLYPH_WARN} warning (non-blocking check failed) · "
        f"{GLYPH_FAIL} failed (blocking check failed) · "
        f"{GLYPH_MISSING} not started"
    )
    return summary_line + "\n\n" + legend + "\n\n" + "\n".join([header, divider, *body])


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
            result[stage_of[key]].setdefault(stem, []).append(
                {"name": key.name, "status": record.status.value, "severity": severity}
            )
    return result


def _compute_pipeline_ledger(context: AssetExecutionContext) -> MaterializeResult:
    roster = list(context.instance.get_dynamic_partitions(stem_partitions.name))

    materialized_by_stage = {
        asset_name: context.instance.get_materialized_partitions(AssetKey(asset_name)) for asset_name, _label in STAGES
    }
    stage_keys = _stage_check_keys()
    stage_has_checks = {stage: len(keys) > 0 for stage, keys in stage_keys.items()}
    checks_by_stage = _fetch_checks_by_stage(context, roster, stage_keys)

    ledger = build_ledger(roster, materialized_by_stage, checks_by_stage, stage_has_checks)
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

    return MaterializeResult(
        metadata={
            "ledger": MetadataValue.md(ledger["markdown"]),
            "files": summary["files"],
            "passed": summary["passed"],
            "in_progress": summary["in_progress"],
            "with_warnings": summary["with_warnings"],
            "with_failures": summary["with_failures"],
            "not_started": summary["not_started"],
            "report_path": str(out_path),
        }
    )


v6b_pipeline_ledger = asset(
    name="v6b_pipeline_ledger",
    group_name="v6b_ops",
    description=(
        "Report-only: one table of every file (syllabus stem) × pipeline stage, each cell "
        "rolling up materialization + asset-check status (✓ passed / ~ checks-not-run / "
        "⚠ warning / ✗ failed / · not started). Reads the Dagster instance + event log; "
        "no deps, never mutates."
    ),
)(_compute_pipeline_ledger)
