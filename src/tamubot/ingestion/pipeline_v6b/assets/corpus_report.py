"""v6b_corpus_report: non-partitioned scan of disk vs Atlas. Reports orphaned
vectors (Atlas stems absent on disk) and per-department volume drift. REPORT
ONLY — never deletes (no-delete-without-asking policy)."""

import json
from pathlib import Path

from dagster import MaterializeResult, OpExecutionContext, asset

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT, dept_from_stem
from tamubot.ingestion.validation.baseline_diff import compute_baseline_delta
from tamubot.ingestion.validation.orphans import compute_orphans

# Atlas collection name and chunk_tag used by silver_atlas_upsert.
_CHUNK_TAG = "v6b_semantic"
_COLLECTION = "chunks_v4"
# Stem field stored per chunk doc in Atlas (set by silver_chunk_semantic).
_STEM_FIELD = "source_file"


def build_report(
    disk_stems_by_dept: dict[str, set[str]],
    atlas_stems_by_dept: dict[str, set[str]],
    volume_history_by_dept: dict[str, list[float]],
) -> dict:
    """Pure function: compute orphan + volume-drift report from pre-scanned inputs."""
    report: dict = {}
    total_orphans = 0
    depts = set(disk_stems_by_dept) | set(atlas_stems_by_dept)
    for dept in sorted(depts):
        disk = disk_stems_by_dept.get(dept, set())
        atlas = atlas_stems_by_dept.get(dept, set())
        orphans = compute_orphans(disk_stems=disk, atlas_stems=atlas)
        vol = compute_baseline_delta(
            current=float(len(disk)),
            history=volume_history_by_dept.get(dept, []),
        )
        total_orphans += len(orphans.orphan_stems)
        report[dept] = {
            "disk_count": len(disk),
            "atlas_count": len(atlas),
            "orphan_stems": orphans.orphan_stems,
            "missing_from_atlas": orphans.missing_from_atlas,
            "volume_drift": {"passed": vol.passed, **vol.metadata},
        }
    report["_summary"] = {"total_orphans": total_orphans, "depts": len(depts)}
    return report


def _scan_disk_stems_by_dept() -> dict[str, set[str]]:
    """Return {dept: {stem, ...}} by globbing bronze blocks files on disk.

    Path pattern: DATA_ROOT / <dept> / v6b / bronze / <stem>.blocks.json
    parts index -4 = dept, filename without '.blocks.json' = stem.
    """
    result: dict[str, set[str]] = {}
    root = Path(DATA_ROOT)
    for p in root.glob("*/v6b/bronze/*.blocks.json"):
        dept = p.parts[-4]
        stem = p.name.removesuffix(".blocks.json")
        result.setdefault(dept, set()).add(stem)
    return result


def _scan_atlas_stems_by_dept() -> dict[str, set[str]]:
    """Return {dept: {stem, ...}} from Atlas chunks_v4 (chunk_tag=v6b_semantic).

    Each chunk doc carries source_file (= stem). dept is derived via
    dept_from_stem() — same helper used by silver_chunk_semantic.
    """
    import os

    from pymongo import MongoClient

    from tamubot.core import config

    uri = os.getenv("MONGO_URI") or config.MONGO_URI
    db_name = os.getenv("MONGO_DB") or config.MONGO_DB
    client: MongoClient = MongoClient(uri)
    db = client[db_name]

    result: dict[str, set[str]] = {}
    cursor = db[_COLLECTION].find(
        {"chunk_tag": _CHUNK_TAG},
        {_STEM_FIELD: 1, "_id": 0},
    )
    for doc in cursor:
        stem = doc.get(_STEM_FIELD)
        if not stem:
            continue
        dept = dept_from_stem(stem)
        result.setdefault(dept, set()).add(stem)
    client.close()
    return result


def _compute_corpus_report(context: OpExecutionContext) -> MaterializeResult:
    disk_by_dept = _scan_disk_stems_by_dept()
    atlas_by_dept = _scan_atlas_stems_by_dept()

    # No volume history available at runtime — pass empty so drift check is skipped
    # gracefully (compute_baseline_delta returns passed=True with empty history).
    report = build_report(
        disk_stems_by_dept=disk_by_dept,
        atlas_stems_by_dept=atlas_by_dept,
        volume_history_by_dept={},
    )

    out_path = Path(DATA_ROOT) / "_meta" / "v6b_corpus_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report.get("_summary", {})
    context.log.info(
        "v6b_corpus_report: %d depts, %d orphaned Atlas stems",
        summary.get("depts", 0),
        summary.get("total_orphans", 0),
    )

    return MaterializeResult(
        metadata={
            "total_orphans": summary.get("total_orphans", 0),
            "depts": summary.get("depts", 0),
            "report_path": str(out_path),
        }
    )


v6b_corpus_report = asset(
    name="v6b_corpus_report",
    group_name="v6b_ops",
    description=("Report-only scan: orphaned Atlas vectors + per-dept volume drift. Never deletes."),
)(_compute_corpus_report)
