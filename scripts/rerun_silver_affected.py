"""One-shot: rerun silver stages on stems affected by recent fixes.

For each stem:
- If in REGISTRY_FIX stems: rerun BoilerplateFilter (02_false_positive -> 03_boilerplate),
  then RelocateTextbookFilter (03_boilerplate -> 03b_relocate_textbook).
- If in CLEANUP_FIX stems (Credit Hours dedup / cover desc strip): rerun only
  RelocateTextbookFilter, since boilerplate output didn't change.

Reuses the same filter classes the Dagster assets use, so output is identical
to a full Dagster re-materialization for those partitions — minus the Dagster
event log.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from tamubot.ingestion.filters.boilerplate import BoilerplateFilter
from tamubot.ingestion.filters.relocate_textbook import RelocateTextbookFilter
from tamubot.ingestion.pipeline_v5 import paths

REGISTRY_FIX = [
    "202611_CSCE_625_600_19180_HP",
    "202611_CSCE_632_600_54784",
    "202611_CSCE_633_600_12367_HP",
    "202611_CSCE_636_600_42745_HP",
    "202641_CSCE_605_600_62077",
]

CLEANUP_FIX = [
    # Credit Hours dedup
    "202621_CSCE_616_700_35075",
    "202621_CSCE_636_700_31735",
    "202621_CSCE_685_300_25489",
    "202621_CSCE_685_326_25490",
    "202621_CSCE_685_350_25983",
    # Cover description strip
    "202611_CSCE_612_600_42640",
    "202611_CSCE_614_600_42743",
    "202611_CSCE_629_700_54783",
    "202611_CSCE_633_601_58706",
    "202621_CSCE_629_700_31702",
    "202641_CSCE_627_600_58328",
]


def rerun_boilerplate(stem: str) -> dict:
    src = paths.silver_path(stem, "false_positive")
    dst = paths.silver_path(stem, "boilerplate")
    sidecar_dst = dst.parent / f"{stem}_stripped.txt"
    log_entry: dict = {}
    with (
        tempfile.TemporaryDirectory() as ti,
        tempfile.TemporaryDirectory() as to,
        patch("tamubot.ingestion.report_writer.get_report", return_value=None),
        patch("tamubot.ingestion.report_writer.update_boilerplate"),
    ):
        ti_p, to_p = Path(ti), Path(to)
        shutil.copy2(src, ti_p / f"{stem}.md")
        result = BoilerplateFilter().apply(ti_p, to_p, config={"file_pattern": f"{stem}.md"})
        shutil.copy2(to_p / f"{stem}.md", dst)
        sidecar_tmp = to_p / f"{stem}_stripped.txt"
        if sidecar_tmp.exists():
            shutil.copy2(sidecar_tmp, sidecar_dst)
        if result.log_entries:
            log_entry = result.log_entries[0]
    return log_entry


def rerun_relocate(stem: str) -> dict:
    src = paths.silver_path(stem, "boilerplate")
    dst = paths.silver_path(stem, "relocate_textbook")
    log_entry: dict = {}
    with tempfile.TemporaryDirectory() as ti, tempfile.TemporaryDirectory() as to:
        ti_p, to_p = Path(ti), Path(to)
        shutil.copy2(src, ti_p / f"{stem}.md")
        result = RelocateTextbookFilter().apply(ti_p, to_p, config={"file_pattern": f"{stem}.md", "mode": "move"})
        shutil.copy2(to_p / f"{stem}.md", dst)
        if result.log_entries:
            log_entry = result.log_entries[0]
    return log_entry


def main() -> None:
    print(f"=== Rerunning boilerplate + relocate for {len(REGISTRY_FIX)} stems (registry adds) ===")
    for stem in REGISTRY_FIX:
        bp = rerun_boilerplate(stem)
        rt = rerun_relocate(stem)
        print(
            f"  {stem}: bp_stripped={bp.get('sections_stripped', 0)} "
            f"bp_tokens_removed={bp.get('tokens_removed', 0)} "
            f"moved={rt.get('blocks_moved', 0)} "
            f"credit_dedup={rt.get('credit_hours_dedup', 0)} "
            f"cover_stripped={rt.get('cover_desc_stripped', 0)}"
        )

    print(f"\n=== Rerunning relocate only for {len(CLEANUP_FIX)} stems (cleanup fixes) ===")
    for stem in CLEANUP_FIX:
        rt = rerun_relocate(stem)
        print(
            f"  {stem}: moved={rt.get('blocks_moved', 0)} "
            f"credit_dedup={rt.get('credit_hours_dedup', 0)} "
            f"cover_stripped={rt.get('cover_desc_stripped', 0)}"
        )


if __name__ == "__main__":
    main()
