"""Run LLM validator on the 16 stems we re-processed, writing v2 snapshots.

Reads input from silver/03b_relocate_textbook (the last v5 silver stage).
Writes both <stem>_validation.json (canonical latest) and
<stem>_validation_v2.json (versioned snapshot) into silver/06_validate.
"""

from __future__ import annotations

from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.validators.llm_validator import validate_file

AFFECTED = [
    # Registry-add stems
    "202611_CSCE_625_600_19180_HP",
    "202611_CSCE_632_600_54784",
    "202611_CSCE_633_600_12367_HP",
    "202611_CSCE_636_600_42745_HP",
    "202641_CSCE_605_600_62077",
    # Credit-Hours-dedup stems
    "202621_CSCE_616_700_35075",
    "202621_CSCE_636_700_31735",
    "202621_CSCE_685_300_25489",
    "202621_CSCE_685_326_25490",
    "202621_CSCE_685_350_25983",
    # Cover-description-strip stems
    "202611_CSCE_612_600_42640",
    "202611_CSCE_614_600_42743",
    "202611_CSCE_629_700_54783",
    "202611_CSCE_633_601_58706",
    "202621_CSCE_629_700_31702",
    "202641_CSCE_627_600_58328",
]


def main() -> None:
    out_dir = paths.silver_path(AFFECTED[0], "validate").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Validating {len(AFFECTED)} stems ...")

    total = 0
    for stem in AFFECTED:
        md_path = paths.silver_path(stem, "relocate_textbook")
        if not md_path.exists():
            print(f"  SKIP {stem}: {md_path} missing")
            continue
        vr = validate_file(
            md_path,
            metadata=None,
            output_dir=out_dir,
            version_label="v2",
        )
        counts = vr.issue_counts
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0) or "clean"
        print(f"  {stem}: {summary} (total={vr.total_issues}, {vr.timing_s:.1f}s)")
        total += vr.total_issues

    print(f"\nTotal v2 findings across {len(AFFECTED)} files: {total}")


if __name__ == "__main__":
    main()
