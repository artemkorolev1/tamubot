"""Append a row to data/syllabi/<DEPT>/v5/logs/manual_edits.csv.

Use this whenever a silver-stage markdown is hand-edited (by main agent,
subagent, or human via Excel/IDE). The v5 report builder reads this CSV
into a Manual Edits column so each iteration's hand-edits stay visible.

Usage:
    python scripts/log_manual_edit.py \\
        --dept CSCE \\
        --stem 202641_CSCE_605_600_62077 \\
        --stage 03b_relocate_textbook \\
        --applied-by agent-a48adbf \\
        --summary "Merged broken URL; demoted Class Participation H4; renumbered list" \\
        [--llm-calls 0]

Schema (append-only): timestamp,stem,stage,applied_by,llm_calls,summary
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path

HEADER = ["timestamp", "stem", "stage", "applied_by", "llm_calls", "summary"]


def log_edit(
    dept: str,
    stem: str,
    stage: str,
    applied_by: str,
    summary: str,
    llm_calls: int = 0,
    log_path: Path | None = None,
) -> Path:
    if log_path is None:
        log_path = Path(f"data/syllabi/{dept.upper()}/v5/logs/manual_edits.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(HEADER)
        w.writerow([ts, stem, stage, applied_by, llm_calls, summary])
    return log_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a row to manual_edits.csv")
    ap.add_argument("--dept", required=True, help="Department code (e.g. CSCE)")
    ap.add_argument("--stem", required=True, help="File stem (e.g. 202641_CSCE_605_600_62077)")
    ap.add_argument("--stage", required=True, help="Silver subdir (e.g. 03b_relocate_textbook)")
    ap.add_argument("--applied-by", required=True, help="agent-<id> | main | human")
    ap.add_argument("--summary", required=True, help="One-line description of the edits")
    ap.add_argument("--llm-calls", type=int, default=0, help="LLM calls consumed by the edit itself")
    args = ap.parse_args()
    out = log_edit(
        dept=args.dept,
        stem=args.stem,
        stage=args.stage,
        applied_by=args.applied_by,
        summary=args.summary,
        llm_calls=args.llm_calls,
    )
    print(f"Appended to {out}")


if __name__ == "__main__":
    main()
