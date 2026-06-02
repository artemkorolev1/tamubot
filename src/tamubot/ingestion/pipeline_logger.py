"""
tamubot.ingestion.pipeline_logger

Versioned step logging for the V3 syllabus pipeline.

Flat output: all versions of a file live in the same step root directory,
distinguished by _vNNN suffix (e.g. 202611_CSCE_670_v001.md, _v002.md).
A single combined CSV+JSONL log per step accumulates across all runs.

Provides:
  - resolve_version(force_new) -> "v001" ... "vNNN"
  - resolve_latest_file(step_dir, stem, suffix) -> Path | None
  - write_run_meta(step_root, version, cli_args)
  - StepLogger — append-only CSV+JSONL writer with Excel-safe atomic writes
"""

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

# Step root directories (no vNNN subfolders)
STEP0_ROOT = Path("tamu_data/processed/v3_step0_source")
STEP1_ROOT = Path("tamu_data/processed/v3_step1_markdown")
STEP2_ROOT = Path("tamu_data/processed/v3_step2_boilerplate")
STEP3_V3_ROOT = Path("tamu_data/processed/v3_step3_flat")

ALL_STEP_ROOTS = [STEP0_ROOT, STEP1_ROOT, STEP2_ROOT, STEP3_V3_ROOT]

# Matches _v001. in filenames (not directory names)
_VERSION_IN_FILENAME = re.compile(r"_v(\d{3})\.")


def resolve_version(force_new: bool = False) -> str:
    """Determine the current pipeline run version string (e.g. 'v001').

    Scans all step root directories for files containing _vNNN. in their name,
    takes the global maximum. If force_new=True, increments by 1.
    Returns 'v001' if no prior runs exist.
    """
    max_n = 0
    for root in ALL_STEP_ROOTS:
        if not root.exists():
            continue
        for f in root.iterdir():
            m = _VERSION_IN_FILENAME.search(f.name)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n

    if force_new or max_n == 0:
        return f"v{max_n + 1:03d}"
    return f"v{max_n:03d}"


def resolve_latest_file(step_root: Path, stem: str, suffix: str) -> "Path | None":
    """Find the highest-versioned file matching {stem}_vNNN{suffix} in step_root.

    Example: resolve_latest_file(STEP1_ROOT, "202611_CSCE_670_600_46627", ".md")
    returns the Path with the largest vNNN, or None if none found.
    """
    pattern = re.compile(
        rf"^{re.escape(stem)}_v(\d{{3}}){re.escape(suffix)}$"
    )
    best: "Path | None" = None
    best_n = -1
    for f in step_root.glob(f"{stem}_v*{suffix}"):
        m = pattern.match(f.name)
        if m:
            n = int(m.group(1))
            if n > best_n:
                best_n = n
                best = f
    return best


def write_run_meta(step_root: Path, version: str, cli_args: dict) -> None:
    """Write _run_meta_vNNN.json into a step root directory."""
    meta = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "cli_args": cli_args,
    }
    (step_root / f"_run_meta_{version}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class StepLogger:
    """Single combined JSONL + atomic CSV log for one pipeline step.

    All runs append to the same log files in the step root. The JSONL is
    source-of-truth; CSV is rebuilt from JSONL on every append using an
    atomic tmp→replace write (up to 5 retries on Excel PermissionError).

    Usage:
        logger = StepLogger(STEP1_ROOT / "step1_pdf_to_markdown_log")
        logger.log({"file": "..._v001.md", "version": "v001", ...})
    """

    def __init__(self, log_prefix: Path) -> None:
        """log_prefix: path without extension, e.g. STEP1_ROOT / 'step1_pdf_to_markdown_log'."""
        self._jsonl = log_prefix.with_suffix(".jsonl")
        self._csv = log_prefix.with_suffix(".csv")
        self._tmp = log_prefix.with_suffix(".csv.tmp")

    def log(self, row: dict) -> None:
        """Append row to JSONL, then atomically rewrite CSV."""
        with self._jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        rows = self._read_jsonl()

        # Collect ordered fieldnames (first row sets order, new keys appended)
        fieldnames: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)

        for attempt in range(5):
            try:
                with self._tmp.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=fieldnames, extrasaction="ignore", restval=""
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                self._tmp.replace(self._csv)
                return
            except PermissionError:
                if attempt == 4:
                    import warnings
                    warnings.warn(
                        f"CSV log locked after 5 retries, skipping CSV update: {self._csv}",
                        stacklevel=3,
                    )
                    return
                time.sleep(1.0)

    def _read_jsonl(self) -> list[dict]:
        rows: list = []
        if not self._jsonl.exists():
            return rows
        with self._jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows

    @staticmethod
    def hyperlink(path: Path, label: str) -> str:
        """Return an Excel HYPERLINK formula for a local file path."""
        abs_path = str(path.resolve()).replace("\\", "/")
        return f'=HYPERLINK("file:///{abs_path}", "{label}")'
