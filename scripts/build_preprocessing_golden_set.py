#!/usr/bin/env python
"""Assemble a golden set of N graduate-course syllabi for the v6b preprocessing pipeline.

Pools raw 2026 graduate PDFs scraped from both Simple Syllabus and Howdy Portal,
deduplicates to unique courses (the same course often appears in both portals), randomly
samples N of them with a fixed seed, and copies the chosen PDFs into a new directory with a
1..N sequence prefix plus a manifest (CSV + JSONL).

Cross-portal dedup: CRN is unique per section per term university-wide, so the dedup key is
(year, semester, CRN). Note the fall term-code mismatch — Simple Syllabus encodes Fall as
``...41`` while Howdy (registrar-authoritative) uses ``...31``; we normalise Fall to ``31``
so the two portals' copies of the same course collapse to one entry.

Usage:
    python scripts/build_preprocessing_golden_set.py --dry-run
    python scripts/build_preprocessing_golden_set.py --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# Scraper output roots (relative to repo root / CWD).
RAW_ROOTS = {
    "simple_syllabus": Path("tamu_data/raw/simple_syllabus"),
    "howdy_portal": Path("tamu_data/raw/howdy_portal"),
}

# A versioned, immutable evaluation dataset: PDFs land in <out>/raw/, manifests + data card
# at <out>/. Pipeline iterations live separately under .../eval_corpus/runs/round_NN/.
DEFAULT_OUT = Path("data/syllabi/eval_corpus/dataset/v1")

# stem: {term:6}_{SUBJECT}_{COURSE}_{SECTION}_{CRN} with an optional _HP suffix.
_STEM_RE = re.compile(
    r"^(?P<term>\d{6})_(?P<subject>[A-Z]+)_(?P<course>\d+)_(?P<section>[^_]+)_(?P<crn>\d+)(?:_HP)?$"
)

# Canonical (registrar) semester codes. Simple Syllabus uses 41 for Fall — normalise to 31.
_SEM_NORMALISE = {"11": "11", "21": "21", "31": "31", "41": "31"}
_SEM_LABEL = {"11": "Spring", "21": "Summer", "31": "Fall"}
TARGET_YEAR = "2026"


def parse_pdf(path: Path, source: str) -> dict | None:
    """Parse a syllabus PDF path into a normalised course record, or None if unparsable."""
    m = _STEM_RE.match(path.stem)
    if not m:
        return None
    g = m.groupdict()
    year, sem_raw = g["term"][:4], g["term"][4:]
    if year != TARGET_YEAR:
        return None
    sem = _SEM_NORMALISE.get(sem_raw)
    if sem is None:
        return None
    canonical_term = f"{year}{sem}"
    canonical_stem = f"{canonical_term}_{g['subject']}_{g['course']}_{g['section']}_{g['crn']}"
    return {
        "source": source,
        "dept": g["subject"],
        "course": g["course"],
        "section": g["section"],
        "crn": g["crn"],
        "term_code": canonical_term,
        "term_label": f"{_SEM_LABEL[sem]} {year}",
        "canonical_stem": canonical_stem,
        "dedup_key": (year, sem, g["crn"]),
        "original_path": str(path),
    }


def collect(depts: set[str] | None = None) -> dict[tuple, dict[str, dict]]:
    """Walk both raw roots; return {dedup_key: {source: record}}.

    If ``depts`` is given, keep only records whose department is in that set.
    """
    by_key: dict[tuple, dict[str, dict]] = {}
    skipped = 0
    for source, root in RAW_ROOTS.items():
        if not root.exists():
            print(f"  [warn] root not found: {root}")
            continue
        pdfs = sorted(root.glob("*/graduate/*_2026/*.pdf"))
        parsed_n = 0
        for pdf in pdfs:
            rec = parse_pdf(pdf, source)
            if rec is None:
                skipped += 1
                continue
            if depts and rec["dept"] not in depts:
                continue
            # First-seen wins within a source (a course rarely repeats per source).
            by_key.setdefault(rec["dedup_key"], {}).setdefault(source, rec)
            parsed_n += 1
        print(f"  {source}: {len(pdfs)} pdfs found, {parsed_n} parsed as 2026 graduate"
              + (f" in {sorted(depts)}" if depts else ""))
    if skipped:
        print(f"  [info] {skipped} files skipped (unparsable stem or non-2026)")
    return by_key


def choose_balanced(keys: list[tuple], by_key: dict[tuple, dict[str, dict]]) -> list[dict]:
    """For each selected key, pick a source, balancing the two portals when both have it."""
    counts: Counter[str] = Counter()
    chosen: list[dict] = []
    for key in keys:
        sources = by_key[key]
        if len(sources) == 1:
            src = next(iter(sources))
        else:
            # Both portals have this course — assign to the currently under-represented one
            # (deterministic tie-break: simple_syllabus first).
            src = min(sources, key=lambda s: (counts[s], s))
        counts[src] += 1
        chosen.append(sources[src])
    return chosen


def choose_preferred(keys: list[tuple], by_key: dict[tuple, dict[str, dict]], prefer: str) -> list[dict]:
    """For each key, use the preferred portal's copy; fall back to the other if absent."""
    chosen: list[dict] = []
    for key in keys:
        sources = by_key[key]
        src = prefer if prefer in sources else next(iter(sources))
        chosen.append(sources[src])
    return chosen


def write_dataset_card(out: Path, rows: list[dict], args) -> None:
    """Write a dataset_card.md documenting provenance + composition (auto-generated)."""
    by_source = Counter(r["source_portal"] for r in rows)
    by_term = Counter(r["term_label"] for r in rows)
    by_dept = Counter(r["dept"] for r in rows)
    depts = args.depts or "all"
    lines = [
        "# Evaluation dataset — TAMU graduate syllabi (preprocessing)",
        "",
        f"- **Version:** `{out.name}`",
        f"- **Built:** {date.today().isoformat()} (auto-generated by `scripts/build_preprocessing_golden_set.py`)",
        f"- **Size:** {len(rows)} syllabi (1 PDF per unique course)",
        f"- **Departments:** {depts}",
        f"- **Source preference:** `{args.prefer}` (fall back to the other portal when absent)",
        f"- **Sampling:** random, seed `{args.seed}` (reproducible)",
        "",
        "## Purpose",
        "Curated input corpus for developing and evaluating the **v6b** preprocessing/ingestion "
        "pipeline (PDF → structured text → chunks). Immutable: treat each `dataset/vN` as a frozen "
        "snapshot; run pipeline iterations under `../../runs/round_NN/`.",
        "",
        "## Provenance",
        "Scraped from Simple Syllabus (`tamu.simplesyllabus.com`) and Howdy Portal "
        "(`howdyportal.tamu.edu`), College Station, graduate courses (course number ≥ 600), "
        "Spring/Summer/Fall 2026. Deduplicated across portals by `(year, semester, CRN)`; the Fall "
        "term code is normalised to the registrar value (`31`).",
        "",
        "## Composition",
        f"- **By source:** " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())),
        f"- **By term:** " + ", ".join(f"{k}={v}" for k, v in sorted(by_term.items())),
        f"- **By department:** " + ", ".join(f"{k}={v}" for k, v in by_dept.most_common()),
        "",
        "## Layout",
        "- `raw/` — source PDFs, named `NNN_<termcode>_<DEPT>_<COURSE>_<SECTION>_<CRN>.pdf`",
        "- `manifest.csv` / `manifest.jsonl` — one row per syllabus (seq, source, course, CRN, term, original path)",
        "",
        "## Reproduce",
        "```bash",
        f"python scripts/build_preprocessing_golden_set.py --n {args.n} --seed {args.seed} "
        + (f"--depts {args.depts} " if args.depts else "")
        + f"--prefer {args.prefer}",
        "```",
        "",
    ]
    (out / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def print_breakdown(records: list[dict]) -> None:
    by_source = Counter(r["source"] for r in records)
    by_term = Counter(r["term_label"] for r in records)
    by_dept = Counter(r["dept"] for r in records)
    print(f"\nSelected {len(records)} courses")
    print("  by source:", dict(by_source))
    print("  by term:  ", dict(sorted(by_term.items())))
    print(f"  by dept:   {len(by_dept)} departments")
    for dept, n in by_dept.most_common():
        print(f"    {dept}: {n}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of courses to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--depts", default="", help="comma-separated dept filter, e.g. CSCE,ISEN,STAT,ECEN (default: all)")
    ap.add_argument(
        "--prefer",
        default="balance",
        choices=["balance", "simple_syllabus", "howdy_portal"],
        help="source preference when a course exists in both portals; 'balance' evens the two out, "
        "otherwise use the named portal and fall back to the other",
    )
    ap.add_argument("--dry-run", action="store_true", help="report the plan without copying")
    args = ap.parse_args()

    depts = {d.strip().upper() for d in args.depts.split(",") if d.strip()} or None

    print("Collecting raw 2026 graduate syllabi from both portals…")
    by_key = collect(depts)
    unique_keys = list(by_key.keys())
    print(f"\nPooled {len(unique_keys)} unique courses (dedup key = year+semester+CRN).")

    if not unique_keys:
        print("ERROR: no syllabi found. Run the scrapers first.", file=sys.stderr)
        return 1

    n = min(args.n, len(unique_keys))
    if n < args.n:
        print(f"WARNING: only {len(unique_keys)} unique courses available — sampling all {n}.")

    rng = random.Random(args.seed)
    sampled_keys = rng.sample(unique_keys, n)
    # Stable, reproducible ordering for sequence numbering.
    sampled_keys.sort(key=lambda k: by_key[k][next(iter(by_key[k]))]["canonical_stem"])
    if args.prefer == "balance":
        records = choose_balanced(sampled_keys, by_key)
    else:
        records = choose_preferred(sampled_keys, by_key, args.prefer)

    print_breakdown(records)

    if args.dry_run:
        print("\n[dry-run] No files copied. Planned golden filenames:")
        for i, r in enumerate(records, 1):
            print(f"  {i:03d}_{r['canonical_stem']}.pdf  <-  {r['original_path']}")
        return 0

    out: Path = args.out
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Remove a prior build in this dataset version so a re-run with a different composition
    # doesn't leave orphaned files (only touches our own NNN_*.pdf + manifests + card).
    stale = list(raw_dir.glob("[0-9][0-9][0-9]_*.pdf"))
    stale += [out / "manifest.csv", out / "manifest.jsonl", out / "dataset_card.md"]
    for old in stale:
        if old.exists():
            old.unlink()

    manifest_rows: list[dict] = []
    print(f"\nCopying {len(records)} PDFs into {raw_dir} …")
    for i, r in enumerate(records, 1):
        filename = f"{i:03d}_{r['canonical_stem']}.pdf"
        shutil.copy2(r["original_path"], raw_dir / filename)
        manifest_rows.append(
            {
                "seq": f"{i:03d}",
                "source_portal": r["source"],
                "dept": r["dept"],
                "course": r["course"],
                "section": r["section"],
                "crn": r["crn"],
                "term_code": r["term_code"],
                "term_label": r["term_label"],
                "canonical_stem": r["canonical_stem"],
                "filename": filename,
                "original_path": r["original_path"],
            }
        )
        if i % 10 == 0 or i == len(records):
            print(f"  copied {i}/{len(records)}")

    fieldnames = list(manifest_rows[0].keys())
    csv_path = out / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    jsonl_path = out / "manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    write_dataset_card(out, manifest_rows, args)

    print(f"\nDone. {len(manifest_rows)} syllabi in {out}")
    print(f"  files:    {raw_dir}/ ({len(manifest_rows)} PDFs)")
    print(f"  manifest: {csv_path}")
    print(f"  manifest: {jsonl_path}")
    print(f"  card:     {out / 'dataset_card.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
