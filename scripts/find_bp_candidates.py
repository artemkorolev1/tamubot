"""Surface candidate boilerplate headers not yet in BOILERPLATE_REGISTRY.

Walks a silver dir (default: CSCE v5 03b_relocate_textbook), enumerates every
H1-H4 header across all files, normalizes (lowercase, strip trailing colon,
strip leading numeric prefix), and ranks headers by file-frequency.

A header is a candidate if it:
  - appears in >= MIN_FILES files,
  - does not match classify_header() (i.e. not already in the registry),
  - either contains a _BP_KEYWORDS token, or is reported in --include-all mode.

Usage:
    python scripts/find_bp_candidates.py
    python scripts/find_bp_candidates.py --silver path/to/dir --min-files 2 --include-all
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from tamubot.ingestion.boilerplate_stripper import classify_header
from tamubot.ingestion.filters.boilerplate import _BP_KEYWORDS

HEADER_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
DEFAULT_SILVER = Path("data/syllabi/CSCE/v5/silver/03b_relocate_textbook")


def normalize(h: str) -> str:
    return h.lower().strip().rstrip(":")


def header_has_bp_keyword(h: str) -> bool:
    low = h.lower()
    return any(kw in low for kw in _BP_KEYWORDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    ap.add_argument("--min-files", type=int, default=2)
    ap.add_argument(
        "--include-all",
        action="store_true",
        help="Include candidates that don't match _BP_KEYWORDS",
    )
    args = ap.parse_args()

    files = sorted(args.silver.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {args.silver}")

    occurrence: dict[str, set[str]] = defaultdict(set)
    level_seen: dict[str, Counter] = defaultdict(Counter)

    for fp in files:
        stem = fp.stem
        for line in fp.read_text(encoding="utf-8").splitlines():
            m = HEADER_RE.match(line)
            if not m:
                continue
            level = len(m.group(1))
            text = m.group(2).strip()
            if not text:
                continue
            norm = normalize(text)
            occurrence[norm].add(stem)
            level_seen[norm][level] += 1

    rows = []
    for norm, stems in occurrence.items():
        if len(stems) < args.min_files:
            continue
        if classify_header(norm) is not None:
            continue
        has_kw = header_has_bp_keyword(norm)
        if not args.include_all and not has_kw:
            continue
        # Pick most common level
        level = level_seen[norm].most_common(1)[0][0]
        rows.append((len(stems), level, norm, has_kw, sorted(stems)))

    rows.sort(key=lambda r: (-r[0], r[2]))

    # Split into two groups for readability
    kw_rows = [r for r in rows if r[3]]
    other_rows = [r for r in rows if not r[3]]

    print(f"Scanned {len(files)} files in {args.silver}")
    print(f"Min file occurrence: {args.min_files}")
    print(f"Total unique normalized headers: {len(occurrence)}")
    print(f"Already-in-registry skipped: {sum(1 for n in occurrence if classify_header(n))}")
    print()

    print("=" * 80)
    print("KEYWORD-MATCHED CANDIDATES (likely boilerplate)")
    print("=" * 80)
    print(f"{'#files':>6}  {'lvl':>3}  header")
    print("-" * 80)
    for n_files, level, norm, _kw, stems in kw_rows:
        print(f"{n_files:>6}  H{level}    {norm[:80]}")
        if n_files <= 6:
            for s in stems:
                print(f"          - {s}")

    if args.include_all:
        print()
        print("=" * 80)
        print("OTHER REPEATED HEADERS (likely course content — review carefully)")
        print("=" * 80)
        print(f"{'#files':>6}  {'lvl':>3}  header")
        print("-" * 80)
        for n_files, level, norm, _kw, _stems in other_rows[:60]:
            print(f"{n_files:>6}  H{level}    {norm[:80]}")


if __name__ == "__main__":
    main()
