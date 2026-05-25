#!/usr/bin/env python3
"""For each Gemini-called file, show bronze marker context and silver replacement."""

from __future__ import annotations

import re
import sys
from pathlib import Path

BRONZE = Path("data/syllabi/CSCE/bronze")
SILVER = Path("data/syllabi/CSCE/silver/01_image_recovery")

FILES = [
    "202611_CSCE_612_600_42640_v012",
    "202611_CSCE_614_600_42743_v012",
    "202611_CSCE_614_700_54779_v012",
    "202611_CSCE_625_601_58601_HP_v012",
    "202611_CSCE_629_600_54978_v012",
]

HDR_RE = re.compile(r"^#{1,6}\s+")


def nearest_header_above(lines: list[str], idx: int) -> tuple[int, str] | None:
    for j in range(idx - 1, -1, -1):
        if HDR_RE.match(lines[j].strip()):
            return j, lines[j].strip()
    return None


def find_section_in_silver(silver: list[str], header: str) -> tuple[int, int]:
    """Return (start, end) line indices of the section under `header` in silver."""
    start = None
    for i, ln in enumerate(silver):
        if ln.strip() == header:
            start = i
            break
    if start is None:
        return (-1, -1)
    end = len(silver)
    for j in range(start + 1, len(silver)):
        if HDR_RE.match(silver[j].strip()):
            end = j
            break
    return (start, end)


def main() -> None:
    for stem in FILES:
        bronze_path = BRONZE / f"{stem}.md"
        silver_path = SILVER / f"{stem}.md"
        if not bronze_path.exists() or not silver_path.exists():
            print(f"!! missing: {stem}")
            continue
        bronze = bronze_path.read_text(encoding="utf-8").splitlines()
        silver = silver_path.read_text(encoding="utf-8").splitlines()

        marker_lines = [i for i, ln in enumerate(bronze) if ln.strip() == "<!-- image -->"]
        print()
        print("=" * 80)
        print(f"FILE: {stem}    bronze markers: {len(marker_lines)}    bronze chars: {len(bronze_path.read_text(encoding='utf-8'))}    silver chars: {len(silver_path.read_text(encoding='utf-8'))}")
        print("=" * 80)
        for m in marker_lines:
            hdr_info = nearest_header_above(bronze, m)
            hdr_text = hdr_info[1] if hdr_info else "(no header above)"
            print()
            print(f"  ── BRONZE marker @ line {m + 1}  under: {hdr_text!r}")
            ctx_lo = max(0, m - 3)
            ctx_hi = min(len(bronze), m + 4)
            for k in range(ctx_lo, ctx_hi):
                arrow = ">> " if k == m else "   "
                print(f"     {arrow}{bronze[k][:120]}")

            print(f"  ── SILVER section under {hdr_text!r}:")
            if hdr_info:
                s_start, s_end = find_section_in_silver(silver, hdr_text)
                if s_start == -1:
                    print("       (header not found in silver)")
                else:
                    section = silver[s_start : min(s_end, s_start + 20)]
                    for ln in section:
                        print(f"       {ln[:120]}")
                    if s_end - s_start > 20:
                        print(f"       ... ({s_end - s_start - 20} more lines)")
            else:
                print(f"       {silver[0][:120]} ...")


if __name__ == "__main__":
    main()
