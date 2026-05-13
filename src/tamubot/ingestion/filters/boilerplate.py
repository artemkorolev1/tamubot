"""Filter: strip boilerplate sections from markdown syllabi.

Removes entire sections (header + body until next header of same or higher
level) that match the boilerplate registry ported from
``tamubot.ingestion.boilerplate_stripper``.

CLI: python -m tamubot.ingestion.filters.boilerplate input_dir/ output_dir/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tamubot.ingestion.boilerplate_stripper import (
    BODY_BOILERPLATE_HEADERS,
    classify_header,
)
from tamubot.ingestion.filters.base import FilterResult

# ── Boilerplate-candidate keywords (ported from process_syllabi_v3) ──────────
_BP_KEYWORDS: frozenset[str] = frozenset(
    [
        "policy",
        "policies",
        "privacy",
        "ferpa",
        "ada",
        "disability",
        "wellness",
        "mental health",
        "nondiscrimination",
        "civil rights",
        "title ix",
        "copyright",
        "attendance",
        "makeup",
        "writing center",
        "technical support",
        "canvas",
        "perusall",
        "peerceptiv",
        "learning resources",
        "statement on",
        "notice of",
        "accommodation",
        "safety",
        "harassment",
        "discrimination",
        "evaluation",
        "accessibility",
        "honor",
        "pronouns",
    ]
)

_HEADER_RE = re.compile(r"^\s*(#{1,6})\s+(.+)$")

_BODY_BP_SET: frozenset[str] = frozenset(h.lower() for h in BODY_BOILERPLATE_HEADERS)


def _estimate_tokens(text: str) -> int:
    return max(0, round(len(text) / 4))


def _flag_new_candidate(header_text: str) -> bool:
    """Return True if *header_text* looks like a potential new boilerplate entry."""
    lower = header_text.lower()
    return any(kw in lower for kw in _BP_KEYWORDS)


# ── Section parser ───────────────────────────────────────────────────────────


def _parse_sections(md_text: str) -> list[tuple[str | None, int, list[str]]]:
    """Parse markdown into (header_text | None, level, body_lines) tuples.

    The first tuple has header=None / level=0 for any preamble before the
    first header.
    """
    sections: list[tuple[str | None, int, list[str]]] = []
    cur_header: str | None = None
    cur_level: int = 0
    cur_lines: list[str] = []

    for line in md_text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            sections.append((cur_header, cur_level, cur_lines))
            cur_header = m.group(2).strip()
            cur_level = len(m.group(1))
            cur_lines = []
        else:
            cur_lines.append(line)
    sections.append((cur_header, cur_level, cur_lines))
    return sections


class BoilerplateFilter:
    """Strip boilerplate sections from markdown syllabi."""

    name: str = "boilerplate"

    def apply(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any] | None = None,
        report_path: Path | None = None,
    ) -> FilterResult:
        config = config or {}
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if report_path is None:
            from tamubot.ingestion.report_writer import get_report

            report_path = get_report()

        result = FilterResult()
        md_files = sorted(input_dir.glob("*.md"))
        result.input_count = len(md_files)

        cat_totals: dict[str, int] = {}
        total_tokens_removed = 0
        all_new_candidates: list[str] = []

        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            sections = _parse_sections(text)

            kept: list[str] = []
            strip_log: list[dict[str, Any]] = []
            new_candidates: list[str] = []
            skip_until_level: int | None = None

            for header, level, body_lines in sections:
                # If we are inside a stripped parent section, skip children
                if skip_until_level is not None:
                    if header is not None and level <= skip_until_level:
                        skip_until_level = None  # we've exited the stripped section
                    else:
                        # Still inside stripped parent — accumulate into last log
                        if strip_log:
                            extra = ""
                            if header:
                                extra += f"{'#' * level} {header}\n"
                            extra += "\n".join(body_lines)
                            strip_log[-1]["chars"] += len(extra)
                            strip_log[-1]["content"] += "\n" + extra
                        continue

                bp_type = classify_header(header) if header else None

                # Also check body-level boilerplate headers
                if not bp_type and header:
                    norm = header.strip().lower().rstrip(":")
                    if norm in _BODY_BP_SET:
                        bp_type = "BODY_BOILERPLATE"

                content = "\n".join(body_lines)

                if bp_type:
                    full_text = ""
                    if header:
                        full_text += f"{'#' * level} {header}\n"
                    full_text += content
                    strip_log.append(
                        {
                            "header": header,
                            "type": bp_type,
                            "level": level,
                            "chars": len(full_text),
                            "content": full_text.strip(),
                        }
                    )
                    cat_totals[bp_type] = cat_totals.get(bp_type, 0) + 1
                    # DEPT_HEADER: only strip the header line, keep children
                    # Other categories: strip the entire section including children
                    if bp_type != "DEPT_HEADER":
                        skip_until_level = level
                else:
                    if header:
                        kept.append(f"{'#' * level} {header}")
                        # Flag new candidates
                        if _flag_new_candidate(header):
                            new_candidates.append(header)
                            all_new_candidates.append(header)
                    # Scan body lines for body-level boilerplate headers
                    # (plain text that matches _BODY_BP_SET, not preceded by #)
                    filtered_body: list[str] = []
                    skip_body = False
                    for bline in body_lines:
                        if skip_body:
                            if bline.strip() == "":
                                skip_body = False
                            continue
                        norm_bline = bline.strip().lower().rstrip(":")
                        if norm_bline and norm_bline in _BODY_BP_SET:
                            skip_body = True
                            strip_log.append(
                                {
                                    "header": bline.strip(),
                                    "type": "BODY_BOILERPLATE",
                                    "level": 0,
                                    "chars": len(bline),
                                    "content": bline.strip(),
                                }
                            )
                            cat_totals["BODY_BOILERPLATE"] = cat_totals.get("BODY_BOILERPLATE", 0) + 1
                            continue
                        filtered_body.append(bline)
                    content = "\n".join(filtered_body)
                    if content.strip():
                        kept.append(content)

            filtered_md = "\n\n".join(kept)
            (output_dir / md_path.name).write_text(filtered_md, encoding="utf-8")

            # Write sidecar listing stripped content
            if strip_log:
                sidecar_name = md_path.stem + "_stripped.txt"
                sidecar_lines = []
                for entry in strip_log:
                    sidecar_lines.append(
                        f"[{entry['type']}] {entry['header']} (level {entry['level']}, {entry['chars']} chars)"
                    )
                    if entry["content"]:
                        sidecar_lines.append(entry["content"])
                    sidecar_lines.append("")
                (output_dir / sidecar_name).write_text("\n".join(sidecar_lines), encoding="utf-8")

            tokens_removed = sum(_estimate_tokens(e["content"]) for e in strip_log)
            total_tokens_removed += tokens_removed

            if strip_log:
                result.modified_count += 1

            result.log_entries.append(
                {
                    "file": md_path.name,
                    "sections_stripped": len(strip_log),
                    "tokens_removed": tokens_removed,
                    "by_category": {},
                    "new_candidates": new_candidates,
                    "strip_details": [
                        {"header": e["header"], "type": e["type"], "chars": e["chars"]} for e in strip_log
                    ],
                }
            )
            # Tally per-file categories
            for e in strip_log:
                cats = result.log_entries[-1]["by_category"]
                cats[e["type"]] = cats.get(e["type"], 0) + 1

            if report_path:
                from tamubot.ingestion.report_writer import update_boilerplate

                tokens_in = _estimate_tokens(text)
                tokens_out = _estimate_tokens(filtered_md)
                update_boilerplate(
                    report_path,
                    md_path.stem,
                    len(strip_log),
                    tokens_in,
                    tokens_out,
                    [e["header"] for e in strip_log if e["header"]],
                )

        result.metrics = {
            "total_sections_stripped": sum(e["sections_stripped"] for e in result.log_entries),
            "total_tokens_removed": total_tokens_removed,
            "by_category": cat_totals,
            "new_candidates": all_new_candidates,
        }
        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tamubot.ingestion.filters.boilerplate INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    filt = BoilerplateFilter()
    res = filt.apply(in_dir, out_dir)

    print(f"Files processed      : {res.input_count}")
    print(f"Files modified       : {res.modified_count}")
    print(f"Sections stripped    : {res.metrics['total_sections_stripped']}")
    print(f"Tokens removed (~)   : {res.metrics['total_tokens_removed']}")

    if res.metrics["by_category"]:
        print("\nBy category:")
        for cat, count in sorted(res.metrics["by_category"].items()):
            print(f"  {cat:<20} {count:>4}")

    if res.metrics["new_candidates"]:
        print(f"\nNew boilerplate candidates ({len(res.metrics['new_candidates'])}):")
        for c in sorted(set(res.metrics["new_candidates"])):
            print(f"  - {c}")
