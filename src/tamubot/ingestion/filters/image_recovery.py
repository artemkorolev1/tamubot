"""Filter: recover content from ``<!-- image -->`` placeholders via Gemini multimodal.

Docling sometimes renders complex tables and figures as ``<!-- image -->``
markers.  This filter sends the original PDF + Docling markdown to Gemini 2.5
Flash, which returns corrected markdown with the placeholders replaced by
faithful content (tables as markdown tables, diagrams as brief descriptions,
logos removed).

Files with fewer than ``MIN_MARKERS`` image placeholders are passed through
unchanged (typically the single marker is just the university logo).

CLI: python -m tamubot.ingestion.filters.image_recovery input_dir/ output_dir/
"""

from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from tamubot.core import config
from tamubot.ingestion.filters.base import FilterResult

# ── Constants ────────────────────────────────────────────────────────────────

RAW_ROOT = Path("data/syllabi/raw")
IMAGE_MARKER = "<!-- image -->"
MIN_MARKERS = 2
MAX_OUTPUT_TOKENS = 16384
MODEL = "gemini-2.5-flash"

_VERSION_RE = re.compile(r"_v\d{3}$")

IMAGE_RECOVERY_PROMPT = """\
You are a university syllabus parser. Below is a markdown conversion of a PDF \
syllabus. The converter failed to extract some content (tables, figures, \
schedules) and left `<!-- image -->` placeholder markers in their place.

Your task:
1. Read the original PDF (attached) alongside the markdown below.
2. Identify every `<!-- image -->` marker in the markdown.
3. For each marker, find the corresponding content in the PDF and replace the \
marker with a faithful markdown rendering of that content.
   - Tables: render as standard markdown tables with | delimiters.
   - Figures/diagrams: describe the content in a brief paragraph.
   - Logos/decorative images: remove the marker entirely (leave nothing).
4. Preserve ALL other markdown content exactly as-is. Do not rewrite, \
reformat, or summarize existing text.
5. Output ONLY the complete corrected markdown — no commentary, no fences.

--- MARKDOWN START ---
{markdown}
--- MARKDOWN END ---
"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_raw_pdf(md_stem: str) -> Path | None:
    """Find the raw PDF corresponding to a bronze markdown file."""
    candidate = RAW_ROOT / f"{md_stem}.pdf"
    if candidate.exists():
        return candidate
    base_stem = _VERSION_RE.sub("", md_stem)
    candidate = RAW_ROOT / f"{base_stem}.pdf"
    if candidate.exists():
        return candidate
    return None


def _count_markers(text: str) -> int:
    return text.count(IMAGE_MARKER)


def _recover_images(md_text: str, pdf_path: Path) -> tuple[str, dict[str, Any]]:
    """Send PDF + markdown to Gemini and return corrected markdown."""
    from google.genai import types

    markers_before = _count_markers(md_text)
    pdf_bytes = pdf_path.read_bytes()
    prompt = IMAGE_RECOVERY_PROMPT.format(markdown=md_text)

    t0 = time.monotonic()
    client = config.get_genai_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=pdf_bytes,
                            mime_type="application/pdf",
                        )
                    ),
                    types.Part(text=prompt),
                ],
                role="user",
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    elapsed = time.monotonic() - t0

    result_text = response.text
    # Strip markdown fences if Gemini wrapped the output
    if result_text.startswith("```"):
        lines = result_text.splitlines()
        result_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    markers_after = _count_markers(result_text)

    # Safety: if output has MORE markers than input, Gemini hallucinated — discard
    if markers_after > markers_before:
        return md_text, {
            "recovered": False,
            "reason": "hallucinated_markers",
            "markers_before": markers_before,
            "markers_after": markers_after,
            "elapsed_s": round(elapsed, 1),
        }

    return result_text, {
        "recovered": True,
        "markers_before": markers_before,
        "markers_after": markers_after,
        "elapsed_s": round(elapsed, 1),
    }


# ── Filter class ─────────────────────────────────────────────────────────────


class ImageRecoveryFilter:
    """Recover content from ``<!-- image -->`` placeholders via Gemini multimodal."""

    name: str = "image_recovery"

    def apply(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any] | None = None,
    ) -> FilterResult:
        config = config or {}
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = FilterResult()
        md_files = sorted(input_dir.glob("*.md"))
        result.input_count = len(md_files)

        total_markers_before = 0
        total_markers_after = 0
        files_recovered = 0
        files_skipped = 0
        files_errored = 0

        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            marker_count = _count_markers(text)
            total_markers_before += marker_count

            if marker_count < MIN_MARKERS:
                # Pass-through
                shutil.copy2(md_path, output_dir / md_path.name)
                total_markers_after += marker_count
                files_skipped += 1
                result.log_entries.append(
                    {
                        "file": md_path.name,
                        "status": "skipped",
                        "markers_before": marker_count,
                        "markers_after": marker_count,
                    }
                )
                continue

            # Find corresponding raw PDF
            pdf_path = _find_raw_pdf(md_path.stem)
            if pdf_path is None:
                shutil.copy2(md_path, output_dir / md_path.name)
                total_markers_after += marker_count
                files_errored += 1
                print(f"    WARN: No raw PDF for {md_path.stem}, passing through")
                result.log_entries.append(
                    {
                        "file": md_path.name,
                        "status": "pdf_not_found",
                        "markers_before": marker_count,
                        "markers_after": marker_count,
                    }
                )
                continue

            # Attempt recovery via Gemini
            print(f"    Recovering {md_path.stem} ({marker_count} markers)...")
            try:
                recovered_text, info = _recover_images(text, pdf_path)
                markers_after = _count_markers(recovered_text)
                total_markers_after += markers_after

                (output_dir / md_path.name).write_text(recovered_text, encoding="utf-8")

                if info.get("recovered"):
                    files_recovered += 1
                    result.modified_count += 1
                    status = "recovered"
                else:
                    files_errored += 1
                    status = info.get("reason", "unknown")

                print(f"      {info['markers_before']} → {markers_after} markers  ({info['elapsed_s']:.1f}s)")
                result.log_entries.append(
                    {
                        "file": md_path.name,
                        "status": status,
                        "markers_before": info["markers_before"],
                        "markers_after": markers_after,
                        "elapsed_s": info["elapsed_s"],
                    }
                )

            except Exception as e:
                # Pass-through on failure
                shutil.copy2(md_path, output_dir / md_path.name)
                total_markers_after += marker_count
                files_errored += 1
                print(f"    ERROR: {md_path.stem}: {e}")
                result.log_entries.append(
                    {
                        "file": md_path.name,
                        "status": "error",
                        "markers_before": marker_count,
                        "markers_after": marker_count,
                        "error": str(e),
                    }
                )

        result.metrics = {
            "files_recovered": files_recovered,
            "files_skipped": files_skipped,
            "files_errored": files_errored,
            "markers_before": total_markers_before,
            "markers_after": total_markers_after,
            "markers_removed": total_markers_before - total_markers_after,
        }
        return result


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m tamubot.ingestion.filters.image_recovery INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    filt = ImageRecoveryFilter()
    res = filt.apply(in_dir, out_dir)

    print(f"\nFiles processed : {res.input_count}")
    print(f"Files recovered : {res.metrics['files_recovered']}")
    print(f"Files skipped   : {res.metrics['files_skipped']}")
    print(f"Files errored   : {res.metrics['files_errored']}")
    print(f"Markers before  : {res.metrics['markers_before']}")
    print(f"Markers after   : {res.metrics['markers_after']}")
    print(f"Markers removed : {res.metrics['markers_removed']}")

    for entry in res.log_entries:
        status = entry["status"].upper()
        before = entry["markers_before"]
        after = entry["markers_after"]
        extra = f"  ({entry['elapsed_s']:.1f}s)" if "elapsed_s" in entry else ""
        print(f"\n  {entry['file']} [{status}]  {before} → {after}{extra}")
