"""Flatten raw syllabus PDFs into .streamlit/static/syllabi/ via symlinks.

Streamlit only serves files under .streamlit/static/ (with enableStaticServing).
The raw PDFs live in deep trees (raw/simple_syllabus/<DEPT>/<level>/<term>/*.pdf
and raw/howdy_portal/...). This script symlinks each PDF flat by its stem so
Streamlit can serve it at /app/static/syllabi/<stem>.pdf, which is the URL
form the chatbot uses for clickable [Source N, p.X] citations.

Run once after ingestion (idempotent — re-running replaces stale symlinks).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOTS = [
    REPO_ROOT / "tamu_data" / "raw" / "simple_syllabus",
    REPO_ROOT / "tamu_data" / "raw" / "howdy_portal",
]
STATIC_DIR = REPO_ROOT / ".streamlit" / "static" / "syllabi"


def sync() -> tuple[int, int]:
    """Create one symlink per PDF stem. Returns (created, skipped)."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    seen: dict[str, Path] = {}
    for root in RAW_ROOTS:
        if not root.exists():
            continue
        for pdf in root.rglob("*.pdf"):
            stem = pdf.stem
            target = STATIC_DIR / f"{stem}.pdf"
            if stem in seen:
                # Same stem from a different source — Simple Syllabus wins
                # (preferred per project memory). Already linked, skip.
                skipped += 1
                continue
            seen[stem] = pdf
            if target.is_symlink() or target.exists():
                if target.is_symlink() and target.readlink() == pdf:
                    skipped += 1
                    continue
                target.unlink()
            target.symlink_to(pdf)
            created += 1
    return created, skipped


if __name__ == "__main__":
    c, s = sync()
    print(f"sync_static_pdfs: {c} symlinks created, {s} unchanged")
