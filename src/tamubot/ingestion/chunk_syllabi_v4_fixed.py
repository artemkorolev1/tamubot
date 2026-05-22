"""
tamubot.ingestion.chunk_syllabi_v4_fixed

Fixed-size re-chunker for v4 Docling pipeline output.

Reads cleaned hierarchical markdown from silver/04_hierarchy/, pulls metadata
from silver/05_enrich/, and writes JSON in the v4 chunk shape (compatible with
`ingest.py --v4`) into silver/06_chunk_fixed_<size>t_<overlap>o/.

No LLM calls. Used to compare fixed-size vs semantic chunking on the same corpus.

Usage:
    python -m tamubot.ingestion.chunk_syllabi_v4_fixed --chunk-size 600 --overlap 100 --department ISEN --files 5
    python -m tamubot.ingestion.chunk_syllabi_v4_fixed --chunk-size 600 --overlap 100 --department ISEN --all
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tamubot.ingestion.chunker_v3 import _tokens_approx

HIERARCHY_DIR = Path("data/syllabi/silver/04_hierarchy")
ENRICH_DIR = Path("data/syllabi/silver/05_enrich")
OUTPUT_BASE = Path("data/syllabi/silver")

CHARS_PER_TOKEN = 4  # matches chunker_v3._tokens_approx


def _has_table(content: str) -> bool:
    return bool(re.search(r"\|.*\|", content))


def fixed_size_chunks(text: str, chunk_size_tok: int, overlap_tok: int, max_tok: int) -> list[str]:
    """Sliding-window fixed-size splitter.

    Each chunk targets `chunk_size_tok` tokens and is capped at `max_tok`. The
    cut point is the earliest whitespace boundary (paragraph > newline > space)
    in the slack window [target, max]. If no whitespace exists in that window
    (unbroken blob), hard-cut at max. Next chunk starts `overlap_tok` tokens
    before the prior cut.
    """
    text = text.strip()
    if not text:
        return []

    target_chars = chunk_size_tok * CHARS_PER_TOKEN
    overlap_chars = overlap_tok * CHARS_PER_TOKEN
    max_chars = max_tok * CHARS_PER_TOKEN

    chunks: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        end_target = pos + target_chars
        if end_target >= n:
            chunks.append(text[pos:n].strip())
            break

        window_end = min(pos + max_chars, n)
        end = -1
        for sep in ("\n\n", "\n", " "):
            cand = text.find(sep, end_target, window_end)
            if cand >= 0:
                end = cand + len(sep)
                break
        if end < 0:
            end = window_end  # unbroken blob — hard cut

        chunks.append(text[pos:end].strip())
        next_pos = max(end - overlap_chars, pos + 1)
        if next_pos >= n:
            break
        pos = next_pos

    return [c for c in chunks if c]


def chunk_file(md_path: Path, enrich_path: Path, chunk_size: int, overlap: int, max_tokens: int) -> dict:
    text = md_path.read_text(encoding="utf-8")
    enrich = json.loads(enrich_path.read_text(encoding="utf-8"))

    raw_chunks = fixed_size_chunks(text, chunk_size, overlap, max_tokens)
    chunks = []
    for i, content in enumerate(raw_chunks):
        chunks.append(
            {
                "chunk_index": i,
                "content": content,
                "header_path": None,
                "token_count": _tokens_approx(content),
                "has_table": _has_table(content),
                "flags": [],
                "split_reason": "fixed",
                "page": None,
            }
        )

    return {
        "source_file": enrich.get("source_file", md_path.stem),
        "pipeline_version": "v4",
        "source": enrich.get("source"),
        "course_type": enrich.get("course_type"),
        "course_metadata": enrich.get("course_metadata", {}),
        "course_summary": enrich.get("course_summary"),
        "chunk_config": {
            "strategy": "fixed",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "max_tokens": max_tokens,
        },
        "total_chunks": len(chunks),
        "chunks": chunks,
        "_parsed_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed-size re-chunker for v4 pipeline output.")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Hard cap per chunk in approximate tokens. Splitter prefers whitespace within "
        "[chunk_size, max_tokens]; hard-cuts if none. Default: 900 (1.5x chunk_size).",
    )
    parser.add_argument(
        "--department", type=str, default="ISEN", help="Filter by department code in filename (default: ISEN)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--files", type=int, default=5, metavar="N", help="Process first N files (alphabetical). Default: 5."
    )
    group.add_argument("--all", action="store_true", help="Process all matching files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    dept_tag = f"_{args.department.upper()}_"
    sources = sorted(f for f in HIERARCHY_DIR.glob("*.md") if dept_tag in f.name)
    if not sources:
        print(f"No {args.department} files found in {HIERARCHY_DIR}", file=sys.stderr)
        sys.exit(1)

    if not args.all:
        sources = sources[: args.files]

    out_dir = OUTPUT_BASE / f"06_chunk_fixed_{args.chunk_size}t_{args.overlap}o"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = errors = 0
    for src in sources:
        enrich_path = ENRICH_DIR / f"{src.stem}.json"
        out_path = out_dir / f"{src.stem}.json"

        if not enrich_path.exists():
            print(f"  SKIP {src.stem}: missing enrich JSON", file=sys.stderr)
            errors += 1
            continue
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            result = chunk_file(src, enrich_path, args.chunk_size, args.overlap, args.max_tokens)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            token_counts = [c["token_count"] for c in result["chunks"]]
            avg = round(sum(token_counts) / len(token_counts)) if token_counts else 0
            print(f"  {src.stem}: {result['total_chunks']} chunks, avg {avg} tok")
            processed += 1
        except Exception as exc:
            print(f"  ERROR {src.stem}: {exc}", file=sys.stderr)
            errors += 1

    print(f"\nDone — {processed} processed, {skipped} skipped, {errors} errors → {out_dir}")


if __name__ == "__main__":
    main()
