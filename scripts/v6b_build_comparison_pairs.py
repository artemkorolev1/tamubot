#!/usr/bin/env python
"""Build an Inspect-ready comparison set for one preprocessing-improvement iteration.

For each stem it writes two aligned views plus the metadata the judge needs:

    <iter_dir>/original/NNN_<stem>.md      faithful extracted text (v6b bronze md)
    <iter_dir>/processed/NNN_<stem>.md     RAG-visible text (03_tag chunks minus
                                           is_boilerplate / is_duplicate)
    <iter_dir>/processed/NNN_<stem>.why.json   per-dropped-chunk exclusion provenance
    <iter_dir>/pdf_pages/<stem>/p###.png   pre-rendered source PDF pages (fidelity ref)
    <iter_dir>/samples.jsonl               lean Inspect dataset (paths + check anchors)
    <iter_dir>/manifest.jsonl              run provenance (sha, judge model, seed, ...)
    <iter_dir>/compare.html                browser side-by-side review (dropped chunks
                                           highlighted with reasons) + index.html

The judge Task (src/tamubot/evals/preprocessing_judge/) reads samples.jsonl, assembles
the multimodal prompt from these files + page PNGs, and grades against
docs/preprocessing_error_taxonomy.md. This builder does no LLM work and is deterministic.

Run inside the tamubot-dev-1 container (needs PyMuPDF + the tamubot package):

    docker exec tamubot-dev-1 python scripts/v6b_build_comparison_pairs.py \
        --from-disk --limit 5
    docker exec tamubot-dev-1 python scripts/v6b_build_comparison_pairs.py \
        --manifest data/syllabi/_golden/preprocessing_v1/manifest.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT, dept_from_stem
from tamubot.ingestion.pipeline_v6b import paths

TAXONOMY_VERSION = "1.0.0"  # must match docs/preprocessing_error_taxonomy.md
LAB_ROOT = DATA_ROOT / "_preprocessing_lab"
DEFAULT_JUDGE_MODEL = os.getenv("PREPROC_JUDGE_MODEL", "google/gemini-2.5-flash")
OVERSIZED_TOKENS = 2048  # mirrors silver_chunk_checks no_oversized
MAX_PDF_PAGES = int(os.getenv("PREPROC_MAX_PDF_PAGES", "12"))  # fidelity render cap
PAGE_DPI = int(os.getenv("PREPROC_PAGE_DPI", "110"))


# --------------------------------------------------------------------------- io
def load_chunks(stem: str) -> list[dict]:
    """Return the ordered chunk list from a 03_tag file (dict-with-chunks or list)."""
    data = json.loads(paths.silver_tag_path(stem).read_text(encoding="utf-8"))
    chunks = data.get("chunks", data) if isinstance(data, dict) else data
    if not isinstance(chunks, list):
        raise ValueError(f"{stem}: unexpected 03_tag shape {type(chunks).__name__}")
    return sorted(chunks, key=lambda c: c.get("chunk_index", 0))


def drop_reason(c: dict) -> str | None:
    """Why this chunk is hidden from RAG, or None if retained."""
    if c.get("is_boilerplate"):
        return "boilerplate"
    if c.get("is_duplicate"):
        return "duplicate"
    return None


def check_anchors(chunks: list[dict]) -> dict:
    """Deterministic context the judge defers to (mirrors the v6b checks)."""
    n = len(chunks) or 1
    bp = sum(1 for c in chunks if c.get("is_boilerplate"))
    dup = sum(1 for c in chunks if c.get("is_duplicate"))
    no_header = sum(1 for c in chunks if not c.get("header_path"))
    oversized = sum(1 for c in chunks if (c.get("token_count") or 0) > OVERSIZED_TOKENS)
    tokens = [c.get("token_count") or 0 for c in chunks]
    return {
        "chunk_count": len(chunks),
        "retained_count": n - bp - dup,
        "boilerplate_count": bp,
        "boilerplate_rate": round(bp / n, 3),
        "duplicate_count": dup,
        "duplicate_rate": round(dup / n, 3),
        "no_header_count": no_header,
        "no_header_rate": round(no_header / n, 3),
        "oversized_count": oversized,
        "max_token_count": max(tokens, default=0),
        "has_table_count": sum(1 for c in chunks if c.get("has_table")),
    }


def render_pdf_pages(stem: str, out_dir: Path) -> list[str]:
    """Render up to MAX_PDF_PAGES source-PDF pages to PNG; return relative paths."""
    pdf = paths.raw_path(stem)
    if not pdf.exists():
        return []
    import fitz  # PyMuPDF — present in container

    out_dir.mkdir(parents=True, exist_ok=True)
    rels: list[str] = []
    zoom = PAGE_DPI / 72.0
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                break
            png = out_dir / f"p{i + 1:03d}.png"
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(png)
            rels.append(str(png))
    return rels


# ---------------------------------------------------------------- view builders
def build_processed_md(chunks: list[dict]) -> str:
    """Concatenate retained chunks with light boundary markers."""
    parts: list[str] = []
    for c in chunks:
        if drop_reason(c):
            continue
        idx, hp = c.get("chunk_index"), c.get("header_path") or ""
        parts.append(f"<!-- chunk {idx} | {hp} | {c.get('token_count')} tok -->")
        parts.append((c.get("content") or "").strip())
    return "\n\n".join(parts) + "\n"


def build_why(chunks: list[dict]) -> list[dict]:
    """Provenance for every chunk hidden from RAG."""
    why: list[dict] = []
    for c in chunks:
        reason = drop_reason(c)
        if not reason:
            continue
        content = (c.get("content") or "").strip()
        why.append({
            "chunk_index": c.get("chunk_index"),
            "reason": reason,
            "header_path": c.get("header_path"),
            "token_count": c.get("token_count"),
            "boilerplate_cluster": c.get("boilerplate_cluster"),
            "cluster_confidence": c.get("cluster_confidence"),
            "duplicate_of_chunk_id": c.get("duplicate_of_chunk_id"),
            "content_preview": content[:280],
        })
    return why


def chunk_html(c: dict) -> str:
    reason = drop_reason(c)
    cls = {"boilerplate": "drop-bp", "duplicate": "drop-dup", None: "keep"}[reason]
    idx, hp = c.get("chunk_index"), html.escape(c.get("header_path") or "—")
    body = html.escape((c.get("content") or "").strip())
    tag = ""
    if reason == "boilerplate":
        tag = (f'<span class="badge bp">BOILERPLATE · {html.escape(str(c.get("boilerplate_cluster")))[:18]}'
               f' · conf {c.get("cluster_confidence")}</span>')
    elif reason == "duplicate":
        tag = (f'<span class="badge dup">DUPLICATE · canon '
               f'{html.escape(str(c.get("duplicate_of_chunk_id")))}</span>')
    return (f'<div class="chunk {cls}"><div class="meta">chunk {idx} · {hp} · '
            f'{c.get("token_count")} tok {tag}</div><pre>{body}</pre></div>')


def build_compare_html(stem: str, seq: str, chunks: list[dict], page_rels: list[str],
                       anchors: dict, iter_dir: Path) -> str:
    pages = "".join(
        f'<img src="{html.escape(os.path.relpath(p, iter_dir))}" loading="lazy">'
        for p in page_rels
    )
    chunks_html = "".join(chunk_html(c) for c in chunks)
    anchor_html = " · ".join(f"{k}={v}" for k, v in anchors.items())
    return f"""<!doctype html><meta charset="utf-8">
<title>{seq} {stem}</title>
<style>
 body{{font:13px/1.5 system-ui,sans-serif;margin:0;background:#faf9f7;color:#222}}
 header{{background:#500000;color:#fff;padding:10px 16px;position:sticky;top:0;z-index:9}}
 header a{{color:#ffd7d7}} .wrap{{display:grid;grid-template-columns:1fr 360px;gap:0}}
 .col{{padding:14px;overflow:auto;height:calc(100vh - 46px)}}
 .anchors{{font:11px monospace;color:#666;padding:6px 16px;background:#f0ede9}}
 .chunk{{border-left:4px solid #ccc;margin:8px 0;padding:6px 10px;background:#fff;border-radius:4px}}
 .chunk pre{{white-space:pre-wrap;margin:4px 0 0;font:12px/1.45 ui-monospace,monospace}}
 .meta{{font:11px monospace;color:#777}}
 .keep{{border-left-color:#2e7d32}}
 .drop-bp{{border-left-color:#c62828;background:#fff4f4;opacity:.85}}
 .drop-dup{{border-left-color:#e08600;background:#fff9ef;opacity:.85}}
 .badge{{font:10px/1 sans-serif;padding:2px 5px;border-radius:3px;color:#fff;margin-left:6px}}
 .badge.bp{{background:#c62828}} .badge.dup{{background:#e08600}}
 .pages img{{width:100%;border:1px solid #ddd;margin-bottom:8px}}
 h2{{margin:.2em 0;font-size:13px;color:#500000}}
</style>
<header><b>{seq} · {html.escape(stem)}</b> &nbsp; <a href="index.html">&larr; all</a>
 &nbsp; legend: <span style="color:#9fd49f">retained</span> /
 <span style="color:#ff9e9e">boilerplate</span> / <span style="color:#ffce86">duplicate</span></header>
<div class="anchors">deterministic anchors &nbsp; {html.escape(anchor_html)}</div>
<div class="wrap">
 <div class="col"><h2>Tagged chunk sequence (dropped = hidden from RAG)</h2>{chunks_html}</div>
 <div class="col pages"><h2>Source PDF (fidelity reference)</h2>{pages or '<i>no PDF</i>'}</div>
</div>"""


def build_index_html(rows: list[dict]) -> str:
    trs = "".join(
        f'<tr><td><a href="{r["seq"]}_{html.escape(r["stem"])}.html">{r["seq"]}</a></td>'
        f'<td>{html.escape(r["stem"])}</td><td>{r["anchors"]["chunk_count"]}</td>'
        f'<td>{r["anchors"]["boilerplate_count"]}</td><td>{r["anchors"]["duplicate_count"]}</td>'
        f'<td>{r["anchors"]["oversized_count"]}</td></tr>'
        for r in rows
    )
    return f"""<!doctype html><meta charset="utf-8"><title>preprocessing review</title>
<style>body{{font:13px system-ui;margin:24px}}table{{border-collapse:collapse}}
 td,th{{border:1px solid #ddd;padding:4px 10px;text-align:left}}th{{background:#500000;color:#fff}}</style>
<h1>Preprocessing comparison — {len(rows)} syllabi</h1>
<table><tr><th>#</th><th>stem</th><th>chunks</th><th>bp</th><th>dup</th><th>oversized</th></tr>
{trs}</table>"""


# ---------------------------------------------------------------------- driver
def resolve_stems(args) -> list[str]:
    if args.stems:
        return [s.strip() for s in args.stems.split(",") if s.strip()]
    if args.manifest:
        stems: list[str] = []
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stems.append(rec["stem"] if isinstance(rec, dict) else str(rec))
        return stems
    if args.from_disk:
        found = {p.stem for p in DATA_ROOT.glob("*/v6b/silver/03_tag/*.json")}
        return sorted(found)
    raise SystemExit("provide --stems, --manifest, or --from-disk")


def git_sha() -> str:
    # -c safe.directory=* avoids "dubious ownership" when run inside the container
    # against the host-mounted /workspace checkout.
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def next_iter_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    sha = git_sha()
    n = 1 + sum(1 for p in LAB_ROOT.glob("iter_*") if p.is_dir()) if LAB_ROOT.exists() else 1
    return LAB_ROOT / f"iter_{n:02d}_{sha}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--stems", help="comma-separated stems")
    src.add_argument("--manifest", help="jsonl/txt with one stem per line or {'stem':..}")
    src.add_argument("--from-disk", action="store_true", help="all materialized 03_tag stems")
    ap.add_argument("--iter-dir", help="output dir (default: auto iter_NN_<sha>)")
    ap.add_argument("--split", default="dev", help="dev|locked|anchor (recorded in manifest)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap number of stems (testing)")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF page rendering")
    args = ap.parse_args()

    stems = resolve_stems(args)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit("no stems resolved")

    # absolute so paths in samples.jsonl survive Inspect's task-loader cwd changes
    iter_dir = next_iter_dir(args.iter_dir).resolve()
    (iter_dir / "original").mkdir(parents=True, exist_ok=True)
    (iter_dir / "processed").mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    index_rows: list[dict] = []
    width = max(3, len(str(len(stems))))

    for i, stem in enumerate(sorted(stems), 1):
        seq = str(i).zfill(width)
        try:
            chunks = load_chunks(stem)
        except FileNotFoundError:
            print(f"  [skip] {stem}: no 03_tag output")
            continue
        bronze = paths.bronze_md_path(stem)
        if not bronze.exists():
            print(f"  [skip] {stem}: no bronze md")
            continue

        anchors = check_anchors(chunks)
        (iter_dir / "original" / f"{seq}_{stem}.md").write_text(
            bronze.read_text(encoding="utf-8"), encoding="utf-8")
        (iter_dir / "processed" / f"{seq}_{stem}.md").write_text(
            build_processed_md(chunks), encoding="utf-8")
        (iter_dir / "processed" / f"{seq}_{stem}.why.json").write_text(
            json.dumps(build_why(chunks), indent=2), encoding="utf-8")

        page_rels = [] if args.no_pdf else render_pdf_pages(
            stem, iter_dir / "pdf_pages" / stem)

        (iter_dir / f"{seq}_{stem}.html").write_text(
            build_compare_html(stem, seq, chunks, page_rels, anchors, iter_dir),
            encoding="utf-8")
        index_rows.append({"seq": seq, "stem": stem, "anchors": anchors})

        # store repo-root-relative paths so both the container Inspect run and host-side
        # Claude Code sub-agents can resolve them (against their own cwd / PREPROC_ROOT)
        def rel(p) -> str:
            return os.path.relpath(str(p), str(Path.cwd()))

        samples.append({
            "id": stem,
            "input": f"Judge preprocessing for syllabus {stem} (dept {dept_from_stem(stem)}).",
            "target": "",  # populated only for the human-anchor split
            "metadata": {
                "stem": stem,
                "seq": seq,
                "dept": dept_from_stem(stem),
                "split": args.split,
                "taxonomy_version": TAXONOMY_VERSION,
                "original_md": rel(iter_dir / "original" / f"{seq}_{stem}.md"),
                "processed_md": rel(iter_dir / "processed" / f"{seq}_{stem}.md"),
                "why_json": rel(iter_dir / "processed" / f"{seq}_{stem}.why.json"),
                "pdf_pages": [rel(p) for p in page_rels],
                "check_anchors": anchors,
            },
        })
        print(f"  [ok] {seq} {stem}  chunks={anchors['chunk_count']} "
              f"bp={anchors['boilerplate_count']} dup={anchors['duplicate_count']} "
              f"pages={len(page_rels)}")

    (iter_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    (iter_dir / "index.html").write_text(build_index_html(index_rows), encoding="utf-8")
    (iter_dir / "manifest.jsonl").write_text(json.dumps({
        "run_id": iter_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "judge_model": DEFAULT_JUDGE_MODEL,
        "seed": args.seed,
        "split": args.split,
        "taxonomy_version": TAXONOMY_VERSION,
        "n_stems": len(samples),
        "stems": [s["id"] for s in samples],
    }) + "\n", encoding="utf-8")

    print(f"\nWrote {len(samples)} samples → {iter_dir}")
    print(f"  dataset:  {iter_dir / 'samples.jsonl'}")
    print(f"  review:   open {iter_dir / 'index.html'} (or serve on :8501)")


if __name__ == "__main__":
    main()
