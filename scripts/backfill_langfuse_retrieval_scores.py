"""Backfill stage-by-stage retrieval scores onto past Langfuse traces.

For each run of a dataset, walk linked trace items, parse retrieval / rerank /
node.retrieval observations, and post these 13 numeric scores:

    Stage A (post-fusion, summed across search calls):
        raw_vector_only, raw_bm25_only, raw_overlap, raw_total
    Stage B (post-rerank, post-threshold+knee cutoff, summed across rerank calls):
        cutoff_vector_only, cutoff_bm25_only, cutoff_overlap, cutoff_total
    Stage C (final list, after dedup + per-course cap):
        final_vector_only, final_bm25_only, final_overlap, chunks_retrieved
    Final tokens:
        chunk_tokens

Old prior-backfill scores are deleted before the new schema is posted so the
Langfuse columns are clean:
    chunks_vector_only, chunks_bm25_only, chunks_both,
    vector_raw_count, bm25_raw_count
    (chunks_retrieved and chunk_tokens get re-posted under the same name.)

Run from /workspace:
    python3 scripts/backfill_langfuse_retrieval_scores.py
"""

from __future__ import annotations

import base64
import os
import sys
from collections import Counter
from typing import Optional

import httpx
from langfuse import Langfuse

_LF_HOST = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
_LF_AUTH = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()
_HTTP = httpx.Client(timeout=30.0, headers={"Authorization": f"Basic {_LF_AUTH}"})

DATASET = "ragas_20260519_curated20_v2"

# Scores from prior backfill (and the initial live posts) to delete.
# chunks_retrieved + chunk_tokens are listed here too because they were posted
# under the same name and a re-post would create duplicates.
SCORES_TO_DELETE = {
    "chunks_vector_only",
    "chunks_bm25_only",
    "chunks_both",
    "vector_raw_count",
    "bm25_raw_count",
    "chunks_retrieved",
    "chunk_tokens",
}


def _tally(chunks: list[dict]) -> dict:
    out = {"vector_only": 0, "bm25_only": 0, "overlap": 0, "total": len(chunks)}
    for c in chunks:
        src = c.get("rrf_source") or ""
        if src == "vector":
            out["vector_only"] += 1
        elif src == "bm25":
            out["bm25_only"] += 1
        elif src == "both":
            out["overlap"] += 1
    return out


def _aggregate_trace(trace) -> Optional[dict]:
    """Walk observations and return the 13 metrics, or None when essential data is missing."""
    raw_vector_only = 0
    raw_bm25_only = 0
    raw_overlap = 0
    raw_total = 0
    cutoff_acc = Counter()
    final_chunks: Optional[list] = None

    saw_search = False
    saw_rerank = False

    for obs in trace.observations or []:
        name = obs.name or ""
        md = obs.metadata or {}
        if name in ("node.retrieval.search.hybrid", "node.retrieval.search.semantic"):
            saw_search = True
            sb = md.get("source_breakdown") or {}
            fused = md.get("fused_results")
            raw_vector_only += int(sb.get("vector", 0) or 0)
            raw_bm25_only += int(sb.get("bm25", 0) or 0)
            raw_overlap += int(sb.get("both", 0) or 0)
            if isinstance(fused, int):
                raw_total += fused
        elif name == "node.retrieval.rerank":
            saw_rerank = True
            fc = md.get("final_chunks") or []
            for ch in fc:
                src = ch.get("rrf_source") or ""
                if src in ("vector", "bm25", "both"):
                    cutoff_acc[src] += 1
                cutoff_acc["total"] += 1
        elif name == "node.retrieval":
            out = obs.output
            if isinstance(out, dict) and isinstance(out.get("retrieved_chunks"), list):
                final_chunks = out["retrieved_chunks"]

    if not (saw_search or saw_rerank or final_chunks is not None):
        return None

    final_tally = _tally(final_chunks or [])
    chunk_tokens = sum(len(c.get("content", "") or "") for c in final_chunks) // 4 if final_chunks else None

    metrics: dict[str, float] = {
        # Stage A
        "raw_vector_only": float(raw_vector_only),
        "raw_bm25_only": float(raw_bm25_only),
        "raw_overlap": float(raw_overlap),
        "raw_total": float(raw_total),
        # Stage B
        "cutoff_vector_only": float(cutoff_acc.get("vector", 0)),
        "cutoff_bm25_only": float(cutoff_acc.get("bm25", 0)),
        "cutoff_overlap": float(cutoff_acc.get("both", 0)),
        "cutoff_total": float(cutoff_acc.get("total", 0)),
        # Stage C
        "final_vector_only": float(final_tally["vector_only"]),
        "final_bm25_only": float(final_tally["bm25_only"]),
        "final_overlap": float(final_tally["overlap"]),
        "chunks_retrieved": float(final_tally["total"]),
    }
    if chunk_tokens is not None:
        metrics["chunk_tokens"] = float(chunk_tokens)
    return metrics


def _delete_old_scores(lf: Langfuse, trace_id: str) -> int:
    """Delete prior-backfill scores so the new schema doesn't collide. Returns count deleted."""
    deleted = 0
    try:
        page = lf.api.scores.get_many(trace_id=trace_id, limit=200)
    except Exception:
        return 0
    for s in page.data or []:
        if (s.name or "") in SCORES_TO_DELETE:
            try:
                resp = _HTTP.delete(f"{_LF_HOST}/api/public/scores/{s.id}")
                if resp.status_code in (200, 202, 204):
                    deleted += 1
            except Exception:
                pass
    return deleted


def main() -> int:
    lf = Langfuse()

    runs_resp = lf.api.datasets.get_runs(dataset_name=DATASET, limit=100)
    runs = runs_resp.data
    print(f"dataset={DATASET}  runs={len(runs)}")

    total_traces = 0
    total_scores_posted = 0
    total_scores_deleted = 0
    skipped = 0

    for run_meta in runs:
        run = lf.api.datasets.get_run(dataset_name=DATASET, run_name=run_meta.name)
        items = run.dataset_run_items or []
        print(f"\n[{run_meta.name}]  items={len(items)}")

        for item in items:
            trace_id = item.trace_id
            if not trace_id:
                skipped += 1
                continue
            try:
                trace = lf.api.trace.get(trace_id)
            except Exception as exc:
                print(f"  ! trace.get {trace_id[:8]}: {exc}")
                skipped += 1
                continue

            metrics = _aggregate_trace(trace)
            if metrics is None:
                skipped += 1
                continue

            total_scores_deleted += _delete_old_scores(lf, trace_id)

            for name, value in metrics.items():
                try:
                    lf.create_score(trace_id=trace_id, name=name, value=float(value))
                    total_scores_posted += 1
                except Exception as exc:
                    print(f"  ! create_score {name} on {trace_id[:8]}: {exc}")
            total_traces += 1

        lf.flush()

    print(
        f"\ndone: {total_traces} traces touched, "
        f"{total_scores_posted} scores posted, {total_scores_deleted} old scores deleted, "
        f"{skipped} skipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
