"""Clean up obsolete-schema scores on past curated20_v2 traces.

The previous backfill posted (now-obsolete) score names and then a second
backfill posted today's new schema without deleting the old ones (the delete
call swallowed a 400 because limit=200 exceeded the API max of 100).

This script:
  1. Walks each trace linked to any run of `ragas_20260519_curated20_v2`.
  2. Lists scores via lf.api.scores.get_many(trace_id=..., limit=100).
  3. Deletes all scores whose name is in OBSOLETE_NAMES.
  4. For names in DEDUP_NAMES that have duplicates, deletes all but the most
     recent (newest timestamp wins).

Run from /workspace:
    python3 scripts/cleanup_old_langfuse_scores.py
"""

from __future__ import annotations

import base64
import os
import sys
import time  # noqa: F401  (used in _delete_score backoff)
from collections import defaultdict

import httpx
from langfuse import Langfuse

DATASET = "ragas_20260519_curated20_v2"

# Names from the old schema — delete every instance.
OBSOLETE_NAMES = {
    "chunks_vector_only",
    "chunks_bm25_only",
    "chunks_both",
    "vector_raw_count",
    "bm25_raw_count",
}
# Names that exist in both schemas — for traces with duplicates, keep newest, delete the rest.
DEDUP_NAMES = {"chunks_retrieved", "chunk_tokens"}

_LF_HOST = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
_LF_AUTH = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()
_HTTP = httpx.Client(timeout=30.0, headers={"Authorization": f"Basic {_LF_AUTH}"})


def _delete_score(score_id: str) -> bool:
    """DELETE with backoff on 429. Sleeps between calls to stay under the rate limit."""
    backoff = 1.0
    for _ in range(6):
        try:
            r = _HTTP.delete(f"{_LF_HOST}/api/public/scores/{score_id}")
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        if r.status_code in (200, 202, 204):
            time.sleep(0.05)  # small steady-state spacing
            return True
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        # Other error — log once and bail
        print(f"    ! delete {score_id} -> {r.status_code} {r.text[:120]}")
        return False
    return False


def _ts(score) -> str:
    return str(getattr(score, "timestamp", None) or getattr(score, "created_at", "") or "")


def _cleanup_trace(lf: Langfuse, trace_id: str) -> tuple[int, int]:
    """Delete obsolete + duplicate scores on this trace. Returns (obsolete_deleted, dups_deleted)."""
    try:
        page = lf.api.scores.get_many(trace_id=trace_id, limit=100)
    except Exception as exc:
        print(f"  ! list scores {trace_id[:8]}: {exc}")
        return 0, 0
    by_name: dict[str, list] = defaultdict(list)
    for s in page.data or []:
        by_name[s.name or ""].append(s)

    obsolete_deleted = 0
    for name in OBSOLETE_NAMES:
        for s in by_name.get(name, []):
            if _delete_score(s.id):
                obsolete_deleted += 1

    dups_deleted = 0
    for name in DEDUP_NAMES:
        instances = by_name.get(name, [])
        if len(instances) <= 1:
            continue
        # Keep the newest, delete the rest.
        instances.sort(key=_ts, reverse=True)
        for s in instances[1:]:
            if _delete_score(s.id):
                dups_deleted += 1

    return obsolete_deleted, dups_deleted


def main() -> int:
    lf = Langfuse()
    runs = lf.api.datasets.get_runs(dataset_name=DATASET, limit=100).data
    print(f"dataset={DATASET}  runs={len(runs)}")
    total_obsolete = 0
    total_dups = 0
    total_traces = 0
    for run_meta in runs:
        run = lf.api.datasets.get_run(dataset_name=DATASET, run_name=run_meta.name)
        items = run.dataset_run_items or []
        run_obsolete = 0
        run_dups = 0
        for item in items:
            tid = item.trace_id
            if not tid:
                continue
            o, d = _cleanup_trace(lf, tid)
            run_obsolete += o
            run_dups += d
            total_traces += 1
        total_obsolete += run_obsolete
        total_dups += run_dups
        print(f"  [{run_meta.name}]  obsolete_deleted={run_obsolete}  dups_deleted={run_dups}")

    print(
        f"\ndone: {total_traces} traces, {total_obsolete} obsolete-schema scores deleted, "
        f"{total_dups} duplicate scores deleted"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
