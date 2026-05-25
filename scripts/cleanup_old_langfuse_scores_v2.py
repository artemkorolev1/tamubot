"""Faster cleanup of obsolete-schema scores — queries by name (much fewer
list calls than per-trace pagination, which was rate-limited).

  - Obsolete names: delete every instance project-wide (they only exist
    because of earlier backfill runs on curated20_v2 traces).
  - Dedup names (chunks_retrieved, chunk_tokens): per trace, keep the
    newest, delete the rest.

Run from /workspace:
    python3 scripts/cleanup_old_langfuse_scores_v2.py
"""

from __future__ import annotations

import base64
import os
import sys
import time
from collections import defaultdict

import httpx
from langfuse import Langfuse

OBSOLETE_NAMES = (
    "chunks_vector_only",
    "chunks_bm25_only",
    "chunks_both",
    "vector_raw_count",
    "bm25_raw_count",
)
DEDUP_NAMES = ("chunks_retrieved", "chunk_tokens")

_LF_HOST = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
_LF_AUTH = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()
_HTTP = httpx.Client(timeout=60.0, headers={"Authorization": f"Basic {_LF_AUTH}"})


def _delete_score(score_id: str) -> bool:
    backoff = 1.0
    for _ in range(6):
        try:
            r = _HTTP.delete(f"{_LF_HOST}/api/public/scores/{score_id}")
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        if r.status_code in (200, 202, 204):
            time.sleep(0.05)
            return True
        if r.status_code in (429, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        return False
    return False


def _list_all_by_name(lf: Langfuse, name: str) -> list:
    """Page through every score with this name."""
    out = []
    page_n = 1
    while True:
        backoff = 1.0
        for _ in range(5):
            try:
                page = lf.api.scores.get_many(name=name, limit=100, page=page_n)
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "504" in msg or "timed out" in msg.lower():
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                raise
        else:
            print(f"  ! list scores name={name} page={page_n}: gave up")
            break
        data = page.data or []
        out.extend(data)
        meta = getattr(page, "meta", None)
        total_pages = getattr(meta, "total_pages", None) if meta else None
        if total_pages is None or page_n >= total_pages:
            break
        page_n += 1
    return out


def _ts(score) -> str:
    return str(getattr(score, "timestamp", None) or getattr(score, "created_at", "") or "")


def main() -> int:
    lf = Langfuse()
    total_deleted = 0

    # 1. Obsolete names — delete every instance.
    for name in OBSOLETE_NAMES:
        scores = _list_all_by_name(lf, name)
        print(f"\n[{name}] found {len(scores)} scores")
        n_deleted = 0
        for s in scores:
            if _delete_score(s.id):
                n_deleted += 1
        print(f"  deleted {n_deleted}/{len(scores)}")
        total_deleted += n_deleted

    # 2. Dedup names — per trace, keep newest, delete the rest.
    for name in DEDUP_NAMES:
        scores = _list_all_by_name(lf, name)
        by_trace: dict[str, list] = defaultdict(list)
        for s in scores:
            tid = s.trace_id or ""
            if tid:
                by_trace[tid].append(s)
        dup_count = 0
        for tid, instances in by_trace.items():
            if len(instances) <= 1:
                continue
            instances.sort(key=_ts, reverse=True)
            for s in instances[1:]:
                if _delete_score(s.id):
                    dup_count += 1
        print(
            f"\n[{name}] {len(scores)} total, {sum(1 for v in by_trace.values() if len(v) > 1)} traces with dups, {dup_count} deleted"
        )
        total_deleted += dup_count

    print(f"\ndone: {total_deleted} scores deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
