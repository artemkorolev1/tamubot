"""Pure scan/cluster/filter logic for v6b_meta_boilerplate_reference.

Kept dagster-free so tests can exercise the logic without the dagster runtime.
The asset wrapper at assets/meta_boilerplate_reference.py is a thin shell over
these functions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasketch import MinHashLSH

from tamubot.ingestion.pipeline_v5.util import dept_from_stem
from tamubot.ingestion.pipeline_v6b.util.text_normalize import (
    minhash_of,
    ngrams,
    normalize_text,
    stable_cluster_id,
)

MIN_DOC_FREQUENCY = 5
MIN_DISTINCT_DEPTS = 3
LSH_THRESHOLD = 0.85
NUM_PERM = 128
TOP_NGRAMS_FOR_SIGNATURE = 32

_REFERENCE_COLUMNS = [
    "cluster_id",
    "representative_text",
    "normalized_text",
    "doc_frequency",
    "distinct_depts",
    "ngram_signature",
]


def scan_corpus_chunks(data_root: Path) -> list[dict]:
    """Globs DATA_ROOT/<dept>/v6b/silver/02_chunk/<stem>.json and yields per-chunk rows."""
    rows: list[dict] = []
    for p in Path(data_root).glob("*/v6b/silver/02_chunk/*.json"):
        stem = p.stem
        try:
            dept = dept_from_stem(stem)
        except Exception:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for idx, chunk in enumerate(data.get("chunks", [])):
            content = chunk.get("content", "")
            if not content.strip():
                continue
            rows.append(
                {
                    "stem": stem,
                    "dept": dept,
                    "chunk_index": idx,
                    "content": content,
                    "normalized": normalize_text(content),
                }
            )
    return rows


def cluster_chunks(
    rows: list[dict],
    lsh_threshold: float = LSH_THRESHOLD,
) -> dict[str, list[dict]]:
    """Cluster by exact normalized hash, then merge via MinHash-LSH at lsh_threshold.

    Returns {cluster_id: members}. cluster_id is stable across runs.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["normalized"]:
            buckets[r["normalized"]].append(r)

    sigs = {norm: minhash_of(norm) for norm in buckets}
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=NUM_PERM)
    for norm, sig in sigs.items():
        lsh.insert(norm, sig)

    parent: dict[str, str] = {n: n for n in buckets}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for norm in buckets:
        for cand in lsh.query(sigs[norm]):
            if cand != norm:
                union(str(norm), str(cand))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for norm, members in buckets.items():
        grouped[find(norm)].extend(members)

    return {stable_cluster_id(root): members for root, members in grouped.items()}


def build_reference_df(
    clusters: dict[str, list[dict]],
    min_doc_freq: int = MIN_DOC_FREQUENCY,
    min_distinct_depts: int = MIN_DISTINCT_DEPTS,
) -> pd.DataFrame:
    """Apply gating thresholds, pick canonical text per cluster, return parquet-ready DataFrame."""
    out_rows: list[dict] = []
    for cluster_id, members in clusters.items():
        distinct_stems = {m["stem"] for m in members}
        distinct_depts = {m["dept"] for m in members}
        if len(distinct_stems) < min_doc_freq or len(distinct_depts) < min_distinct_depts:
            continue
        norm_counter = Counter(m["normalized"] for m in members)
        top = norm_counter.most_common()
        max_count = top[0][1]
        canonical_norm = sorted(n for n, c in top if c == max_count)[0]
        rep_text = next(m["content"] for m in members if m["normalized"] == canonical_norm)
        sig_ngrams = sorted(set(ngrams(rep_text)))[:TOP_NGRAMS_FOR_SIGNATURE]
        out_rows.append(
            {
                "cluster_id": cluster_id,
                "representative_text": rep_text,
                "normalized_text": canonical_norm,
                "doc_frequency": len(distinct_stems),
                "distinct_depts": len(distinct_depts),
                "ngram_signature": sig_ngrams,
            }
        )
    return pd.DataFrame(out_rows, columns=_REFERENCE_COLUMNS)
