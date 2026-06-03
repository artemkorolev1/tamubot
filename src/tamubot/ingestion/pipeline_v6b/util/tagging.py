"""Pure three-pass tagging logic for silver_tag.

Kept dagster-free so tests can import these helpers directly. The asset
wrapper at assets/silver_tag.py is a thin shell over _tag_chunks.
"""

from __future__ import annotations

import pandas as pd
from datasketch import MinHash, MinHashLSH

from tamubot.ingestion.pipeline_v6b.util.signature_index import (
    NUM_PERM,
    build_lsh_from_df,
    chunk_id_of,
)
from tamubot.ingestion.pipeline_v6b.util.text_normalize import (
    ReferenceIndex,
    minhash_of,
    normalize_text,
)

BP_THRESHOLD = 0.80
WITHIN_SYL_THRESHOLD = 0.92
CROSS_SYL_THRESHOLD = 0.95


def flag_boilerplate(chunks: list[dict], reference_index: ReferenceIndex) -> None:
    for c in chunks:
        cluster_id, conf = reference_index.match(c.get("content", ""), threshold=BP_THRESHOLD)
        if cluster_id is not None:
            c["is_boilerplate"] = True
            c["boilerplate_cluster"] = cluster_id
            c["cluster_confidence"] = conf


def build_local_signatures(chunks: list[dict]) -> dict[int, MinHash]:
    """MinHash per non-boilerplate, non-empty chunk. Keyed by position in the list."""
    sigs: dict[int, MinHash] = {}
    for i, c in enumerate(chunks):
        if c.get("is_boilerplate"):
            continue
        if not normalize_text(c.get("content", "")):
            continue
        sigs[i] = minhash_of(c["content"])
    return sigs


def flag_within_syllabus_dups(
    chunks: list[dict],
    stem: str,
    local_sigs: dict[int, MinHash],
) -> None:
    """Collapse near-duplicate chunks within one syllabus.

    Builds dup *components* via union-find over all true-duplicate pairs (not
    per-seed neighbourhoods), so transitive chains and query order don't change
    the outcome. Canonical per component = longest content, lowest chunk_index
    tie-break. Every non-canonical member points at that one canonical.
    """
    if not local_sigs:
        return
    lsh = MinHashLSH(threshold=WITHIN_SYL_THRESHOLD, num_perm=NUM_PERM)
    for i, sig in local_sigs.items():
        lsh.insert(str(i), sig)

    parent: dict[int, int] = {i: i for i in local_sigs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, sig in local_sigs.items():
        for c in lsh.query(sig):
            j = int(c)
            if j == i or j not in local_sigs:
                continue
            if sig.jaccard(local_sigs[j]) >= WITHIN_SYL_THRESHOLD:
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in local_sigs:
        components.setdefault(find(i), []).append(i)

    def canonical_key(idx: int) -> tuple[int, int]:
        content_len = len(chunks[idx].get("content", "") or "")
        ci = chunks[idx].get("chunk_index")
        return (content_len, -(int(ci) if ci is not None else idx))

    for members in components.values():
        if len(members) < 2:
            continue
        canonical = max(members, key=canonical_key)
        canonical_chunk_index = chunks[canonical].get("chunk_index", canonical)
        for idx in members:
            if idx == canonical:
                continue
            chunks[idx]["is_duplicate"] = True
            chunks[idx]["duplicate_of_chunk_id"] = chunk_id_of(stem, canonical_chunk_index)


def flag_cross_syllabus_dups(
    chunks: list[dict],
    stem: str,
    local_sigs: dict[int, MinHash],
    cross_lsh: MinHashLSH,
    cross_sigs: dict[str, MinHash],
    cross_metadata: dict[str, dict],
) -> None:
    """Canonical = lex-min (stem, chunk_index) across the dup cluster.

    Same-stem candidates are excluded — intra-syllabus duplicates are owned by
    flag_within_syllabus_dups, and letting the cross pass re-flag them (with a
    different canonical rule) could mark a within-syllabus canonical as a dup,
    leaving an intra-doc group with no surviving copy.
    """
    for i, sig in local_sigs.items():
        if chunks[i].get("is_duplicate"):
            continue
        my_chunk_index = chunks[i].get("chunk_index", i)
        my_id = chunk_id_of(stem, my_chunk_index)
        candidates: list[str] = []
        for cid in cross_lsh.query(sig):
            cid = str(cid)
            if cid == my_id:
                continue
            meta = cross_metadata.get(cid)
            if meta is not None and meta.get("stem") == stem:
                continue  # same-stem twin → handled by within-syllabus dedup
            candidates.append(cid)
        true_dups = [
            cid for cid in candidates
            if cid in cross_sigs and sig.jaccard(cross_sigs[cid]) >= CROSS_SYL_THRESHOLD
        ]
        if not true_dups:
            continue
        cluster_ids = [my_id, *true_dups]

        def sort_key(cid: str) -> tuple[str, int]:
            meta = cross_metadata.get(cid)
            if meta is not None:
                return (meta["stem"], int(meta["chunk_index"]))
            try:
                s, idx_str = cid.rsplit("#", 1)
                return (s, int(idx_str))
            except ValueError:
                return (cid, 0)

        canonical_id = min(cluster_ids, key=sort_key)
        if canonical_id != my_id:
            chunks[i]["is_duplicate"] = True
            chunks[i]["duplicate_of_chunk_id"] = canonical_id


def tag_chunks(
    chunks: list[dict],
    stem: str,
    reference_index: ReferenceIndex,
    cross_lsh: MinHashLSH | None,
    cross_sigs: dict[str, MinHash],
    cross_metadata: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Three-pass tagger: boilerplate -> within-syllabus dedup -> cross-syllabus dedup.

    Never drops chunks. Boilerplate wins over dedup (a boilerplate chunk is never also
    flagged as a duplicate).
    """
    flag_boilerplate(chunks, reference_index)
    local_sigs = build_local_signatures(chunks)
    flag_within_syllabus_dups(chunks, stem, local_sigs)
    if cross_lsh is not None:
        flag_cross_syllabus_dups(chunks, stem, local_sigs, cross_lsh, cross_sigs, cross_metadata)
    stats = {
        "tagged_boilerplate": sum(1 for c in chunks if c.get("is_boilerplate")),
        "tagged_duplicate": sum(1 for c in chunks if c.get("is_duplicate")),
    }
    return chunks, stats


def load_cross_syllabus_index_from_parquet(
    parquet_path,
) -> tuple[MinHashLSH, dict[str, MinHash], dict[str, dict]]:
    """Build the cross-syl LSH + sigs + metadata from a signature-index parquet on disk."""
    df = pd.read_parquet(parquet_path)
    lsh, sigs = build_lsh_from_df(df, threshold=CROSS_SYL_THRESHOLD, num_perm=NUM_PERM)
    metadata: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        metadata[str(row.chunk_id)] = {
            "stem": str(row.stem),
            "chunk_index": int(row.chunk_index),
            "dept": str(row.dept),
        }
    return lsh, sigs, metadata
