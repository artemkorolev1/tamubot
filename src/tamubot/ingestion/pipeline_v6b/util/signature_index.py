"""Pure scan + signature-build logic for the cross-syllabus dedup index.

Stores per-chunk MinHash signatures in a parquet so the per-syllabus tagger
can look up near-duplicates across the entire corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from tamubot.ingestion.pipeline_v5.util import dept_from_stem
from tamubot.ingestion.pipeline_v6b.util.text_normalize import minhash_of, normalize_text

NUM_PERM = 128
DEFAULT_CROSS_SYL_THRESHOLD = 0.95

_INDEX_COLUMNS = ["chunk_id", "stem", "dept", "chunk_index", "token_count", "hashvalues"]


def chunk_id_of(stem: str, chunk_index: int) -> str:
    """Synthetic stable chunk id: '<stem>#<index>'. Used in both within- and cross-syllabus dedup."""
    return f"{stem}#{chunk_index}"


def scan_corpus_signatures(data_root: Path) -> pd.DataFrame:
    """Walk the silver_chunk_semantic outputs, build a MinHash per non-empty chunk,
    and return a DataFrame ready to write as the signature index parquet.

    Boilerplate chunks ARE included in the signature index. Boilerplate filtering
    happens in retrieval; the signature index is the raw corpus picture.
    """
    rows: list[dict] = []
    for p in Path(data_root).glob("*/v6b/silver/02_chunk/*.json"):
        stem = p.stem
        try:
            dept = dept_from_stem(stem)
        except Exception:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for pos, chunk in enumerate(data.get("chunks", [])):
            content = chunk.get("content", "")
            if not normalize_text(content):
                continue
            chunk_index = int(chunk.get("chunk_index", pos))
            mh = minhash_of(content)
            rows.append(
                {
                    "chunk_id": chunk_id_of(stem, chunk_index),
                    "stem": stem,
                    "dept": dept,
                    "chunk_index": chunk_index,
                    "token_count": int(chunk.get("token_count") or 0),
                    "hashvalues": mh.hashvalues.tolist(),
                }
            )
    return pd.DataFrame(rows, columns=_INDEX_COLUMNS)


def minhash_from_row(hashvalues: list[int], num_perm: int = NUM_PERM) -> MinHash:
    """Reconstruct a MinHash from a row's hashvalues list.

    Guards against a parquet written under a different NUM_PERM: a length mismatch
    would otherwise produce silently wrong Jaccard comparisons. Rebuild the index
    if this fires.
    """
    if len(hashvalues) != num_perm:
        raise ValueError(
            f"hashvalues length {len(hashvalues)} != num_perm {num_perm}; the "
            "signature index was built with a different NUM_PERM — rebuild "
            "v6b_meta_chunk_signature_index."
        )
    mh = MinHash(num_perm=num_perm)
    mh.hashvalues = np.array(hashvalues, dtype=np.uint64)
    return mh


def build_lsh_from_df(
    df: pd.DataFrame,
    threshold: float = DEFAULT_CROSS_SYL_THRESHOLD,
    num_perm: int = NUM_PERM,
) -> tuple[MinHashLSH, dict[str, MinHash]]:
    """Build an LSH index keyed by chunk_id, plus a chunk_id -> MinHash map for true-Jaccard scoring."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs: dict[str, MinHash] = {}
    for row in df.itertuples(index=False):
        mh = minhash_from_row(list(row.hashvalues), num_perm=num_perm)
        sigs[str(row.chunk_id)] = mh
        if row.chunk_id not in lsh:
            lsh.insert(str(row.chunk_id), mh)
    return lsh, sigs
