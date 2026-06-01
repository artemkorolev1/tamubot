"""Pure text-normalization + MinHash helpers for v6b boilerplate / dedup tagging."""

from __future__ import annotations

import hashlib
import re
import string
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")

_NUM_PERM = 128
_NGRAM_N = 5


def normalize_text(text: str) -> str:
    """Lowercase, strip ASCII punctuation, collapse whitespace. Idempotent."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", lowered).strip()


def ngrams(text: str, n: int = 5) -> list[str]:
    """Return word n-grams (space-joined) over normalized text."""
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(a: set[str], b: set[str]) -> float:
    """Set Jaccard. 0.0 when both sides are empty."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def minhash_of(text: str, num_perm: int = _NUM_PERM) -> MinHash:
    """MinHash over the 5-gram set of normalize_text(text). Empty MinHash if no n-grams."""
    mh = MinHash(num_perm=num_perm)
    grams = ngrams(text, n=_NGRAM_N)
    if not grams:
        return mh
    for g in set(grams):
        mh.update(g.encode("utf-8"))
    return mh


def stable_cluster_id(normalized_text: str) -> str:
    """Stable cluster id: bp_ + first 16 hex chars of sha256(normalized_text)."""
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return f"bp_{digest[:16]}"


class ReferenceIndex:
    """Wraps the boilerplate reference parquet for exact + fuzzy lookup."""

    def __init__(self, parquet_path: Path):
        self._exact: dict[str, str] = {}
        self._lsh: MinHashLSH = MinHashLSH(threshold=0.80, num_perm=_NUM_PERM)
        self._cluster_to_signature: dict[str, MinHash] = {}

        df = pd.read_parquet(parquet_path)
        for row in df.itertuples(index=False):
            cluster_id = str(row.cluster_id)
            rep_text = str(row.representative_text)
            normalized = normalize_text(rep_text)
            self._exact[normalized] = cluster_id
            sig = minhash_of(rep_text)
            self._cluster_to_signature[cluster_id] = sig
            if cluster_id not in self._lsh:
                self._lsh.insert(cluster_id, sig)

    @classmethod
    def empty(cls) -> "ReferenceIndex":
        """Empty index — used when the parquet does not yet exist."""
        inst = cls.__new__(cls)
        inst._exact = {}
        inst._lsh = MinHashLSH(threshold=0.80, num_perm=_NUM_PERM)
        inst._cluster_to_signature = {}
        return inst

    def match(
        self, content: str, threshold: float = 0.80
    ) -> tuple[str | None, float | None]:
        """Exact-normalized match first; else fuzzy via LSH + true Jaccard against candidate sigs."""
        normalized = normalize_text(content)
        exact_hit = self._exact.get(normalized)
        if exact_hit is not None:
            return exact_hit, 1.0

        query_sig = minhash_of(content)
        candidates = self._lsh.query(query_sig)
        best_id: str | None = None
        best_score: float = 0.0
        for cid in candidates:
            cand_sig = self._cluster_to_signature.get(cid)
            if cand_sig is None:
                continue
            score = query_sig.jaccard(cand_sig)
            if score >= threshold and score > best_score:
                best_id = cid
                best_score = score
        if best_id is None:
            return None, None
        return best_id, best_score
