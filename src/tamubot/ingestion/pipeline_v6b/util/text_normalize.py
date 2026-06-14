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

# OCR ligature-loss folding (STAT_651 class). PDF text layers frequently drop the
# f-ligature glyphs (fi/fl/ff/ffi/ffl), so "office"->"ofce", "confidentiality"->
# "confdentiality", "staff"->"staf". A *clean* corpus copy and an OCR-damaged copy
# of the same university policy then share almost no 5-grams, sinking the MinHash
# Jaccard below the boilerplate floor (Mental Health 0.094, Title IX 0.086).
# Rather than try to REPAIR (we can't know the dropped letters), we FOLD both forms
# to the same damaged shape: collapse each f-ligature cluster to a bare "f". Applied
# symmetrically to every text in the signature path, so clean and damaged copies of
# the same policy land on identical shingles. Longest clusters first so "office"
# (ffi) folds in one pass. Applied ONLY to MinHash shingles — normalize_text and the
# stable_cluster_id hash are deliberately untouched (changing them would rebuild
# every cluster id). NOTE: minhash_of feeds the persisted signature index, so
# v6b_meta_chunk_signature_index must be rebuilt after this change.
_LIGATURE_RE = re.compile(r"ffi|ffl|ff|fi|fl")


def fold_ligatures(text: str) -> str:
    """Collapse OCR-prone f-ligature clusters (ffi/ffl/ff/fi/fl) to a bare ``f``.

    Maps a clean word and its ligature-dropped OCR twin to one canonical form
    ("office" and "ofce" both -> "ofce"). Idempotent. Operates on already-lowered
    text; callers normalize first.
    """
    return _LIGATURE_RE.sub("f", text)

# Header-anchored boilerplate fallback (see ReferenceIndex.match_by_header).
# A header only joins the allow-list if its cluster is a genuine cross-corpus
# policy: present in at least this many distinct departments. The real
# university-policy clusters span all 4 depts (doc_freq 53-105); low-frequency
# one-offs (e.g. a course-specific "Attendance Policy" seen in df=6, 3 depts)
# stay out, so a same-titled course section is never header-flagged.
HEADER_ALLOWLIST_MIN_DEPTS = 4
# The header match is the dominant signal, but the body floor must still reject a
# same-titled COURSE-CONFIGURABLE section (e.g. an instructor's own "Late Work
# Policy" / "Makeup Work Policy") whose body diverges from the standard wording.
# Observed: ECEN_749's course-specific late-homework rules scored 0.055 (false
# positive at 0.05); STAT_651's genuine-but-OCR-damaged Mental Health statement
# scored 0.094. 0.12 rejects the former (over-hide) while keeping real policy
# variants (>=0.125); the two too-damaged STAT_651 statements (0.086/0.094) fall
# back to a BP_MISSED miss rather than risk hiding course content.
HEADER_BODY_JACCARD_FLOOR = 0.12


def normalize_header(text: str) -> str:
    """Normalize a chunk/cluster leading header for allow-list comparison.

    Strips a leading markdown header run (``#``/``##``/...), a trailing colon,
    then applies :func:`normalize_text` (lowercase, drop punctuation, collapse
    whitespace). The ligature-loss OCR damage ("Ofce", "confdentiality") that
    shreds bodies is mostly absent from these short policy headers, so no
    special ligature repair is needed here.
    """
    first_line = text.lstrip().split("\n", 1)[0]
    stripped = first_line.lstrip("#").strip().rstrip(":").strip()
    return normalize_text(stripped)


def normalize_text(text: str) -> str:
    """Lowercase, strip ASCII punctuation, collapse whitespace. Idempotent."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", lowered).strip()


def ngrams(text: str, n: int = 5) -> list[str]:
    """Return word n-grams (space-joined) over normalized, ligature-folded text.

    Folding makes the shingles robust to OCR f-ligature loss (see fold_ligatures);
    it is intentionally absent from normalize_text so only the fuzzy-match path is
    affected.
    """
    tokens = fold_ligatures(normalize_text(text)).split()
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
    """MinHash over the 5-gram set of normalize_text(text).

    Short text (<5 tokens) has no 5-grams. Rather than return an all-default
    ("empty") MinHash — which has Jaccard 1.0 with *every* other empty MinHash and
    so makes all short strings collide as "identical" — fall back to the whole
    normalized string as a single shingle. Distinct short strings then get
    distinct signatures (Jaccard 0), while identical short strings still match
    (Jaccard 1). Truly empty text still yields an empty MinHash (callers skip it).
    """
    mh = MinHash(num_perm=num_perm)
    grams = ngrams(text, n=_NGRAM_N)
    if not grams:
        norm = fold_ligatures(normalize_text(text))
        if norm:
            mh.update(norm.encode("utf-8"))
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
        # normalized policy header -> cluster_id, for the header-anchored fallback.
        self._header_to_cluster: dict[str, str] = {}

        df = pd.read_parquet(parquet_path)
        has_depts = "distinct_depts" in df.columns
        has_freq = "doc_frequency" in df.columns
        # Keep the highest-doc_frequency cluster per normalized header, so a
        # header maps to its most representative policy text.
        header_freq: dict[str, int] = {}
        for row in df.itertuples(index=False):
            cluster_id = str(row.cluster_id)
            rep_text = str(row.representative_text)
            normalized = normalize_text(rep_text)
            self._exact[normalized] = cluster_id
            sig = minhash_of(rep_text)
            self._cluster_to_signature[cluster_id] = sig
            if cluster_id not in self._lsh:
                self._lsh.insert(cluster_id, sig)

            # Dynamic header allow-list: only genuine cross-corpus policies
            # (seen across >= HEADER_ALLOWLIST_MIN_DEPTS departments).
            distinct_depts = int(getattr(row, "distinct_depts", 0)) if has_depts else 0
            doc_freq = int(getattr(row, "doc_frequency", 0)) if has_freq else 0
            if distinct_depts < HEADER_ALLOWLIST_MIN_DEPTS:
                continue
            header = normalize_header(rep_text)
            if not header:
                continue
            if header not in header_freq or doc_freq > header_freq[header]:
                header_freq[header] = doc_freq
                self._header_to_cluster[header] = cluster_id

    @classmethod
    def empty(cls) -> "ReferenceIndex":
        """Empty index — used when the parquet does not yet exist."""
        inst = cls.__new__(cls)
        inst._exact = {}
        inst._lsh = MinHashLSH(threshold=0.80, num_perm=_NUM_PERM)
        inst._cluster_to_signature = {}
        inst._header_to_cluster = {}
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

    def match_by_header(
        self, content: str, body_floor: float = HEADER_BODY_JACCARD_FLOOR
    ) -> tuple[str | None, float | None]:
        """Header-anchored boilerplate fallback for below-threshold policy variants.

        When a chunk's leading header, normalized, equals a known cross-corpus
        policy header (the dynamically-derived allow-list), AND its body shares
        at least ``body_floor`` MinHash-Jaccard with that cluster, return that
        cluster. The header equality is the dominant signal; the floor only
        rejects a same-titled but genuinely course-specific section that has no
        real overlap with the canonical policy text.

        Returns ``(cluster_id, body_jaccard)`` or ``(None, None)``. Callers
        should treat this as a *fallback* after :meth:`match` declines.
        """
        header = normalize_header(content)
        if not header:
            return None, None
        cluster_id = self._header_to_cluster.get(header)
        if cluster_id is None:
            return None, None
        cand_sig = self._cluster_to_signature.get(cluster_id)
        if cand_sig is None:
            return None, None
        score = minhash_of(content).jaccard(cand_sig)
        if score < body_floor:
            return None, None
        return cluster_id, score
