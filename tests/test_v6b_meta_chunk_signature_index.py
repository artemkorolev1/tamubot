"""Tests for v6b_meta_chunk_signature_index — verifies scan, MinHash round-trip,
and LSH lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tamubot.ingestion.pipeline_v6b.util.signature_index import (
    build_lsh_from_df,
    chunk_id_of,
    minhash_from_row,
    scan_corpus_signatures,
)
from tamubot.ingestion.pipeline_v6b.util.text_normalize import minhash_of


def _make_chunk_json(path: Path, contents: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"chunks": [{"content": c, "token_count": len(c.split())} for c in contents]}),
        encoding="utf-8",
    )


def _seed(root: Path) -> None:
    a = root / "CSCE" / "v6b" / "silver" / "02_chunk" / "202531_CSCE_121_500_11111.json"
    b = root / "CSCE" / "v6b" / "silver" / "02_chunk" / "202531_CSCE_121_501_11112.json"
    c = root / "MATH" / "v6b" / "silver" / "02_chunk" / "202531_MATH_151_500_22221.json"
    shared = (
        "Students must attend lecture in person and complete weekly programming assignments "
        "due each Friday by eleven fifty nine pm submitted via the course management system."
    )
    _make_chunk_json(a, [shared, "Unique to A: linked lists and recursion practice."])
    _make_chunk_json(b, [shared, "Unique to B: graph algorithms and dynamic programming."])
    _make_chunk_json(c, ["Calculus content unrelated to programming or assignments."])


def test_chunk_id_format():
    assert chunk_id_of("202531_CSCE_121_500_12345", 7) == "202531_CSCE_121_500_12345#7"


def test_scan_signatures_returns_expected_columns(tmp_path: Path):
    _seed(tmp_path)
    df = scan_corpus_signatures(tmp_path)
    assert set(df.columns) == {"chunk_id", "stem", "dept", "chunk_index", "token_count", "hashvalues"}
    assert len(df) == 5


def test_scan_skips_empty_chunks(tmp_path: Path):
    p = tmp_path / "CSCE" / "v6b" / "silver" / "02_chunk" / "202531_CSCE_121_500_1.json"
    _make_chunk_json(p, ["real content goes here", "", "   "])
    df = scan_corpus_signatures(tmp_path)
    assert len(df) == 1


def test_minhash_roundtrip_via_hashvalues(tmp_path: Path):
    """A MinHash reconstructed from its hashvalues must Jaccard-match the original."""
    text = "the quick brown fox jumps over the lazy dog several times in succession"
    original = minhash_of(text)
    df = pd.DataFrame([{"chunk_id": "x#0", "stem": "x", "chunk_index": 0,
                        "hashvalues": original.hashvalues.tolist()}])
    rt = minhash_from_row(list(df.iloc[0]["hashvalues"]))
    assert original.jaccard(rt) == 1.0


def test_lsh_finds_cross_syllabus_duplicate(tmp_path: Path):
    _seed(tmp_path)
    df = scan_corpus_signatures(tmp_path)
    lsh, sigs = build_lsh_from_df(df, threshold=0.80)
    # The shared chunk appears in stems A and B at chunk_index 0.
    a_id = chunk_id_of("202531_CSCE_121_500_11111", 0)
    b_id = chunk_id_of("202531_CSCE_121_501_11112", 0)
    hits = lsh.query(sigs[a_id])
    assert b_id in hits
    # Self-match present too — caller is responsible for excluding it.
    assert a_id in hits


def test_lsh_excludes_non_duplicate(tmp_path: Path):
    _seed(tmp_path)
    df = scan_corpus_signatures(tmp_path)
    lsh, sigs = build_lsh_from_df(df, threshold=0.80)
    # The MATH chunk is unrelated.
    math_id = chunk_id_of("202531_MATH_151_500_22221", 0)
    hits = set(lsh.query(sigs[math_id])) - {math_id}
    assert hits == set()


def test_parquet_roundtrip_preserves_signatures(tmp_path: Path):
    _seed(tmp_path)
    df = scan_corpus_signatures(tmp_path)
    out = tmp_path / "chunk_signature_index.parquet"
    df.to_parquet(out, index=False)
    rt = pd.read_parquet(out)
    assert len(rt) == len(df)
    # Sample one row and verify MinHash reconstructs identically.
    original_row = df.iloc[0]
    rt_row = rt[rt["chunk_id"] == original_row["chunk_id"]].iloc[0]
    mh_orig = minhash_from_row(list(original_row["hashvalues"]))
    mh_rt = minhash_from_row(list(rt_row["hashvalues"]))
    assert mh_orig.jaccard(mh_rt) == 1.0
