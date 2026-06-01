"""Tests for v6b_meta_boilerplate_reference asset.

Synthesizes a tiny on-disk corpus and verifies the scan -> cluster -> filter
pipeline produces a sensible parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tamubot.ingestion.pipeline_v6b.util.boilerplate_clustering import (
    build_reference_df,
    cluster_chunks,
    scan_corpus_chunks,
)

# A chunk widely shared across syllabi from multiple depts — should clear gating.
_BP_TEXT = (
    "Texas A and M University will not tolerate any form of academic dishonesty "
    "and students are bound by the Aggie Honor Code which states that an Aggie "
    "does not lie cheat or steal or tolerate those who do."
)

# Dept-specific chunk — only appears within one dept; should NOT clear distinct_depts.
_SINGLE_DEPT_REPEAT = (
    "Welcome to the Department of Computer Science and Engineering at Texas A and M "
    "University. We are pleased that you are joining us this semester."
)

# Unique-per-syllabus chunks — content particular to each course.
_UNIQUE_PREFIX = (
    "This course covers the following learning outcomes specific to section number "
)


def _make_chunk_json(path: Path, contents: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": [{"content": c} for c in contents]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_corpus(root: Path) -> None:
    """Build a synthetic 10-stem corpus across 3 depts."""
    # Depts: CSCE, MATH, ENGR. 4 syllabi in CSCE, 3 in MATH, 3 in ENGR.
    layout = [
        ("CSCE", "202531_CSCE_121_500_12345"),
        ("CSCE", "202531_CSCE_221_501_12346"),
        ("CSCE", "202531_CSCE_311_502_12347"),
        ("CSCE", "202531_CSCE_411_503_12348"),
        ("MATH", "202531_MATH_151_500_22345"),
        ("MATH", "202531_MATH_152_501_22346"),
        ("MATH", "202531_MATH_251_502_22347"),
        ("ENGR", "202531_ENGR_102_500_32345"),
        ("ENGR", "202531_ENGR_217_501_32346"),
        ("ENGR", "202531_ENGR_310_502_32347"),
    ]
    for dept, stem in layout:
        p = root / dept / "v6b" / "silver" / "02_chunk" / f"{stem}.json"
        chunks = [_BP_TEXT, _UNIQUE_PREFIX + stem]
        if dept == "CSCE":
            chunks.append(_SINGLE_DEPT_REPEAT)
        _make_chunk_json(p, chunks)


def test_scan_corpus_finds_all_chunks(tmp_path: Path):
    _seed_corpus(tmp_path)
    rows = scan_corpus_chunks(tmp_path)
    # 10 syllabi * 2 chunks each, plus 4 extra on CSCE = 24.
    assert len(rows) == 24
    assert {r["dept"] for r in rows} == {"CSCE", "MATH", "ENGR"}


def test_scan_skips_empty_content(tmp_path: Path):
    p = tmp_path / "CSCE" / "v6b" / "silver" / "02_chunk" / "202531_CSCE_121_500_1.json"
    _make_chunk_json(p, ["real content here", "   ", ""])
    rows = scan_corpus_chunks(tmp_path)
    assert len(rows) == 1


def test_cluster_groups_exact_matches(tmp_path: Path):
    _seed_corpus(tmp_path)
    rows = scan_corpus_chunks(tmp_path)
    clusters = cluster_chunks(rows)
    # The boilerplate text should be in a single cluster with 10 members.
    bp_cluster = next(c for c in clusters.values() if len(c) == 10)
    assert {m["dept"] for m in bp_cluster} == {"CSCE", "MATH", "ENGR"}


def test_build_reference_applies_gating(tmp_path: Path):
    _seed_corpus(tmp_path)
    rows = scan_corpus_chunks(tmp_path)
    clusters = cluster_chunks(rows)
    df = build_reference_df(clusters)
    # _BP_TEXT clears both gates (10 syllabi, 3 depts).
    # _SINGLE_DEPT_REPEAT does not (4 syllabi, 1 dept).
    # _UNIQUE_PREFIX + stem are all distinct.
    assert len(df) == 1
    row = df.iloc[0]
    assert row["doc_frequency"] == 10
    assert row["distinct_depts"] == 3
    assert "academic dishonesty" in row["representative_text"].lower()
    assert isinstance(row["ngram_signature"], list)
    assert len(row["ngram_signature"]) > 0


def test_build_reference_handles_empty_clusters():
    df = build_reference_df({})
    assert len(df) == 0
    # Schema columns present even when empty.
    assert list(df.columns) == [
        "cluster_id",
        "representative_text",
        "normalized_text",
        "doc_frequency",
        "distinct_depts",
        "ngram_signature",
    ]


def test_parquet_roundtrip(tmp_path: Path):
    _seed_corpus(tmp_path)
    rows = scan_corpus_chunks(tmp_path)
    clusters = cluster_chunks(rows)
    df = build_reference_df(clusters)
    out = tmp_path / "boilerplate_reference.parquet"
    df.to_parquet(out, index=False)
    rt = pd.read_parquet(out)
    assert len(rt) == len(df)
    assert list(rt.columns) == list(df.columns)
