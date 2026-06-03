"""silver_tag parquet-cache invalidation — the cache must be content-keyed so a
same-path index rebuild (the two-phase dedup workflow) is picked up, not served
stale (H2)."""

from __future__ import annotations

import pandas as pd

from tamubot.ingestion.pipeline_v6b.assets import silver_tag as mod
from tamubot.ingestion.pipeline_v6b.util.text_normalize import normalize_text, stable_cluster_id


def _write_ref(path, rep: str) -> None:
    norm = normalize_text(rep)
    pd.DataFrame(
        [
            {
                "cluster_id": stable_cluster_id(norm),
                "representative_text": rep,
                "normalized_text": norm,
                "doc_frequency": 7,
                "distinct_depts": 4,
            }
        ]
    ).to_parquet(path)


def _reset_cache() -> None:
    mod._REFERENCE_INDEX_CACHE = None
    mod._REFERENCE_INDEX_CACHED_PATH = None


def test_reference_index_cache_invalidates_on_content_change(tmp_path, monkeypatch):
    path = tmp_path / "boilerplate_reference.parquet"
    monkeypatch.setattr(mod.paths, "boilerplate_reference_path", lambda: path)
    _reset_cache()

    _write_ref(path, "first representative policy text block " * 5)
    idx1, avail1, sha1 = mod._load_reference_index()
    assert avail1

    # Same path, different content → must NOT be served from the path-only cache.
    _write_ref(path, "a totally different representative policy text block " * 5)
    idx2, avail2, sha2 = mod._load_reference_index()
    assert avail2
    assert sha2 != sha1
    assert idx2 is not idx1


def test_reference_index_cache_hits_when_content_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "boilerplate_reference.parquet"
    monkeypatch.setattr(mod.paths, "boilerplate_reference_path", lambda: path)
    _reset_cache()

    _write_ref(path, "stable policy text block " * 5)
    idx1, _, sha1 = mod._load_reference_index()
    idx2, _, sha2 = mod._load_reference_index()
    assert idx1 is idx2  # unchanged content → same cached object reused
    assert sha1 == sha2
