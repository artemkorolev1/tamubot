"""Verify silver_embed writes its own output (not upstream's) and stamps
embedding_model on each chunk."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_silver_embed_writes_to_silver_embed_path(tmp_path, monkeypatch):
    """silver_embed must NOT mutate the silver_tag output file."""
    from tamubot.ingestion.pipeline_v6b import paths

    stem = "202531_FAKE_101_500_99999"
    tag_path = tmp_path / "tag.json"
    embed_path = tmp_path / "embed.json"

    tag_data = {
        "source_file": stem,
        "chunks": [
            {"crn": "99999", "chunk_index": 0, "content": "abc", "embedding": None},
        ],
    }
    tag_path.write_text(json.dumps(tag_data), encoding="utf-8")
    tag_mtime_before = tag_path.stat().st_mtime

    monkeypatch.setattr(paths, "silver_tag_path", lambda s, v="semantic": tag_path)
    monkeypatch.setattr(paths, "silver_embed_path", lambda s: embed_path)

    with patch("tamubot.ingestion.pipeline_v6b.assets.silver_embed._embed_chunks") as mock_embed:

        def _stamp(chunks):
            for c in chunks:
                c["embedding"] = [0.1, 0.2, 0.3]
                c["embedding_model"] = "voyage-3"
            return 1

        mock_embed.side_effect = _stamp

        from tamubot.ingestion.pipeline_v6b.assets.silver_embed import _compute_embed

        ctx = MagicMock()
        ctx.partition_key = stem
        _compute_embed(ctx)

    assert embed_path.exists(), "silver_embed must write to its own path"
    assert tag_path.stat().st_mtime == tag_mtime_before, "silver_tag output must not be mutated"

    out = json.loads(embed_path.read_text(encoding="utf-8"))
    assert out["chunks"][0]["embedding_model"] == "voyage-3"
    assert out["chunks"][0]["embedding"] == [0.1, 0.2, 0.3]
