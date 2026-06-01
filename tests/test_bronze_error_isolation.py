
from tamubot.ingestion.pipeline_v6b.assets import bronze_blocks as bb


def test_parse_failure_writes_deadletter(tmp_path, monkeypatch):
    pdf = tmp_path / "stem.pdf"
    pdf.write_bytes(b"%PDF-broken")
    dead = tmp_path / "stem.error.json"
    monkeypatch.setattr(bb.paths, "raw_path", lambda stem: pdf)
    monkeypatch.setattr(bb.paths, "bronze_blocks_path", lambda stem: tmp_path / "stem.blocks.json")
    monkeypatch.setattr(bb, "_deadletter_path", lambda stem: dead)

    def boom(**kwargs):
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(bb, "_run_docling", boom)
    import pytest

    with pytest.raises(RuntimeError):
        bb._parse_or_deadletter("stem", pdf, tmp_path)
    assert dead.exists()
    assert "docling exploded" in dead.read_text()
