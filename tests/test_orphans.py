from tamubot.ingestion.validation.orphans import compute_orphans


def test_orphans_are_atlas_minus_disk():
    res = compute_orphans(disk_stems={"a", "b"}, atlas_stems={"a", "b", "c"})
    assert res.orphan_stems == ["c"]
    assert res.missing_from_atlas == []


def test_missing_from_atlas():
    res = compute_orphans(disk_stems={"a", "b"}, atlas_stems={"a"})
    assert res.orphan_stems == []
    assert res.missing_from_atlas == ["b"]
