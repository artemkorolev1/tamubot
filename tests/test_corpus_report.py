from tamubot.ingestion.pipeline_v6b.assets.corpus_report import build_report


def test_build_report_flags_orphans_and_volume(monkeypatch):
    report = build_report(
        disk_stems_by_dept={"STAT": {"a", "b"}},
        atlas_stems_by_dept={"STAT": {"a", "b", "c"}},
        volume_history_by_dept={"STAT": [2, 2, 2]},
    )
    stat = report["STAT"]
    assert stat["orphan_stems"] == ["c"]
    assert stat["disk_count"] == 2
    assert stat["volume_drift"]["passed"] is True  # 2 vs median 2
    assert report["_summary"]["total_orphans"] == 1
