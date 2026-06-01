from tamubot.ingestion.pipeline_v6b.sensors import format_alert


def test_format_alert_lists_failed_checks():
    msg = format_alert([("v6b_bronze_blocks", "source_integrity", "STAT_692_x")])
    assert "source_integrity" in msg
    assert "STAT_692_x" in msg


def test_format_alert_empty_is_none():
    assert format_alert([]) is None
