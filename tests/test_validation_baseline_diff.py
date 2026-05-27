"""Tests for baseline_diff helpers."""

from tamubot.ingestion.validation.baseline_diff import compute_baseline_delta


def test_within_threshold_passes():
    out = compute_baseline_delta(
        current=100,
        history=[100, 105, 95, 110, 90],
        max_drift_pct=0.20,
    )
    assert out.passed
    assert out.metadata["baseline_median"] == 100
    assert abs(out.metadata["delta_pct"]) < 0.20


def test_above_threshold_fails():
    out = compute_baseline_delta(
        current=150,
        history=[100, 100, 100, 100, 100],
        max_drift_pct=0.20,
    )
    assert out.passed is False
    assert out.metadata["delta_pct"] == 0.50


def test_below_threshold_fails():
    out = compute_baseline_delta(
        current=50,
        history=[100, 100, 100, 100, 100],
        max_drift_pct=0.20,
    )
    assert out.passed is False
    assert out.metadata["delta_pct"] == -0.50


def test_empty_history_passes_as_first_run():
    """No history → first run → no baseline to violate."""
    out = compute_baseline_delta(current=100, history=[], max_drift_pct=0.20)
    assert out.passed
    assert out.metadata["baseline_median"] is None


def test_zero_baseline_skipped():
    """Median of 0 makes delta_pct undefined → pass with note."""
    out = compute_baseline_delta(current=10, history=[0, 0, 0], max_drift_pct=0.20)
    assert out.passed
    assert out.metadata["baseline_median"] == 0
