"""Baseline-diff helpers. The pure-math helper is unit-tested; the
Dagster-coupled reader is exercised end-to-end in the UI."""

from __future__ import annotations

import statistics
from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome


def compute_baseline_delta(
    current: float,
    history: list[float],
    max_drift_pct: float = 0.20,
) -> CheckOutcome:
    """Pass iff |current - median(history)| / median(history) <= max_drift_pct.

    Special cases:
      - empty history → pass (first run, no baseline)
      - median 0 → pass (delta undefined)
    """
    if not history:
        return CheckOutcome(
            passed=True,
            metadata={"current": current, "baseline_median": None, "history_n": 0},
        )
    baseline = statistics.median(history)
    if baseline == 0:
        return CheckOutcome(
            passed=True,
            metadata={
                "current": current,
                "baseline_median": 0,
                "history_n": len(history),
                "note": "baseline median is 0; delta undefined",
            },
        )
    delta_pct = (current - baseline) / baseline
    return CheckOutcome(
        passed=abs(delta_pct) <= max_drift_pct,
        metadata={
            "current": current,
            "baseline_median": baseline,
            "delta_pct": round(delta_pct, 4),
            "max_drift_pct": max_drift_pct,
            "history_n": len(history),
        },
    )


def read_metadata_history(
    instance: Any,
    asset_key: str,
    partition_key: str,
    metadata_key: str,
    last_n: int = 5,
) -> list[float]:
    """Read the last N materialization metadata values for (asset, partition).

    Returns numeric values only — entries where the metadata key is missing or
    non-numeric are skipped (not zeroed) so they don't poison the baseline.
    """
    from dagster import AssetKey, DagsterEventType, EventRecordsFilter

    records = instance.get_event_records(
        EventRecordsFilter(
            event_type=DagsterEventType.ASSET_MATERIALIZATION,
            asset_key=AssetKey(asset_key),
            asset_partitions=[partition_key],
        ),
        limit=last_n + 1,  # +1 because the current run is in the log
        ascending=False,
    )
    values: list[float] = []
    for r in records[1:]:  # skip the most recent (= current run)
        mat = r.event_log_entry.dagster_event.event_specific_data.materialization
        entry = mat.metadata.get(metadata_key)
        if entry is None:
            continue
        try:
            values.append(float(entry.value))
        except TypeError, ValueError:
            continue
    return values
