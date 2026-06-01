"""Alert sensor: surfaces failed v6b asset checks. format_alert() is the pure,
tested core; the sensor wraps it and emits to the configured channel (logged by
default; wire Slack/email webhook via ALERT_WEBHOOK_URL when available)."""

from __future__ import annotations

import os

from dagster import DefaultSensorStatus, RunFailureSensorContext, run_failure_sensor


def format_alert(failures: list[tuple[str, str, str]]) -> str | None:
    """failures: list of (asset, check_name, partition). Returns a message, or
    None when there is nothing to report."""
    if not failures:
        return None
    lines = [f"- {asset} / {check} [{partition}]" for asset, check, partition in failures]
    return "v6b check/run failures:\n" + "\n".join(lines)


@run_failure_sensor(default_status=DefaultSensorStatus.STOPPED)
def v6b_check_failure_alert(context: RunFailureSensorContext) -> None:
    """Logs (and optionally webhooks) on any failed run in the v6b code location.
    Default STOPPED so it ships dark; enable in the Dagster UI per environment."""
    msg = format_alert([("run", context.dagster_run.job_name or "unknown", context.dagster_run.run_id)])
    if not msg:
        return
    context.log.error(msg)
    url = os.getenv("ALERT_WEBHOOK_URL")
    if url:
        import urllib.request

        req = urllib.request.Request(url, data=msg.encode("utf-8"), headers={"Content-Type": "text/plain"})
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:  # noqa: BLE001 - alerting must never crash the sensor
            context.log.warning("alert webhook failed")
