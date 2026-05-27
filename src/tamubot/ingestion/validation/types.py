"""Shared types for ingestion validation helpers.

All helpers in this package return a CheckOutcome. Dagster check decorators in
pipeline_v6{b,c}/checks/ are thin adapters that wrap CheckOutcome in
AssetCheckResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckOutcome:
    """Result of a validation check. Pure data, no Dagster coupling."""

    passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)
