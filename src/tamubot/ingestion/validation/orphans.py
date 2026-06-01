"""Orphan detection for v6b: Atlas vectors whose source stem no longer exists
on disk (report-only — never deletes; honors the no-delete-without-asking
policy)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrphanResult:
    orphan_stems: list[str]  # in Atlas, not on disk
    missing_from_atlas: list[str]  # on disk, not in Atlas


def compute_orphans(disk_stems: set[str], atlas_stems: set[str]) -> OrphanResult:
    return OrphanResult(
        orphan_stems=sorted(atlas_stems - disk_stems),
        missing_from_atlas=sorted(disk_stems - atlas_stems),
    )
