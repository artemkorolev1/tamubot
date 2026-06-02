"""Stage-batched v6b backfill: run each GPU-heavy stage across ALL stems before
moving to the next, so docling (bronze) and NuExtract (modal) never share the
8 GB card at the same time.

Phases:
  1. v6b_bronze_blocks        — docling owns the GPU
  2. v6b_silver_modal         — NuExtract sidecar owns the GPU
  3. structured→chunk→tag→embed→atlas — CPU / TAMU / Voyage (no GPU contention)
  4. meta indexes, then a 2nd silver_tag pass so cross-syllabus dedup populates

Each phase is one Dagster asset backfill (one run per partition), polled to
completion before the next phase starts. Run from the host:

    python scripts/v6b_staged_run.py            # all 10 pilot CSCE stems
    python scripts/v6b_staged_run.py --stems A B # explicit subset

The NuExtract sidecar must be reachable (NUEXTRACT_BACKEND=http). This script
does not start/stop it; with the stages separated you can safely run the sidecar
at a high gpu-memory-utilization since docling is finished before Phase 2.
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

DAGSTER = "http://localhost:3000/graphql"
LOCATION = "definitions.py"
REPO = "__repository__"

PILOT_STEMS = [
    "202611_CSCE_608_600_46648",
    "202611_CSCE_611_600_50668",
    "202611_CSCE_612_600_42640",
    "202611_CSCE_614_600_42743",
    "202611_CSCE_625_600_19180_HP",
    "202611_CSCE_629_600_54978",
    "202611_CSCE_633_600_12367_HP",
    "202611_CSCE_635_600_46646",
    "202611_CSCE_641_600_58437",
    "202611_CSCE_648_600_60319",
]

# Ordered phases: each entry is (label, [asset_names]).
#
# Phase 1 (bronze/docling) is kept separate and serial — docling is the GPU hog
# and must not coexist with NuExtract on a small card.
#
# Phase 2 fuses modal with all downstream stages so each document flows
# modal -> structured -> chunk -> tag -> embed -> atlas without a phase barrier.
# The downstream stages are CPU/TAMU/Voyage (no GPU), so when this phase runs
# with run concurrency > 1 (set QueuedRunCoordinator max_concurrent_runs > 1 and
# the sidecar's --max-num-seqs to match), one document's cheap tail overlaps the
# next document's modal AND concurrent modal calls batch on the warm sidecar.
# Concurrency is safe here only because bronze (the GPU hog) already finished:
# extra runs are just HTTP clients to the one sidecar, adding no GPU memory.
PHASES: list[tuple[str, list[str]]] = [
    ("bronze (docling, serial)", ["v6b_bronze_blocks"]),
    (
        "modal+downstream (sidecar-batched; run concurrent)",
        [
            "v6b_silver_modal",
            "v6b_silver_structured",
            "v6b_silver_chunk_semantic",
            "v6b_silver_tag_semantic",
            "v6b_silver_embed",
            "v6b_silver_atlas_upsert",
        ],
    ),
]

_LAUNCH = """
mutation L($bp: LaunchBackfillParams!) {
  launchPartitionBackfill(backfillParams: $bp) {
    __typename
    ... on LaunchBackfillSuccess { backfillId }
    ... on PythonError { message }
  }
}
"""

_STATUS = """
query S($id: String!) {
  partitionBackfillOrError(backfillId: $id) {
    ... on PartitionBackfill { status }
  }
}
"""


def _gql(query: str, variables: dict) -> dict:
    r = requests.post(DAGSTER, json={"query": query, "variables": variables}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(body["errors"])
    return body["data"]


def launch_phase(assets: list[str], stems: list[str]) -> str:
    bp = {
        "assetSelection": [{"path": [a]} for a in assets],
        "partitionNames": stems,
    }
    data = _gql(_LAUNCH, {"bp": bp})
    res = data["launchPartitionBackfill"]
    if res["__typename"] != "LaunchBackfillSuccess":
        raise RuntimeError(f"launch failed: {res}")
    return res["backfillId"]


def wait(backfill_id: str, poll_s: int = 30) -> str:
    while True:
        status = _gql(_STATUS, {"id": backfill_id})["partitionBackfillOrError"]["status"]
        print(f"    [{time.strftime('%H:%M:%S')}] {backfill_id} = {status}", flush=True)
        if status in ("COMPLETED_SUCCESS", "COMPLETED_FAILED", "CANCELED"):
            return status
        time.sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=PILOT_STEMS)
    args = ap.parse_args()
    stems = args.stems

    print(f"staged run over {len(stems)} stems", flush=True)
    for label, assets in PHASES:
        print(f"\n=== phase: {label} -> {assets} ===", flush=True)
        bid = launch_phase(assets, stems)
        status = wait(bid)
        if status != "COMPLETED_SUCCESS":
            print(f"phase '{label}' ended {status}; stopping.", flush=True)
            return 1

    print("\nAll phases done. Now materialize meta + a 2nd tag pass:", flush=True)
    print("  - v6b_meta_chunk_signature_index, v6b_meta_boilerplate_reference, v6b_corpus_report", flush=True)
    print("  - then re-run v6b_silver_tag_semantic so cross-syllabus dedup populates", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
