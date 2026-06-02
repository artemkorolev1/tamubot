"""Programmatic mirror of docs/preprocessing_error_taxonomy.md (v1.0.0).

Single source of truth for code that needs the enum (the report ranker, finding
validation). Keep in sync with the markdown doc and prompts.py when the version bumps.
"""

from __future__ import annotations

TAXONOMY_VERSION = "1.0.0"
SEVERITY_WEIGHT = {"blocker": 3, "major": 2, "minor": 1}

# type -> (dimension, severity, [owning code files])
ERROR_TYPES: dict[str, tuple[str, str, list[str]]] = {
    # boilerplate
    "BP_MISSED": ("boilerplate", "major", ["util/tagging.py", "util/boilerplate_clustering.py"]),
    "BP_OVERREACH": ("boilerplate", "blocker", ["util/tagging.py", "util/boilerplate_clustering.py"]),
    "BP_WRONG_CLUSTER": ("boilerplate", "minor", ["util/text_normalize.py"]),
    # dedup
    "DUP_MISSED": ("dedup", "minor", ["util/tagging.py", "util/signature_index.py"]),
    "DUP_FALSE": ("dedup", "blocker", ["util/tagging.py"]),
    "DUP_WRONG_CANON": ("dedup", "major", ["util/tagging.py"]),
    # chunking
    "CHUNK_ORPHAN_HEADER": ("chunking", "major", ["ingestion/chunker_v4.py"]),
    "CHUNK_OVERSIZED": ("chunking", "major", ["ingestion/chunker_v4.py"]),
    "CHUNK_FRAGMENTED": ("chunking", "minor", ["ingestion/chunker_v4.py"]),
    "CHUNK_MIDSENTENCE": ("chunking", "minor", ["ingestion/chunker_v4.py"]),
    "CHUNK_MERGED_TOPICS": ("chunking", "minor", ["ingestion/chunker_v4.py"]),
    # fidelity
    "FID_TABLE_LOST": ("fidelity", "blocker", ["assets/silver_modal.py", "assets/bronze_blocks.py"]),
    "FID_IMAGE_LOST": ("fidelity", "minor", ["assets/silver_modal.py"]),
    "FID_HEADER_BROKEN": ("fidelity", "major", ["assets/bronze_blocks.py"]),
    "FID_CONTENT_DROPPED": ("fidelity", "blocker", ["assets/bronze_blocks.py"]),
    "FID_HALLUCINATION": ("fidelity", "blocker", ["assets/silver_modal.py"]),
    "FID_REPLACEMENT_CHARS": ("fidelity", "minor", ["assets/bronze_blocks.py"]),
}

DIMENSIONS = ["boilerplate", "dedup", "chunking", "fidelity"]


def is_valid_type(t: str) -> bool:
    return t in ERROR_TYPES


def owners(t: str) -> list[str]:
    return ERROR_TYPES.get(t, ("", "", []))[2]


def severity(t: str) -> str:
    return ERROR_TYPES.get(t, ("", "minor", []))[1]
