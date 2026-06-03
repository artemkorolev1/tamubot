"""silver_tag: three-pass tagger over semantic chunks.

Pass 1 — Boilerplate via the shared reference parquet
(`paths.boilerplate_reference_path()`).
Pass 2 — Within-syllabus dedup (MinHash-LSH at Jaccard >= WITHIN_SYL_THRESHOLD).
Pass 3 — Cross-syllabus dedup against the signature-index parquet
(`paths.chunk_signature_index_path()`) at Jaccard >= CROSS_SYL_THRESHOLD.

Never drops chunks (enforced by `v6b_silver_tag_chunk_count_preserved`). When
either parquet is missing, that pass is a no-op and the metadata flags it.

Pure logic lives in util/tagging.py; this file is the asset wrapper + parquet
memoization.
"""

import json

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)
from datasketch import MinHash, MinHashLSH

from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.pipeline_v6b.util.tagging import (
    load_cross_syllabus_index_from_parquet,
    tag_chunks,
)
from tamubot.ingestion.pipeline_v6b.util.text_normalize import ReferenceIndex

# Module-level cache so a Dagster run doesn't reread parquets per-partition.
_REFERENCE_INDEX_CACHE: ReferenceIndex | None = None
_REFERENCE_INDEX_CACHED_PATH: str | None = None
_CROSS_SYL_CACHE: tuple[MinHashLSH, dict[str, MinHash], dict[str, dict]] | None = None
_CROSS_SYL_CACHED_PATH: str | None = None


def _load_reference_index() -> tuple[ReferenceIndex, bool, str]:
    global _REFERENCE_INDEX_CACHE, _REFERENCE_INDEX_CACHED_PATH
    parquet = paths.boilerplate_reference_path()
    if not parquet.exists():
        return ReferenceIndex.empty(), False, ""
    # Key the cache on (path, content-hash) so a same-path parquet rebuild — the
    # two-phase dedup workflow — invalidates a stale index instead of serving it.
    sha = hash_file(parquet)
    key = f"{parquet}@{sha}"
    if _REFERENCE_INDEX_CACHE is not None and _REFERENCE_INDEX_CACHED_PATH == key:
        return _REFERENCE_INDEX_CACHE, True, sha
    idx = ReferenceIndex(parquet)
    _REFERENCE_INDEX_CACHE = idx
    _REFERENCE_INDEX_CACHED_PATH = key
    return idx, True, sha


def _load_cross_syllabus_index() -> tuple[
    "MinHashLSH | None", dict[str, MinHash], dict[str, dict], bool, str
]:
    global _CROSS_SYL_CACHE, _CROSS_SYL_CACHED_PATH
    parquet = paths.chunk_signature_index_path()
    if not parquet.exists():
        return None, {}, {}, False, ""
    # (path, content-hash) key — see _load_reference_index.
    sha = hash_file(parquet)
    key = f"{parquet}@{sha}"
    if _CROSS_SYL_CACHE is not None and _CROSS_SYL_CACHED_PATH == key:
        lsh, sigs, metadata = _CROSS_SYL_CACHE
        return lsh, sigs, metadata, True, sha
    lsh, sigs, metadata = load_cross_syllabus_index_from_parquet(parquet)
    _CROSS_SYL_CACHE = (lsh, sigs, metadata)
    _CROSS_SYL_CACHED_PATH = key
    return lsh, sigs, metadata, True, sha


def _compute_silver_tag(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)

    src_path = paths.silver_chunk_semantic_path(stem)
    dst_path = paths.silver_tag_path(stem, "semantic")

    data = json.loads(src_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    ref_index, ref_available, ref_sha = _load_reference_index()
    cross_lsh, cross_sigs, cross_metadata, cross_available, cross_sha = _load_cross_syllabus_index()

    tagged, stats = tag_chunks(chunks, stem, ref_index, cross_lsh, cross_sigs, cross_metadata)
    data["chunks"] = tagged

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(tagged)
    bp_rate = stats["tagged_boilerplate"] / total if total else 0.0
    dup_rate = stats["tagged_duplicate"] / total if total else 0.0

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "boilerplate_reference_available": ref_available,
            "boilerplate_reference_sha256": ref_sha,
            "cross_syllabus_index_available": cross_available,
            "cross_syllabus_index_sha256": cross_sha,
            "tagged_boilerplate": stats["tagged_boilerplate"],
            "tagged_duplicate": stats["tagged_duplicate"],
            "boilerplate_rate": round(bp_rate, 4),
            "duplicate_rate": round(dup_rate, 4),
            "total_chunks": total,
            "output_path": MetadataValue.path(str(dst_path)),
            "sha256": hash_file(dst_path),
        }
    )


silver_tag_semantic = asset(
    name="v6b_silver_tag_semantic",
    deps=[AssetKey("v6b_silver_chunk_semantic")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_tag),
    group_name="v6b_silver",
    description="Three-pass boilerplate + within/cross-syllabus dedup tagger. Never drops chunks.",
)(_compute_silver_tag)
