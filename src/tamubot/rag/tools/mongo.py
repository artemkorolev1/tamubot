"""MongoDB tool — all search and fetch operations against chunks_v4 / courses_v4.

Exposes:
  hybrid_search(query, course_id, k) -> list[dict]
  semantic_search(query, k) -> list[dict]   (hybrid: vector + BM25)
  fetch_anchor_chunks(course_ids) -> (list[dict], list[tuple[str,str]], bool)
  get_meeting_times(course_ids) -> dict[str, Any]
  get_syllabus_urls(course_ids) -> dict[str, str]
  get_course_summaries(course_ids) -> dict[str, str]
  get_course_summary_chunks(course_ids) -> list[dict]

Canonical location: rag/tools/mongo.py
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from langfuse import observe
from pymongo import MongoClient

from tamubot.core import config

CHUNKS_COLLECTION = os.getenv("CHUNKS_COLLECTION", "chunks_v4")
COURSES_COLLECTION = os.getenv("COURSES_COLLECTION", "courses_v4")
VECTOR_INDEX = os.getenv("VECTOR_INDEX", "vector_index_v4")
TEXT_INDEX = os.getenv("TEXT_INDEX", "text_index_v4")


_client: Optional[MongoClient] = None


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI, tlsAllowInvalidCertificates=True)
    return _client[config.MONGODB_DB]


def get_db():
    """Public accessor for the shared MongoDB database handle (lazy singleton)."""
    return _get_db()


def _projection() -> dict:
    return {
        "$project": {
            "course_id": 1,
            "chunk_index": 1,
            "content": 1,
            "header_path": 1,
            "anchor": 1,
            "section": 1,
            "term": 1,
            "score": 1,
            "source": 1,
            "page": 1,
            "source_file": 1,
            "instructor_name": 1,
            "pipeline_version": 1,
            "has_table": 1,
            "flags": 1,
            "split_reason": 1,
            "token_count": 1,
        }
    }


def _atlas_filter(course_id: str | None, term: str | None) -> dict | None:
    f: dict = {}
    if course_id:
        f["course_id"] = course_id
    if term:
        f["term"] = term
    chunk_tag = os.getenv("CHUNK_TAG_FILTER")
    if chunk_tag:
        f["chunk_tag"] = chunk_tag
    # v6: default-exclude boilerplate. `$ne: True` also matches docs missing
    # the field (pre-v6 / pre-backfill chunks), so this is back-compat-safe.
    if not config.INCLUDE_BOILERPLATE:
        f["is_boilerplate"] = {"$ne": True}
    # v6b Phase 2: default-exclude duplicates. Same back-compat pattern.
    if not config.INCLUDE_DUPLICATE:
        f["is_duplicate"] = {"$ne": True}
    return f if f else None


def _build_vector_stage(embedding: list[float], k: int, filters: dict | None) -> dict:
    stage: dict = {
        "$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": k * 10,
            "limit": k,
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters
    return stage


def _build_text_stage(query: str, k: int, course_id: str | None) -> list[dict]:
    compound: dict = {"must": [{"text": {"query": query, "path": "content"}}]}
    text_filters = []
    if course_id:
        text_filters.append({"equals": {"path": "course_id", "value": course_id}})
    chunk_tag = os.getenv("CHUNK_TAG_FILTER")
    if chunk_tag:
        text_filters.append({"equals": {"path": "chunk_tag", "value": chunk_tag}})
    if text_filters:
        compound["filter"] = text_filters
    # v6: default-exclude boilerplate via mustNot. Docs missing the field are kept.
    mustNot: list[dict] = []
    if not config.INCLUDE_BOILERPLATE:
        mustNot.append({"equals": {"path": "is_boilerplate", "value": True}})
    # v6b Phase 2: default-exclude duplicates.
    if not config.INCLUDE_DUPLICATE:
        mustNot.append({"equals": {"path": "is_duplicate", "value": True}})
    if mustNot:
        compound["mustNot"] = mustNot
    return [
        {"$search": {"index": TEXT_INDEX, "compound": compound}},
        {"$limit": k},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        _projection(),
    ]


def _rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Tags each doc with ``rrf_source``: ``"vector_only"``, ``"bm25_only"``, or ``"both"``.
    Expects exactly two result lists: [vector_results, text_results].
    """
    scores: dict = {}
    docs: dict = {}
    sources: dict[str, set[str]] = {}
    source_labels = ["vector", "bm25"]
    for list_idx, results in enumerate(result_lists):
        label = source_labels[list_idx] if list_idx < len(source_labels) else f"list_{list_idx}"
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("_id", id(doc)))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs[doc_id] = doc
            sources.setdefault(doc_id, set()).add(label)
    fused = []
    for did in sorted(scores, key=scores.__getitem__, reverse=True):
        doc = docs[did]
        s = sources.get(did, set())
        doc["rrf_source"] = "both" if len(s) > 1 else next(iter(s))
        fused.append(doc)
    return fused


@observe(name="node.retrieval.search.hybrid")
def hybrid_search(query: str, course_id: str, k: int, *, with_meta: bool = False):
    """RRF hybrid search (vector + BM25) filtered to one course.

    Returns the fused chunk list. When ``with_meta=True``, returns a
    ``(results, meta)`` tuple where ``meta`` is
    ``{"vector_raw": int, "bm25_raw": int, "fused": int}``.
    """
    from langfuse import get_client as _lf

    from tamubot.rag.tools.voyage import embed_query

    db = _get_db()
    emb = embed_query(query)
    atlas_f = _atlas_filter(course_id, None)
    vector_pipeline = [
        _build_vector_stage(emb, k, atlas_f),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    text_pipeline = _build_text_stage(query, k, course_id)
    try:
        vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))
    except Exception:
        vector_results = []
    try:
        text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline))
    except Exception:
        text_results = []
    results = _rrf_fuse([vector_results, text_results])[:k]
    for r in results:
        r.pop("_id", None)

    # Trace metadata: search source breakdown
    source_counts = {"vector": 0, "bm25": 0, "both": 0}
    for r in results:
        src = r.get("rrf_source", "unknown")
        if src in source_counts:
            source_counts[src] += 1
    try:
        _lf().update_current_span(
            metadata={
                "course_id": course_id,
                "k": k,
                "vector_results": len(vector_results),
                "bm25_results": len(text_results),
                "fused_results": len(results),
                "source_breakdown": source_counts,
            }
        )
    except Exception:
        pass

    if with_meta:
        meta = {
            "vector_only": source_counts["vector"],
            "bm25_only": source_counts["bm25"],
            "overlap": source_counts["both"],
            "total": len(results),
        }
        return results, meta
    return results


@observe(name="node.retrieval.search.semantic")
def semantic_search(query: str, k: int, *, with_meta: bool = False):
    """Corpus-wide hybrid search (vector + BM25).

    Returns the fused chunk list. When ``with_meta=True``, returns a
    ``(results, meta)`` tuple where ``meta`` is
    ``{"vector_raw": int, "bm25_raw": int, "fused": int}``.
    """
    from langfuse import get_client as _lf

    from tamubot.rag.tools.voyage import embed_query

    db = _get_db()
    emb = embed_query(query)
    vector_pipeline = [
        _build_vector_stage(emb, k, filters=None),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    text_pipeline = _build_text_stage(query, k, course_id=None)
    try:
        vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))
    except Exception:
        vector_results = []
    try:
        text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline))
    except Exception:
        text_results = []
    results = _rrf_fuse([vector_results, text_results])[:k]
    for r in results:
        r.pop("_id", None)

    # Trace metadata: search source breakdown
    source_counts = {"vector": 0, "bm25": 0, "both": 0}
    for r in results:
        src = r.get("rrf_source", "unknown")
        if src in source_counts:
            source_counts[src] += 1
    try:
        _lf().update_current_span(
            metadata={
                "k": k,
                "vector_results": len(vector_results),
                "bm25_results": len(text_results),
                "fused_results": len(results),
                "source_breakdown": source_counts,
            }
        )
    except Exception:
        pass

    if with_meta:
        meta = {
            "vector_only": source_counts["vector"],
            "bm25_only": source_counts["bm25"],
            "overlap": source_counts["both"],
            "total": len(results),
        }
        return results, meta
    return results


def fetch_anchor_chunks(
    course_ids: list[str],
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Fetch all chunks for each course (v4 has no category field).

    Returns (chunks, data_gaps, data_integrity).
    data_gaps = list of (course_id, "anchor") pairs where no chunks were found.
    data_integrity = False if any gaps found.
    """
    db = _get_db()
    all_chunks: list[dict] = []
    data_gaps: list[tuple[str, str]] = []

    for course_id in course_ids:
        pipeline = [
            {"$match": {"course_id": course_id}},
            {"$sort": {"chunk_index": 1}},
            {"$limit": 30},
            _projection(),
        ]
        results = list(db[CHUNKS_COLLECTION].aggregate(pipeline))
        for r in results:
            r.pop("_id", None)
        if not results:
            data_gaps.append((course_id, "anchor"))
        else:
            all_chunks.extend(results)

    data_integrity = len(data_gaps) == 0
    return all_chunks, data_gaps, data_integrity


@lru_cache(maxsize=128)
def _get_meeting_times_cached(course_ids: tuple[str, ...]) -> dict[str, Any]:
    db = _get_db()
    result: dict[str, Any] = {}
    if not course_ids:
        return result
    docs = db[COURSES_COLLECTION].find(
        {"course_id": {"$in": list(course_ids)}},
        {"course_id": 1, "meeting_times": 1, "_id": 0},
    )
    for doc in docs:
        cid = doc["course_id"]
        mt = doc.get("meeting_times")
        if cid not in result or (mt is not None and result[cid] is None):
            result[cid] = mt
    for cid in course_ids:
        result.setdefault(cid, None)
    return result


def get_meeting_times(course_ids: list[str]) -> dict[str, Any]:
    """Return {course_id: meeting_times_string} for the given course IDs.

    Courses with no meeting_times field (async/online) map to None.
    When multiple sections exist for the same course_id, the first non-None value wins.
    """
    if not course_ids:
        return {}
    return dict(_get_meeting_times_cached(tuple(sorted(set(course_ids)))))


@lru_cache(maxsize=128)
def _get_syllabus_urls_cached(course_ids: tuple[str, ...]) -> dict[str, str]:
    db = _get_db()
    result: dict[str, str] = {}
    if not course_ids:
        return result
    docs = db[COURSES_COLLECTION].find(
        {"course_id": {"$in": list(course_ids)}},
        {"course_id": 1, "syllabus_url": 1, "_id": 0},
    )
    for doc in docs:
        url = doc.get("syllabus_url")
        if url:
            result[doc["course_id"]] = url
    return result


def get_syllabus_urls(course_ids: list[str]) -> dict[str, str]:
    """Return {course_id: syllabus_url} for the given course IDs."""
    if not course_ids:
        return {}
    return dict(_get_syllabus_urls_cached(tuple(sorted(set(course_ids)))))


@lru_cache(maxsize=128)
def _get_course_summaries_cached(course_ids: tuple[str, ...]) -> dict[str, str]:
    db = _get_db()
    result: dict[str, str] = {}
    if not course_ids:
        return result
    docs = db[COURSES_COLLECTION].find(
        {"course_id": {"$in": list(course_ids)}, "course_summary": {"$exists": True, "$ne": None}},
        {"course_id": 1, "course_summary": 1, "_id": 0},
    )
    for doc in docs:
        if doc.get("course_summary"):
            result[doc["course_id"]] = doc["course_summary"]
    return result


def get_course_summaries(course_ids: list[str]) -> dict[str, str]:
    """Return {course_id: course_summary} for courses that have summaries."""
    if not course_ids:
        return {}
    return dict(_get_course_summaries_cached(tuple(sorted(set(course_ids)))))


def get_missing_sections(course_id: str) -> list[str]:
    """Return header_path values present for a course (v4 has no fixed categories)."""
    db = _get_db()
    return sorted(db[CHUNKS_COLLECTION].distinct("header_path", {"course_id": course_id}))


@observe(name="node.retrieval.course_summary")
def get_course_summary_chunks(course_ids: list[str]) -> list[dict]:
    """Fetch course summaries and format as pseudo-chunks for the retrieval node.

    Returns one pseudo-chunk per course with ``source="course_summary"``
    so downstream nodes (generator, context assembly) can identify them.
    Carries ``source_file`` so the citation rewriter can link [Source N] to
    the syllabus PDF (whole-document link; no page anchor).
    """
    from langfuse import get_client as _lf

    requested = list(set(course_ids or []))
    if not course_ids:
        try:
            _lf().update_current_span(metadata={"requested_ids": [], "found_ids": [], "missing_ids": []})
        except Exception:
            pass
        return []
    db = _get_db()
    docs = db[COURSES_COLLECTION].find(
        {
            "course_id": {"$in": requested},
            "course_summary": {"$exists": True, "$ne": None},
        },
        {"course_id": 1, "course_summary": 1, "source_file": 1, "_id": 0},
    )
    chunks: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        summary = doc.get("course_summary")
        cid = doc.get("course_id")
        if not summary or cid in seen:
            continue
        seen.add(cid)
        chunks.append(
            {
                "course_id": cid,
                "content": summary,
                "header_path": "Course Overview",
                "section": "",
                "source": "course_summary",
                "source_file": doc.get("source_file"),
                "chunk_index": 0,
                "pipeline_version": "v4",
            }
        )
    found = sorted(seen)
    try:
        _lf().update_current_span(
            metadata={
                "requested_ids": requested,
                "found_ids": found,
                "missing_ids": sorted(set(requested) - set(found)),
            }
        )
    except Exception:
        pass
    return chunks
