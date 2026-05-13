"""MongoDB tool — all search and fetch operations against chunks_v3 / courses_v3.

Exposes:
  hybrid_search(query, course_id, k) -> list[dict]
  semantic_search(query, k) -> list[dict]   (hybrid: vector + BM25)
  fetch_anchor_chunks(course_ids) -> (list[dict], list[tuple[str,str]], bool)
  get_meeting_times(course_ids) -> dict[str, Any]
  get_syllabus_urls(course_ids) -> dict[str, str]
  get_missing_sections(course_id) -> list[str]

Canonical location: rag/tools/mongo.py
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from langfuse import observe
from pymongo import MongoClient

from tamubot.core import config

CHUNKS_COLLECTION = os.getenv("CHUNKS_COLLECTION", "chunks_v3")
COURSES_COLLECTION = os.getenv("COURSES_COLLECTION", "courses_v3")
VECTOR_INDEX = os.getenv("VECTOR_INDEX", "vector_index_v3")
TEXT_INDEX = os.getenv("TEXT_INDEX", "text_index_v3")


_client: Optional[MongoClient] = None


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI, tlsAllowInvalidCertificates=True)
    return _client[config.MONGODB_DB]


def _projection() -> dict:
    return {
        "$project": {
            "course_id": 1,
            "chunk_index": 1,
            "content": 1,
            "header_text": 1,
            "header_path": 1,
            "anchor": 1,
            "section": 1,
            "term": 1,
            "score": 1,
            "category": 1,
            "source": 1,
            "page": 1,
            "instructor_name": 1,
            "pipeline_version": 1,
        }
    }


def _atlas_filter(course_id: str | None, term: str | None) -> dict | None:
    f: dict = {}
    if course_id:
        f["course_id"] = course_id
    if term:
        f["term"] = term
    if ct := os.getenv("CHUNK_TAG_FILTER"):
        f["chunk_tag"] = ct
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
    if ct := os.getenv("CHUNK_TAG_FILTER"):
        text_filters.append({"equals": {"path": "chunk_tag", "value": ct}})
    if text_filters:
        compound["filter"] = text_filters
    return [
        {"$search": {"index": TEXT_INDEX, "compound": compound}},
        {"$limit": k},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        _projection(),
    ]


def _rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    scores: dict = {}
    docs: dict = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("_id", id(doc)))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs[doc_id] = doc
    return [docs[did] for did in sorted(scores, key=scores.__getitem__, reverse=True)]


@observe(name="pipeline.retrieval.search.hybrid")
def hybrid_search(query: str, course_id: str, k: int) -> list[dict]:
    """RRF hybrid search (vector + BM25) filtered to one course."""
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
    return results


@observe(name="pipeline.retrieval.search.semantic")
def semantic_search(query: str, k: int) -> list[dict]:
    """Corpus-wide hybrid search (vector + BM25)."""
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
    return results


def fetch_anchor_chunks(
    course_ids: list[str],
    categories: list[str] | None = None,
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Fetch anchor chunks for each course, optionally filtered by category.

    Args:
        course_ids: List of course IDs to fetch anchor chunks for.
        categories: Optional list of categories to filter by. If None, fetches all chunks.

    Returns (chunks, data_gaps, data_integrity).
    data_gaps = list of (course_id, section) pairs where anchor chunks are missing.
    data_integrity = False if any gaps found.
    """
    db = _get_db()
    all_chunks: list[dict] = []
    data_gaps: list[tuple[str, str]] = []

    for course_id in course_ids:
        match: dict = {"course_id": course_id}
        if categories:
            match["category"] = {"$in": categories}
        pipeline = [
            {"$match": match},
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
    """Return section names missing from course_id chunks."""
    db = _get_db()
    present = set(db[CHUNKS_COLLECTION].distinct("section", {"course_id": course_id}))
    from tamubot.rag.models import VALID_CATEGORIES

    return [s for s in VALID_CATEGORIES if s not in present]
