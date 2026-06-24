"""Postgres (pgvector) retrieval tool — the Layer-1 query primitives.

Mirrors ``rag/tools/mongo.py`` one-for-one so the retrieval node, context
assembly, and the citation rewriter are untouched: every function returns the
**same chunk dict shape** Mongo produced, and hybrid search reuses Mongo's
``_rrf_fuse`` + the unchanged Voyage rerank.

  hybrid_search(query, course_id, k)   -> list[dict]   (vector + FTS, RRF-fused)
  semantic_search(query, k)            -> list[dict]   (corpus-wide hybrid)
  fetch_anchor_chunks(course_ids)      -> (list[dict], list[tuple[str,str]], bool)
  get_meeting_times(course_ids)        -> dict[str, Any]
  get_syllabus_urls(course_ids)        -> dict[str, str]
  get_course_summaries(course_ids)     -> dict[str, str]
  get_course_summary_chunks(course_ids)-> list[dict]

The vector half ranks by cosine distance (``embedding <=> q``, matching the
Atlas cosine index); the lexical half ranks by ``ts_rank_cd`` over the
``content_tsv`` GIN index (the BM25-flavored analogue of Atlas ``$search``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langfuse import observe

from tamubot.core import config
from tamubot.ingestion.postgres.pool import get_pool

# Reuse Mongo's fusion verbatim so the two backends rank identically.
from tamubot.rag.tools.mongo import _rrf_fuse

# Columns projected into the chunk dict — mirrors mongo._projection() so
# downstream consumers see an identical shape. ``id`` is carried as ``_id`` only
# for RRF dedup and popped before returning (exactly as Mongo does).
_SELECT_COLS = (
    "id, course_id, chunk_index, content, header_path, anchor, section, term, "
    "source, page, source_file, instructor_name, has_table, flags, split_reason, token_count"
)


def _row_to_chunk(row: dict, score: float) -> dict:
    """Map a SELECT row to the canonical chunk dict (Mongo-compatible shape)."""
    return {
        "_id": row["id"],
        "course_id": row.get("course_id"),
        "chunk_index": row.get("chunk_index"),
        "content": row.get("content"),
        "header_path": row.get("header_path"),
        "anchor": row.get("anchor"),
        "section": row.get("section"),
        "term": row.get("term"),
        "score": score,
        "source": row.get("source"),
        "page": row.get("page"),
        "source_file": row.get("source_file"),
        "instructor_name": row.get("instructor_name"),
        "pipeline_version": "v4",
        "has_table": row.get("has_table"),
        "flags": row.get("flags"),
        "split_reason": row.get("split_reason"),
        "token_count": row.get("token_count"),
    }


def _filter_clauses(course_id: str | None, term: str | None) -> tuple[list[str], dict]:
    """Build the shared WHERE conditions + params (course/term/tag + boilerplate/dup).

    Mirrors ``mongo._atlas_filter`` / the ``$search`` mustNot logic:
      * default-exclude boilerplate / duplicates (``IS NOT TRUE`` also keeps
        rows where the flag is NULL — the back-compat-safe analogue of
        Mongo's ``$ne: True``).
      * honor ``CHUNK_TAG_FILTER`` when set.
    """
    conds: list[str] = []
    params: dict[str, Any] = {}
    if course_id:
        conds.append("course_id = %(course_id)s")
        params["course_id"] = course_id
    if term:
        conds.append("term = %(term)s")
        params["term"] = term
    # Default to the production 'semantic' tag: the table stores BOTH 'semantic'
    # and '600t_100o_fixed' variants per chunk, so an unfiltered search would
    # double-count. Set CHUNK_TAG_FILTER="" to search every tag (experiments).
    chunk_tag = os.getenv("CHUNK_TAG_FILTER", "semantic")
    if chunk_tag:
        conds.append("chunk_tag = %(chunk_tag)s")
        params["chunk_tag"] = chunk_tag
    if not config.INCLUDE_BOILERPLATE:
        conds.append("is_boilerplate IS NOT TRUE")
    if not config.INCLUDE_DUPLICATE:
        conds.append("is_duplicate IS NOT TRUE")
    return conds, params


# Filtered ANN tuning. The single global HNSW is post-filtered by
# course_id/doc_type/chunk_tag/boilerplate, so without iterative scan a
# selective filter can exhaust the default ef_search(40) candidate pool and
# under-return. relaxed_order is the right trade for an RRF hybrid that re-ranks.
_HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
_HNSW_ITERATIVE_SCAN = os.getenv("HNSW_ITERATIVE_SCAN", "relaxed_order")


def _vector_results(conn, embedding: list[float], k: int, course_id: str | None, term: str | None) -> list[dict]:
    conds, params = _filter_clauses(course_id, term)
    conds.append("embedding IS NOT NULL")
    params["q"] = embedding
    params["k"] = k
    where = " WHERE " + " AND ".join(conds)
    # SET LOCAL scopes these to the current transaction only (pool-safe).
    conn.execute(f"SET LOCAL hnsw.iterative_scan = {_HNSW_ITERATIVE_SCAN}")
    conn.execute(f"SET LOCAL hnsw.ef_search = {_HNSW_EF_SEARCH}")
    sql = (
        f"SELECT {_SELECT_COLS}, 1 - (embedding <=> %(q)s::vector) AS _score "
        f"FROM chunks{where} ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s"
    )
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_chunk(r, float(r["_score"])) for r in rows]


# OR-semantics tsquery: plainto_tsquery ANDs terms ('grade & calcul'), but Atlas
# $search ranks any-term matches (BM25 OR). Rewrite '&' → '|' so a chunk matching
# *any* query lexeme is a candidate, ranked by ts_rank_cd. Empty query (all
# stopwords) yields ''::tsquery, which matches nothing — same as Mongo.
_TSQUERY_EXPR = "replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery"


def _text_results(conn, query: str, k: int, course_id: str | None, term: str | None) -> list[dict]:
    conds, params = _filter_clauses(course_id, term)
    conds.append(f"content_tsv @@ {_TSQUERY_EXPR}")
    params["query"] = query
    params["k"] = k
    where = " WHERE " + " AND ".join(conds)
    sql = (
        f"SELECT {_SELECT_COLS}, ts_rank_cd(content_tsv, {_TSQUERY_EXPR}) AS _score "
        f"FROM chunks{where} ORDER BY _score DESC LIMIT %(k)s"
    )
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_chunk(r, float(r["_score"])) for r in rows]


def _hybrid(query: str, k: int, course_id: str | None, *, with_meta: bool):
    from tamubot.rag.tools.voyage import embed_query

    from psycopg.rows import dict_row

    emb = embed_query(query)
    pool = get_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            vector_results = _vector_results(conn, emb, k, course_id, None)
        except Exception:
            vector_results = []
        try:
            text_results = _text_results(conn, query, k, course_id, None)
        except Exception:
            text_results = []
    results = _rrf_fuse([vector_results, text_results])[:k]
    for r in results:
        r.pop("_id", None)

    source_counts = {"vector": 0, "bm25": 0, "both": 0}
    for r in results:
        src = r.get("rrf_source", "unknown")
        if src in source_counts:
            source_counts[src] += 1
    _emit_trace_meta(course_id, k, len(vector_results), len(text_results), len(results), source_counts)

    if with_meta:
        meta = {
            "vector_only": source_counts["vector"],
            "bm25_only": source_counts["bm25"],
            "overlap": source_counts["both"],
            "total": len(results),
        }
        return results, meta
    return results


def _emit_trace_meta(course_id, k, n_vec, n_text, n_fused, source_counts) -> None:
    try:
        from langfuse import get_client as _lf

        md = {
            "k": k,
            "vector_results": n_vec,
            "bm25_results": n_text,
            "fused_results": n_fused,
            "source_breakdown": source_counts,
            "backend": "postgres",
        }
        if course_id is not None:
            md["course_id"] = course_id
        _lf().update_current_span(metadata=md)
    except Exception:
        pass


@observe(name="node.retrieval.search.hybrid")
def hybrid_search(query: str, course_id: str, k: int, *, with_meta: bool = False):
    """RRF hybrid search (vector + FTS) filtered to one course."""
    return _hybrid(query, k, course_id, with_meta=with_meta)


@observe(name="node.retrieval.search.semantic")
def semantic_search(query: str, k: int, *, with_meta: bool = False):
    """Corpus-wide RRF hybrid search (vector + FTS)."""
    return _hybrid(query, k, None, with_meta=with_meta)


def fetch_anchor_chunks(
    course_ids: list[str],
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Fetch up to 30 chunks per course, ordered by chunk_index.

    Returns (chunks, data_gaps, data_integrity) — mirrors mongo.fetch_anchor_chunks.
    """
    from psycopg.rows import dict_row

    all_chunks: list[dict] = []
    data_gaps: list[tuple[str, str]] = []
    pool = get_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        for course_id in course_ids:
            rows = conn.execute(
                f"SELECT {_SELECT_COLS} FROM chunks WHERE course_id = %s "
                "ORDER BY chunk_index LIMIT 30",
                (course_id,),
            ).fetchall()
            results = [_row_to_chunk(r, 0.0) for r in rows]
            for r in results:
                r.pop("_id", None)
            if not results:
                data_gaps.append((course_id, "anchor"))
            else:
                all_chunks.extend(results)
    return all_chunks, data_gaps, len(data_gaps) == 0


@lru_cache(maxsize=128)
def _get_meeting_times_cached(course_ids: tuple[str, ...]) -> dict[str, Any]:
    from psycopg.rows import dict_row

    result: dict[str, Any] = {}
    if not course_ids:
        return result
    pool = get_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            "SELECT course_id, meeting_times FROM sections WHERE course_id = ANY(%s)",
            (list(course_ids),),
        ).fetchall()
    for r in rows:
        cid, mt = r["course_id"], r["meeting_times"]
        if cid not in result or (mt is not None and result[cid] is None):
            result[cid] = mt
    for cid in course_ids:
        result.setdefault(cid, None)
    return result


def get_meeting_times(course_ids: list[str]) -> dict[str, Any]:
    """Return {course_id: meeting_times}; first non-None section wins; missing → None."""
    if not course_ids:
        return {}
    return dict(_get_meeting_times_cached(tuple(sorted(set(course_ids)))))


@lru_cache(maxsize=128)
def _get_syllabus_urls_cached(course_ids: tuple[str, ...]) -> dict[str, str]:
    from psycopg.rows import dict_row

    result: dict[str, str] = {}
    if not course_ids:
        return result
    pool = get_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            "SELECT course_id, syllabus_url FROM sections WHERE course_id = ANY(%s)",
            (list(course_ids),),
        ).fetchall()
    for r in rows:
        if r["syllabus_url"]:
            result[r["course_id"]] = r["syllabus_url"]
    return result


def get_syllabus_urls(course_ids: list[str]) -> dict[str, str]:
    """Return {course_id: syllabus_url} for the given course IDs."""
    if not course_ids:
        return {}
    return dict(_get_syllabus_urls_cached(tuple(sorted(set(course_ids)))))


@lru_cache(maxsize=128)
def _get_course_summaries_cached(course_ids: tuple[str, ...]) -> dict[str, str]:
    from psycopg.rows import dict_row

    result: dict[str, str] = {}
    if not course_ids:
        return result
    pool = get_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            "SELECT course_id, course_summary FROM sections "
            "WHERE course_id = ANY(%s) AND course_summary IS NOT NULL",
            (list(course_ids),),
        ).fetchall()
    for r in rows:
        if r["course_summary"] and r["course_id"] not in result:
            result[r["course_id"]] = r["course_summary"]
    return result


def get_course_summaries(course_ids: list[str]) -> dict[str, str]:
    """Return {course_id: course_summary} for courses that have summaries."""
    if not course_ids:
        return {}
    return dict(_get_course_summaries_cached(tuple(sorted(set(course_ids)))))


@observe(name="node.retrieval.course_summary")
def get_course_summary_chunks(course_ids: list[str]) -> list[dict]:
    """One pseudo-chunk per course summary, shaped like mongo.get_course_summary_chunks.

    Carries ``source="course_summary"`` + ``source_file`` so the citation
    rewriter can whole-document-link the primer.
    """
    from psycopg.rows import dict_row

    requested = list(set(course_ids or []))
    if not requested:
        return []
    pool = get_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        # Pick one document per section deterministically (a CRN may have an HP
        # and a Simple Syllabus copy) for the citation link.
        rows = conn.execute(
            "SELECT s.course_id, s.course_summary, "
            "  (SELECT source_file FROM documents d WHERE d.crn = s.crn "
            "   ORDER BY d.source, d.id LIMIT 1) AS source_file "
            "FROM sections s "
            "WHERE s.course_id = ANY(%s) AND s.course_summary IS NOT NULL",
            (requested,),
        ).fetchall()
    chunks: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        cid, summary, source_file = r["course_id"], r["course_summary"], r["source_file"]
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
                "source_file": source_file,
                "chunk_index": 0,
                "pipeline_version": "v4",
            }
        )
    return chunks
