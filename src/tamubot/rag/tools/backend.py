"""Retrieval backend selector.

Dispatches the Layer-1 query primitives to MongoDB Atlas (``tools.mongo``) or
Postgres + pgvector (``tools.queries``) based on ``config.RETRIEVAL_BACKEND``,
read **at call time** so the flag can be flipped at runtime (eval gate, tests)
without re-importing. Both backends expose an identical interface and return the
same chunk dict shape, so callers (`retrieval_node`, `recursive_retrieval_node`,
the eval harness) import from here and stay backend-agnostic.

``RETRIEVAL_BACKEND=postgres`` → Postgres; anything else (``mongodb`` default,
or the legacy ``vertex``) → Mongo.
"""

from __future__ import annotations

from tamubot.core import config


def _impl():
    """Return the active backend module (re-evaluated per call)."""
    if config.RETRIEVAL_BACKEND == "postgres":
        from tamubot.rag.tools import queries

        return queries
    from tamubot.rag.tools import mongo

    return mongo


# Transparent passthroughs: forward *args/**kwargs verbatim so call semantics
# (including the retrieval node's ``with_meta`` TypeError fallback and existing
# ``patch("...tools.mongo.hybrid_search")`` test seams) are preserved exactly.
def hybrid_search(*args, **kwargs):
    return _impl().hybrid_search(*args, **kwargs)


def semantic_search(*args, **kwargs):
    return _impl().semantic_search(*args, **kwargs)


def fetch_anchor_chunks(*args, **kwargs):
    return _impl().fetch_anchor_chunks(*args, **kwargs)


def get_meeting_times(*args, **kwargs):
    return _impl().get_meeting_times(*args, **kwargs)


def get_syllabus_urls(*args, **kwargs):
    return _impl().get_syllabus_urls(*args, **kwargs)


def get_course_summaries(*args, **kwargs):
    return _impl().get_course_summaries(*args, **kwargs)


def get_course_summary_chunks(*args, **kwargs):
    return _impl().get_course_summary_chunks(*args, **kwargs)
