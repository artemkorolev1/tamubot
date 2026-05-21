"""Reciprocal Rank Fusion for merging multi-subquery retrieval results.

Pure function. Given N ranked lists of chunk dicts (one per subquery variant),
returns a single ranked list of unique chunks ordered by the fused score:

    score(d) = Σ_i  1 / (k + rank_i(d))

where ``rank_i(d)`` is the 1-based position of ``d`` in the i-th list (omitted
from the sum when ``d`` is not in list i). ``k=60`` is the canonical default
from the RRF paper.
"""

from __future__ import annotations

from typing import Any, Callable, Union

IdKey = Union[str, Callable[[dict], Any]]


def rrf_fuse(
    result_lists: list[list[dict]],
    *,
    k: int = 60,
    final_k: int,
    id_key: IdKey = "chunk_id",
) -> list[dict]:
    """Fuse multiple ranked chunk lists via Reciprocal Rank Fusion.

    Args:
        result_lists: One ranked list of chunk dicts per subquery variant.
            Empty lists contribute nothing. Each chunk must carry an identifier
            resolvable via ``id_key``.
        k: RRF constant. Default 60.
        final_k: Number of fused results to return.
        id_key: Either a field name (``doc[id_key]``) or a callable
            ``id_key(doc)`` that returns a hashable identifier. Docs whose
            identifier is None are skipped.

    Returns:
        Up to ``final_k`` chunks ordered by descending fused score. Duplicates
        are merged; the first-seen dict (across lists, in list order) wins as
        the representative payload.
    """
    if final_k <= 0:
        return []

    resolve: Callable[[dict], Any]
    if callable(id_key):
        resolve = id_key
    else:
        field_name = id_key
        resolve = lambda d: d.get(field_name)  # noqa: E731

    scores: dict[Any, float] = {}
    representatives: dict[Any, dict] = {}

    for results in result_lists:
        for rank_zero, doc in enumerate(results):
            chunk_id = resolve(doc)
            if chunk_id is None:
                continue
            rank = rank_zero + 1  # 1-based
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in representatives:
                representatives[chunk_id] = doc

    ordered_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    fused: list[dict] = []
    for cid in ordered_ids[:final_k]:
        doc = dict(representatives[cid])
        doc["rrf_score"] = scores[cid]
        fused.append(doc)
    return fused
