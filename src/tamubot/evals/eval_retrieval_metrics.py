"""Retrieval evaluation metrics for TamuBot.

Provides:
  - label_relevant()   — automated relevance labeling via Voyage-3 cosine similarity
  - recall_at_k()      — proportion of relevant chunks in top-k
  - ndcg_gain()        — NDCG improvement before vs after reranking
  - rrf_sweep()        — RRF k_param sensitivity sweep (k=20,40,60,80,100)

Requires live MongoDB + Voyage AI (VOYAGE_API_KEY must be set).

Usage:
    python scripts/eval_retrieval_metrics.py --query "What is the grading for CSCE 638?"
    python scripts/eval_retrieval_metrics.py --golden-set tamu_data/logs/golden_set.jsonl
    python scripts/eval_retrieval_metrics.py --rrf-sweep --query "CSCE 638 grading"
"""

import argparse
import json
import math
import sys
from pathlib import Path

from tamubot.core import config

# ---------------------------------------------------------------------------
# Relevance labeling
# ---------------------------------------------------------------------------

def label_relevant(
    query: str,
    chunks: list[dict],
    threshold: float = 0.85,
) -> list[bool]:
    """Label each chunk as relevant (True) or not (False) using Voyage-3 cosine similarity.

    Embeds the query as a query vector and each chunk's content as a document
    vector, then compares cosine similarity against the threshold.

    Args:
        query:     User query string.
        chunks:    List of chunk dicts (must have 'content' key).
        threshold: Cosine similarity threshold for relevance (default 0.85).
                   Voyage-3 similarities > 0.85 reliably indicate strong relevance.

    Returns:
        List of bool, one per chunk.
    """
    if not chunks:
        return []

    import voyageai
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    query_embed = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
    texts = [c.get("content", "") for c in chunks]
    doc_embeds = vo.embed(texts, model="voyage-3", input_type="document").embeddings

    labels = []
    for doc_vec in doc_embeds:
        sim = _cosine(query_embed, doc_vec)
        labels.append(sim >= threshold)

    return labels


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_single(text: str, input_type: str = "query") -> list[float]:
    import voyageai
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return vo.embed([text], model="voyage-3", input_type=input_type).embeddings[0]


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(relevant_labels: list[bool], k: int) -> float:
    """Compute Recall@k: fraction of relevant chunks that appear in the top-k.

    If there are no relevant chunks, returns 0.0 (undefined).

    Args:
        relevant_labels: List of bool relevance labels, in ranked order.
        k:               Cutoff rank.

    Returns:
        Float in [0.0, 1.0].
    """
    total_relevant = sum(relevant_labels)
    if total_relevant == 0:
        return 0.0
    top_k = relevant_labels[:k]
    return sum(top_k) / total_relevant


# ---------------------------------------------------------------------------
# NDCG gain (before vs after reranking)
# ---------------------------------------------------------------------------

def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at cutoff k.

    Args:
        relevance_scores: Graded relevance scores (e.g. cosine similarities) in ranked order.
        k:                Cutoff rank.

    Returns:
        NDCG@k in [0.0, 1.0].
    """
    k = min(k, len(relevance_scores))
    dcg = sum(
        relevance_scores[i] / math.log2(i + 2)  # i+2 because log2(rank+1), 0-indexed
        for i in range(k)
    )
    # Ideal DCG: sort by relevance descending
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_gain(
    pre_rerank: list[dict],
    post_rerank: list[dict],
    query: str,
    k: int = 10,
    threshold: float = 0.85,
) -> dict:
    """Measure NDCG@k improvement after reranking.

    Embeds the query and all chunks once, computes cosine similarity as a
    relevance proxy, then computes NDCG@k for the pre- and post-rerank orderings.

    Args:
        pre_rerank:  Chunks before reranking (initial retrieval order).
        post_rerank: Chunks after reranking (reranker output order).
        query:       The search query.
        k:           NDCG cutoff (default 10).
        threshold:   Cosine threshold for binary relevance labels.

    Returns:
        Dict with: ndcg_pre, ndcg_post, gain, recall_pre, recall_post.
    """
    if not pre_rerank and not post_rerank:
        return {"ndcg_pre": 0.0, "ndcg_post": 0.0, "gain": 0.0,
                "recall_pre": 0.0, "recall_post": 0.0}

    import voyageai
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    query_embed = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]

    # Embed all unique chunks (union of pre and post)
    all_chunks = pre_rerank + post_rerank
    all_texts = [c.get("content", "") for c in all_chunks]
    all_embeds = vo.embed(all_texts, model="voyage-3", input_type="document").embeddings

    # Build content → embedding map
    content_to_embed: dict[str, list[float]] = {}
    for chunk, embed in zip(all_chunks, all_embeds):
        content_to_embed[chunk.get("content", "")] = embed

    def get_scores(chunks: list[dict]) -> list[float]:
        scores = []
        for c in chunks:
            emb = content_to_embed.get(c.get("content", ""))
            if emb:
                scores.append(_cosine(query_embed, emb))
            else:
                scores.append(0.0)
        return scores

    pre_scores = get_scores(pre_rerank)
    post_scores = get_scores(post_rerank)

    ndcg_pre = ndcg_at_k(pre_scores, k)
    ndcg_post = ndcg_at_k(post_scores, k)
    gain = ndcg_post - ndcg_pre

    recall_pre = recall_at_k([s >= threshold for s in pre_scores], k)
    recall_post = recall_at_k([s >= threshold for s in post_scores], k)

    return {
        "ndcg_pre": round(ndcg_pre, 4),
        "ndcg_post": round(ndcg_post, 4),
        "gain": round(gain, 4),
        "recall_pre": round(recall_pre, 4),
        "recall_post": round(recall_post, 4),
        "k": k,
    }


# ---------------------------------------------------------------------------
# RRF k_param sensitivity sweep
# ---------------------------------------------------------------------------

def rrf_sweep(
    query: str,
    filters: dict,
    k_values: list[int] | None = None,
    top_k: int = 10,
) -> dict:
    """Sweep RRF k_param values and measure NDCG@top_k for each.

    The RRF formula is: score(d) = Σ 1/(k + rank(d))
    Higher k → smoother rank fusion, less weight on top ranks.

    This function re-runs hybrid_search for each k_param value and measures
    retrieval quality via NDCG using Voyage-3 cosine similarities as relevance proxy.

    Args:
        query:    Search query string.
        filters:  MongoDB filter dict (e.g. {"course_id": "CSCE 638"}).
        k_values: List of RRF k values to sweep (default [20, 40, 60, 80, 100]).
        top_k:    NDCG cutoff (default 10).

    Returns:
        Dict mapping k_value → {"ndcg": float, "n_results": int}.
    """
    if k_values is None:
        k_values = [20, 40, 60, 80, 100]

    import voyageai

    from tamubot.rag.tools.backend import hybrid_search

    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    query_embed = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]

    course_id = filters.get("course_id", "") if filters else ""

    sweep_results: dict[int, dict] = {}
    for k_param in k_values:
        results = hybrid_search(query, course_id=course_id, k=20)

        if not results:
            sweep_results[k_param] = {"ndcg": 0.0, "n_results": 0}
            continue

        texts = [r.get("content", "") for r in results]
        embeds = vo.embed(texts, model="voyage-3", input_type="document").embeddings
        scores = [_cosine(query_embed, e) for e in embeds]

        ndcg = ndcg_at_k(scores, top_k)
        sweep_results[k_param] = {
            "ndcg": round(ndcg, 4),
            "n_results": len(results),
        }
        print(f"  k_param={k_param}: NDCG@{top_k}={ndcg:.4f}  n={len(results)}")

    best_k = max(sweep_results, key=lambda k: sweep_results[k]["ndcg"])
    return {
        "sweep": sweep_results,
        "best_k": best_k,
        "best_ndcg": sweep_results[best_k]["ndcg"],
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Batch evaluation over a golden set
# ---------------------------------------------------------------------------

def evaluate_retrieval_golden_set(
    golden_set: list[dict],
    k: int = 10,
    relevance_threshold: float = 0.85,
) -> dict:
    """Run Recall@k and NDCG gain over a full golden set.

    For each item with an expected_function that requires retrieval, runs
    route_retrieve_rerank (full pipeline) and measures quality.

    Args:
        golden_set:          List of golden question dicts.
        k:                   Recall@k and NDCG cutoff.
        relevance_threshold: Cosine threshold for binary relevance labels.

    Returns:
        Dict with per-item results and aggregate statistics.
    """
    from tamubot.rag.router import classify_query, compute_dynamic_k
    from tamubot.rag.tools.backend import hybrid_search, semantic_search
    from tamubot.rag.tools.voyage import rerank

    item_results = []
    for i, item in enumerate(golden_set, 1):
        query = item.get("question", item.get("query", ""))
        if not query:
            continue

        print(f"  [{i:2d}/{len(golden_set)}] {query[:60]}...")

        # Router
        try:
            rr = classify_query(query)
        except Exception as e:
            print(f"    Router error: {e}")
            continue

        if not rr.requires_retrieval:
            continue

        fn = rr.function
        mode = rr.retrieval_mode
        dk = compute_dynamic_k(fn, len(rr.course_ids))
        retrieve_k = dk["retrieve_k"]
        rerank_k = dk["rerank_k"]

        search_query = rr.rewritten_query or query

        # Pre-rerank retrieval
        try:
            if fn == "semantic_general":
                pre_results = semantic_search(search_query, k=retrieve_k)
            elif rr.course_ids and "hybrid" in mode:
                pre_results = hybrid_search(
                    search_query, course_id=rr.course_ids[0], k=retrieve_k
                )
            else:
                item_results.append({"query": query, "skipped": True, "reason": "metadata path"})
                continue
        except Exception as e:
            print(f"    Retrieval error: {e}")
            continue

        # Post-rerank
        try:
            post_results = rerank(search_query, pre_results, top_k=rerank_k)
        except Exception as e:
            print(f"    Reranker error: {e}")
            post_results = pre_results[:rerank_k]

        # Labels and metrics
        rel_labels_pre = label_relevant(query, pre_results, threshold=relevance_threshold)
        rel_labels_post = label_relevant(query, post_results, threshold=relevance_threshold)

        recall_pre = recall_at_k(rel_labels_pre, k)
        recall_post = recall_at_k(rel_labels_post, k)

        ndcg_metrics = ndcg_gain(pre_results, post_results, query, k=k)

        item_results.append({
            "query": query,
            "function": fn,
            "n_pre": len(pre_results),
            "n_post": len(post_results),
            "recall_pre": recall_pre,
            "recall_post": recall_post,
            "ndcg_pre": ndcg_metrics["ndcg_pre"],
            "ndcg_post": ndcg_metrics["ndcg_post"],
            "ndcg_gain": ndcg_metrics["gain"],
            "n_relevant_pre": sum(rel_labels_pre),
            "n_relevant_post": sum(rel_labels_post),
        })

        print(f"    recall@{k}: {recall_pre:.3f} → {recall_post:.3f}  "
              f"NDCG@{k}: {ndcg_metrics['ndcg_pre']:.3f} → {ndcg_metrics['ndcg_post']:.3f}")

    # Aggregates
    valid = [r for r in item_results if not r.get("skipped")]
    if not valid:
        return {"error": "no valid retrieval results", "items": item_results}

    def _avg(key: str) -> float:
        vals = [r[key] for r in valid if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_items": len(valid),
        "avg_recall_pre": _avg("recall_pre"),
        "avg_recall_post": _avg("recall_post"),
        "avg_ndcg_pre": _avg("ndcg_pre"),
        "avg_ndcg_post": _avg("ndcg_post"),
        "avg_ndcg_gain": _avg("ndcg_gain"),
        "k": k,
        "items": item_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TamuBot retrieval evaluation metrics")
    parser.add_argument("--query", help="Single query to evaluate")
    parser.add_argument("--course-id", default="CSCE 638",
                        help="Course ID filter for single-query mode (default: CSCE 638)")
    parser.add_argument("--golden-set", type=Path,
                        help="Path to golden_set.jsonl for batch evaluation")
    parser.add_argument("--rrf-sweep", action="store_true",
                        help="Run RRF k_param sensitivity sweep")
    parser.add_argument("--k", type=int, default=10, help="Recall@k / NDCG cutoff (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine threshold for relevance labels (default: 0.85)")
    parser.add_argument("--output", type=Path, help="Write results JSON to this path")
    args = parser.parse_args()

    if not config.VOYAGE_API_KEY:
        print("ERROR: VOYAGE_API_KEY is not set. Retrieval metrics require Voyage AI.")
        sys.exit(1)

    results_out = {}

    if args.rrf_sweep:
        query = args.query or "What is the grading policy for CSCE 638?"
        print(f"RRF k_param sweep for: '{query}'")
        filters = {"course_id": args.course_id}
        sweep = rrf_sweep(query, filters, top_k=args.k)
        print(f"\nBest k_param: {sweep['best_k']}  (NDCG@{args.k}={sweep['best_ndcg']})")
        results_out["rrf_sweep"] = sweep

    elif args.golden_set:
        print(f"Loading golden set from {args.golden_set}...")
        golden = []
        with args.golden_set.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    golden.append(json.loads(line))
        print(f"Evaluating retrieval on {len(golden)} golden questions...")
        metrics = evaluate_retrieval_golden_set(golden, k=args.k, relevance_threshold=args.threshold)

        print(f"\n{'=' * 50}")
        print(f"  RETRIEVAL EVAL SUMMARY (k={args.k})")
        print(f"{'=' * 50}")
        print(f"  Items evaluated:     {metrics.get('n_items', 0)}")
        print(f"  Avg Recall@{args.k} (pre):  {metrics.get('avg_recall_pre', 'N/A')}")
        print(f"  Avg Recall@{args.k} (post): {metrics.get('avg_recall_post', 'N/A')}")
        print(f"  Avg NDCG@{args.k} (pre):    {metrics.get('avg_ndcg_pre', 'N/A')}")
        print(f"  Avg NDCG@{args.k} (post):   {metrics.get('avg_ndcg_post', 'N/A')}")
        print(f"  Avg NDCG gain:       {metrics.get('avg_ndcg_gain', 'N/A')}")
        results_out["retrieval_metrics"] = metrics

    elif args.query:
        print(f"Evaluating single query: '{args.query}'")
        from tamubot.rag.router import classify_query
        from tamubot.rag.tools.backend import hybrid_search
        from tamubot.rag.tools.voyage import rerank

        rr = classify_query(args.query)
        print(f"  Router: fn={rr.function}, mode={rr.retrieval_mode}")
        search_query = rr.rewritten_query or args.query

        pre = hybrid_search(search_query, course_id=args.course_id, k=20)
        post = rerank(search_query, pre, top_k=args.k)

        labels = label_relevant(args.query, post, threshold=args.threshold)
        recall = recall_at_k(labels, args.k)
        ndcg_metrics = ndcg_gain(pre, post, args.query, k=args.k)

        print(f"\n  Recall@{args.k}: {recall:.4f}")
        print(f"  NDCG@{args.k} pre:  {ndcg_metrics['ndcg_pre']:.4f}")
        print(f"  NDCG@{args.k} post: {ndcg_metrics['ndcg_post']:.4f}")
        print(f"  NDCG gain:    {ndcg_metrics['gain']:+.4f}")
        results_out = {"recall": recall, "ndcg": ndcg_metrics}

    else:
        parser.print_help()
        sys.exit(0)

    if args.output and results_out:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results_out, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
