"""Retrieval-only chunking benchmark.

Measures precision_at_k, recall_at_k (RAGAS ContextRecall), f1_at_k,
hit_rate_at_k, and retrieved_tokens per query. Logs to a Langfuse dataset
experiment run — each query is one row, each metric is a column.

Usage:
    python evals/eval_chunking.py \\
        --golden-set tamu_data/evals/golden_sets/golden_20260411_v1.xlsx \\
        [--experiment chunk_600_ov100] \\
        [--dataset chunking_golden_v1] \\
        [--top-k 7] \\
        [--threshold 0.35] \\
        [--ragas] \\
        [--output tamu_data/evals/reports/chunking_YYYYMMDD.json]
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pre-parse collection/filter args BEFORE rag modules are imported.
# rag/tools/mongo.py reads CHUNKS_COLLECTION etc. at import time, so env vars
# must be set here — before the late `from tamubot.rag.*` imports below (line ~102).
# ---------------------------------------------------------------------------
def _pre_arg(flag: str) -> "str | None":
    """Extract the value of a --flag VALUE pair from sys.argv without argparse."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


if _col := _pre_arg("--chunks-collection"):
    _suffix = _col.removeprefix("chunks_")
    os.environ.setdefault("CHUNKS_COLLECTION", _col)
    os.environ.setdefault("VECTOR_INDEX", f"vector_index_{_suffix}")
    os.environ.setdefault("TEXT_INDEX", f"text_index_{_suffix}")
if _ct := _pre_arg("--chunk-tag"):
    os.environ.setdefault("CHUNK_TAG_FILTER", _ct)


logger = logging.getLogger("tamubot.eval_chunking")


# ---------------------------------------------------------------------------
# Embedding-based metrics (cheap, always computed)
# ---------------------------------------------------------------------------


def compute_embedding_metrics(
    query: str,
    chunks: list[dict],
    threshold: float = 0.35,
    _labels: Optional[list[bool]] = None,
) -> dict:
    """Compute embedding-based retrieval metrics without an LLM.

    Args:
        query:     User query string.
        chunks:    Reranked chunk dicts (must have 'content' key).
        threshold: Voyage-3 cosine similarity threshold (default 0.35).
        _labels:   Pre-computed relevance labels (skips Voyage AI; for tests).

    Returns:
        Dict with keys: precision_at_k (float), hit_rate_at_k (float),
        retrieved_tokens (int).
    """
    if not chunks:
        return {"precision_at_k": 0.0, "hit_rate_at_k": 0.0, "retrieved_tokens": 0}

    from tamubot.evals.eval_retrieval_metrics import label_relevant

    labels = _labels if _labels is not None else label_relevant(query, chunks, threshold)
    k = len(labels)
    n_relevant = sum(labels)

    return {
        "precision_at_k": round(n_relevant / k, 4) if k > 0 else 0.0,
        "hit_rate_at_k": 1.0 if n_relevant > 0 else 0.0,
        "retrieved_tokens": sum(len(c.get("content", "")) // 4 for c in chunks),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 4)


def _compute_aggregates(results: list[dict]) -> dict:
    """Mean of each numeric metric across all query results."""
    metrics = [
        "retrieved_tokens",
        "latency_ms",
        "recall_at_k",
        "context_precision",
        "precision_at_k",
        "hit_rate_at_k",
        "f1_at_k",
    ]
    aggregates: dict = {}
    for m in metrics:
        values = [r[m] for r in results if r.get(m) is not None]
        if values:
            aggregates[f"avg_{m}"] = round(sum(values) / len(values), 4)
    aggregates["n_queries"] = len(results)
    return aggregates


# ---------------------------------------------------------------------------
# Retrieval — via the production eval graph (same code path as normal runs)
# ---------------------------------------------------------------------------

from tamubot.rag.graph.pipeline import run_pipeline_eval  # noqa: E402

# ---------------------------------------------------------------------------
# Langfuse dataset upsert
# ---------------------------------------------------------------------------
from tamubot.rag.observability import (  # noqa: E402
    EvalInputs,
    chunking_config,
    run_evals,
    trace_context,
)


def _item_id(question: str) -> str:
    """Stable 16-char ID for a question — used for idempotent upsert."""
    return hashlib.md5(question.encode()).hexdigest()[:16]


def upsert_langfuse_dataset(lf, golden_items: list[dict], dataset_name: str):
    """Create-or-upsert a Langfuse dataset and upload all golden items.

    Dataset column layout (flat, Langfuse-friendly):
      input           — question string (renders as readable text column)
      expected_output — reference answer string (renders as readable text column)
      metadata        — flat dict of structured params (each key = a column)

    Returns the DatasetClient, or None on failure.
    """
    try:
        lf.create_dataset(
            name=dataset_name,
            description=(
                "TamuBot chunking eval — retrieval quality benchmark. "
                "Each item is a golden question with router expectations and reference answer."
            ),
        )
    except Exception:
        pass  # dataset already exists — that's fine

    # Fetch existing items to avoid duplicate creation
    existing: set[str] = set()
    try:
        page = 1
        while True:
            resp = lf.api.dataset_items.list(dataset_name=dataset_name, page=page, limit=100)
            for it in resp.data:
                q = it.input if isinstance(it.input, str) else ""
                if q:
                    existing.add(q)
            if len(resp.data) < 100:
                break
            page += 1
    except Exception:
        pass  # dataset may not exist yet — will be created below

    uploaded = 0
    for item in golden_items:
        question = item.get("question", item.get("query", ""))
        if not question or question in existing:
            continue
        try:
            lf.create_dataset_item(
                dataset_name=dataset_name,
                input=question,
                expected_output=item.get("reference_answer") or "",
                metadata={
                    "expected_function": item.get("expected_function"),
                    "source_course_id": item.get("source_course_id"),
                    "stratum": item.get("stratum"),
                },
            )
            uploaded += 1
        except Exception as e:
            logger.warning(f"Dataset item upsert failed for '{question[:40]}': {e}")

    print(f"  Langfuse dataset '{dataset_name}': {uploaded}/{len(golden_items)} items upserted.")

    try:
        return lf.get_dataset(dataset_name)
    except Exception as e:
        logger.warning(f"Could not fetch dataset after upsert: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-query runner (shared by Langfuse and fallback paths)
# ---------------------------------------------------------------------------


def _run_one_query(
    question: str,
    reference: str,
    top_k: Optional[int],
    threshold: float,
    ragas_enabled: bool,
    i: int,
    total: int,
    span=None,
) -> Optional[dict]:
    """Run router + retrieval via the production eval graph. Returns result dict or None.

    RAGAS is intentionally NOT run here — it runs after span.end() in the loop
    so it doesn't inflate the Langfuse trace latency.

    Args:
        span: Open Langfuse observation — passed to run_pipeline_eval() so the
              graph's CallbackHandler nests router/retrieval spans under it.
    """
    if not question:
        return None

    if ragas_enabled and not reference:
        print(f"  [{i:2d}/{total}] SKIP (no reference_answer): {question[:60]}")
        return None

    print(f"  [{i:2d}/{total}] {question[:65]}...")

    try:
        t0 = time.perf_counter()
        chunks, router_result, timing_ms = run_pipeline_eval(question, trace=span)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        print(f"    Pipeline error: {e}")
        return None

    if not router_result.requires_retrieval:
        print(f"    Skip: {router_result.function} has no retrieval")
        return None

    # top_k is a metric evaluation cutoff — slice the ranked chunks from pipeline
    eval_chunks = chunks[:top_k] if top_k is not None else chunks

    emb_metrics = compute_embedding_metrics(question, eval_chunks, threshold)

    print(f"    tokens={emb_metrics['retrieved_tokens']}  latency={latency_ms:.0f}ms")

    return {
        "query": question,
        "_chunks": eval_chunks,  # kept for RAGAS in loop, excluded from results
        "chunks_retrieved": len(eval_chunks),
        "latency_ms": latency_ms,
        "recall_at_k": None,  # filled by loop after RAGAS
        "context_precision": None,
        "router_function": router_result.function,
        "course_ids": router_result.course_ids,
        "f1_at_k": None,
        **emb_metrics,
    }


def _score_trace(
    lf,
    trace_id: str,
    row: dict,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    top_k: Optional[int],
    threshold: float,
) -> None:
    """Post numeric metrics as scores on a Langfuse trace via create_score().

    Called after span.end() so scores don't block trace latency measurement.
    RAGAS scores (context_precision, context_recall) are posted by
    compute_retrieval_ragas() directly — not duplicated here.
    """
    # Per-query retrieval metrics
    for name in (
        "retrieved_tokens",
        "avg_chunk_score",
        "precision_at_k",
        "hit_rate_at_k",
        "f1_at_k",
        "chunks_retrieved",
    ):
        value = row.get(name)
        if value is not None:
            lf.create_score(trace_id=trace_id, name=name, value=float(value))

    # Run-level config — same value for every item but visible as score columns
    if chunk_size is not None:
        lf.create_score(trace_id=trace_id, name="chunk_size", value=float(chunk_size))
    if chunk_overlap is not None:
        lf.create_score(trace_id=trace_id, name="chunk_overlap", value=float(chunk_overlap))
    lf.create_score(trace_id=trace_id, name="top_k", value=float(top_k if top_k is not None else -1))
    lf.create_score(trace_id=trace_id, name="threshold", value=float(threshold))


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def run_eval(
    golden_items: list[dict],
    experiment: str,
    dataset_name: str,
    top_k: Optional[int],
    threshold: float,
    ragas_enabled: bool,
    lf,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    description: Optional[str] = None,
) -> tuple[list[dict], str, dict]:
    """Run retrieval eval over all golden items.

    Langfuse layout when lf is set:
      • One trace per query in the Traces list — add score names as columns
        via the Columns toggle to see precision_at_k / hit_rate_at_k / etc.
      • Dataset items (input=question, expected_output=reference) linked to
        their traces via source_trace_id.
      • All traces tagged [experiment, run_name, "chunking_eval"] for filtering.

    Returns:
        Tuple of (results list, run_name string, run_col_results dict mapping question id to chunk string).
    """
    run_name = f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # question → Langfuse dataset item ID (for dataset run linking)
    lf_item_ids: dict[str, str] = {}
    if lf:
        upsert_langfuse_dataset(lf, golden_items, dataset_name)
        try:
            page = 1
            while True:
                resp = lf.api.dataset_items.list(dataset_name=dataset_name, page=page, limit=100)
                for it in resp.data:
                    q = it.input if isinstance(it.input, str) else ""
                    if q:
                        lf_item_ids[q] = it.id
                if len(resp.data) < 100:
                    break
                page += 1
        except Exception as e:
            logger.warning(f"Could not load dataset item IDs: {e}")
        print(f"\nRunning eval: experiment={experiment!r}  run={run_name!r}\n")

    results: list[dict] = []
    question_to_id = {item.get("question", ""): item.get("id") for item in golden_items}
    run_col_results: dict = {}

    for i, item in enumerate(golden_items, 1):
        question = item.get("question", item.get("query", ""))
        reference = item.get("reference_answer", "") or ""

        obs = chunking_config(experiment=experiment, run_name=run_name, ragas=ragas_enabled)
        trace_id: Optional[str] = None  # safety-net default before context-manager binds it
        span_id: Optional[str] = None

        # Trace wraps the pipeline call; RAGAS runs AFTER the with-block so
        # RAGAS observations don't nest under the benchmark span.
        with trace_context(obs, query=question) as (trace, trace_id):  # noqa: F811
            span_id = trace.id if trace is not None else None
            row = _run_one_query(
                question,
                reference,
                top_k,
                threshold,
                ragas_enabled,
                i,
                len(golden_items),
                span=trace,
            )

            # Update trace output before context manager exits
            if trace is not None:
                try:
                    if row is not None:
                        trace.update(
                            output={"router_function": row["router_function"], "n_chunks": len(row["_chunks"])},
                            metadata={
                                "experiment": experiment,
                                "run_name": run_name,
                                "router_function": row["router_function"],
                                "course_ids": row["course_ids"],
                            },
                        )
                except Exception as e:
                    logger.warning(f"Langfuse trace update failed for '{question[:40]}': {e}")

        if row is None:
            # Still link skipped items to the dataset run so they appear in Langfuse
            if lf and trace_id and dataset_name and question in lf_item_ids:
                try:
                    lf.api.dataset_run_items.create(
                        run_name=run_name,
                        dataset_item_id=lf_item_ids[question],
                        trace_id=trace_id,
                        metadata={
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "top_k": top_k,
                            "threshold": threshold,
                            "skipped": True,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Langfuse skip link failed for '{question[:40]}': {e}")
            continue

        # Run RAGAS after trace finalized — scores posted by trace_id, not OTEL context.
        if ragas_enabled and row["_chunks"] and reference:
            contexts = [c.get("content", "") for c in row["_chunks"]]
            ragas_scores = run_evals(
                obs,
                EvalInputs(
                    question=question,
                    contexts=contexts,
                    reference=reference,
                    trace_id=trace_id,
                ),
            )
            recall = ragas_scores.get("context_recall")
            precision_ragas = ragas_scores.get("context_precision")
            row["recall_at_k"] = recall
            row["context_precision"] = precision_ragas
            if recall is not None:
                row["f1_at_k"] = _compute_f1(row.get("precision_at_k", 0.0), recall)
            recall_str = f"  recall={recall:.3f}" if recall is not None else ""
            prec_str = f"  context_precision={precision_ragas:.3f}" if precision_ragas is not None else ""
            print(f"    ...{recall_str}{prec_str}")

        # Compute avg reranker score before dropping chunks
        chunk_scores = [c["score"] for c in row.get("_chunks", []) if isinstance(c.get("score"), (int, float))]
        row["avg_chunk_score"] = sum(chunk_scores) / len(chunk_scores) if chunk_scores else None

        # Build run column value: "CSCE 670 0.87, CSCE 638 0.71"
        run_col_parts = []
        for c in row.get("_chunks", []):
            cid = c.get("course_id", "?")
            score = c.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            run_col_parts.append(f"{cid} {score_str}")
        qid = question_to_id.get(question)
        if qid is not None and run_col_parts:
            run_col_results[qid] = ", ".join(run_col_parts)

        # Strip internal _chunks before storing
        row.pop("_chunks", None)
        results.append(row)

        # Post embedding-based scores + config metadata to Langfuse
        if lf and trace_id:
            try:
                _score_trace(lf, trace_id, row, chunk_size, chunk_overlap, top_k, threshold)
                lf.api.dataset_run_items.create(
                    run_name=run_name,
                    run_description=description,
                    metadata={
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "top_k": top_k if top_k is not None else "auto",
                        "threshold": threshold,
                    },
                    dataset_item_id=lf_item_ids.get(question, _item_id(question)),
                    trace_id=trace_id,
                    observation_id=span_id,
                )
            except Exception as e:
                logger.warning(f"Langfuse scoring failed for '{question[:40]}': {e}")

    if lf:
        lf.flush()

    return results, run_name, run_col_results


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], run_name: str, aggregates: dict) -> None:
    """Print aligned per-query and aggregate metrics to stdout."""
    has_ragas = any(r.get("recall_at_k") is not None for r in results)
    print(f"\n{'=' * 80}")
    print(f"  RETRIEVAL EVAL: {run_name}  |  {len(results)} queries")
    print(f"{'=' * 80}")

    header = f"  {'Query':<42} {'Tokens':>7} {'Lat(ms)':>8}"
    if has_ragas:
        header += f" {'Recall':>7} {'CtxPrec':>8} {'F1':>6}"
    print(header)
    print(f"  {'-' * 78}")

    for r in results:
        ragas_str = ""
        if has_ragas:
            rec = r.get("recall_at_k")
            cp = r.get("context_precision")
            f1 = r.get("f1_at_k")
            rec_s = f"{rec:>7.3f}" if rec is not None else f"{'N/A':>7}"
            cp_s = f"{cp:>8.3f}" if cp is not None else f"{'N/A':>8}"
            f1_s = f"{f1:>6.3f}" if f1 is not None else f"{'N/A':>6}"
            ragas_str = f" {rec_s} {cp_s} {f1_s}"
        lat = r.get("latency_ms")
        lat_str = f"{lat:>8.0f}" if lat is not None else f"{'N/A':>8}"
        print(f"  {r.get('query', '')[:42]:<42} {r['retrieved_tokens']:>7d} {lat_str}{ragas_str}")

    print(f"{'=' * 80}")
    print("  AGGREGATES:")
    for k, v in aggregates.items():
        print(f"    {k:<30} {v}")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Deprecated: retrieval-only chunking eval. Delegates to tamubot.evals.run_eval."""
    parser = argparse.ArgumentParser(
        description="DEPRECATED: use `python -m tamubot.evals.run_eval` (no --with-generation) instead."
    )
    parser.add_argument("--golden-set", type=Path, required=True, help="Path to golden set .xlsx")
    parser.add_argument("--experiment", default="chunking_eval", help="Experiment name")
    parser.add_argument("--dataset", help="Langfuse dataset name (default: golden-set stem)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--ragas", action="store_true", help="Enable context_precision/context_recall")
    parser.add_argument("--chunks-collection", type=str, default=None)
    parser.add_argument("--chunk-tag", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    args = parser.parse_args()

    print("[deprecation] eval_chunking.py is a shim — forwarding to run_eval.py.")
    from tamubot.evals.run_eval import default_metrics, run
    from tamubot.rag.observability import resolve_metrics

    metrics = resolve_metrics(None, default_metrics(with_generation=False)) if args.ragas else []
    run(
        golden_path=args.golden_set,
        experiment=args.experiment,
        with_generation=False,
        metrics=metrics,
        ids=None,
        capture_state=False,
        description=args.description,
        top_k=args.top_k,
        threshold=args.threshold,
        chunks_collection=args.chunks_collection,
        chunk_tag=args.chunk_tag,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
