"""Lightweight probe script — run 1-N queries through the full RAG pipeline with Langfuse tracing.

Useful for iterating on prompt changes, retrieval tuning, or spot-checking regressions
without running the full Streamlit app.

Usage:
    # Ad-hoc query
    python evals/run_probe.py --query "What is the grading breakdown for CSCE 638?"

    # Named tests from TEST_SUITE (1-based IDs)
    python evals/run_probe.py --test-ids 1 3 7

    # Run all named tests
    python evals/run_probe.py --suite all

    # Tag traces for before/after comparison
    python evals/run_probe.py --query "..." --tag "generator_v2"

    # Use conversation-memory pipeline (multi-turn simulation)
    python evals/run_probe.py --query "tell me about CSCE 638" --memory --thread-id sess1
    python evals/run_probe.py --query "what are the assignments?" --memory --thread-id sess1

    # Also run RAGAS faithfulness+relevancy scoring (async, ~30s)
    python evals/run_probe.py --test-ids 1 --ragas
"""

import argparse
import re
import sys
import time
from datetime import datetime

# UTF-8 stdout on Windows (avoids cp1252 encode errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tamubot.core import config
from tamubot.evals.eval_pipeline import (  # noqa: F401  (TestCase re-exported for callers)
    TEST_SUITE,
    TestCase,
)
from tamubot.rag import run_pipeline_with_memory
from tamubot.rag.graph.pipeline import run_pipeline
from tamubot.rag.observability import (
    EvalInputs,
    get_langfuse,
    probe_config,
    run_evals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(query: str, max_len: int = 40) -> str:
    """Collapse non-alphanumeric runs to underscores, trim to max_len."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query)
    return slug[:max_len].strip("_")


def _trace_url(trace_id: str) -> str:
    return f"{config.LANGFUSE_BASE_URL}/trace/{trace_id}"


def _est_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Core probe runner
# ---------------------------------------------------------------------------


def run_probe(
    query: str,
    tag: str | None = None,
    session_id: str | None = None,
    ragas: bool = False,
    index: int = 1,
    total: int = 1,
    memory: bool = False,
    thread_id: str | None = None,
) -> dict:
    """Run one query through the full 3-stage pipeline and print a summary.

    Args:
        query:      The user query to probe.
        tag:        Optional trace tag for before/after comparisons.
        session_id: Groups all probes in a multi-query run under one session.
        ragas:      If True, kick off async RAGAS faithfulness+relevancy scoring.
        index:      1-based position in the current batch (for display).
        total:      Total queries in the batch (for display).
        memory:     If True, use run_pipeline_with_memory for multi-turn context.
        thread_id:  Thread ID for memory pipeline (default: session_id).

    Returns:
        Summary dict with function, chunk count, trace URL, token estimates, etc.
    """
    from tamubot.rag.observability import trace_context

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    obs = probe_config(tag=tag, session_id=session_id or f"probe_{ts}", ragas=ragas)
    if memory:
        obs.tags.append("memory")
    obs.metadata.update({"tag": tag, "memory": memory})

    if get_langfuse() is None:
        print(
            "  [WARNING] Langfuse not configured (LANGFUSE_PUBLIC_KEY unset) — pipeline runs without tracing.",
            file=sys.stderr,
        )

    t0 = time.time()
    answer = ""
    reranked: list[dict] = []
    router_result = None
    trace_id = None  # safety-net default before context-manager binds it

    with trace_context(obs, query=query) as (trace, trace_id):  # noqa: F811
        if memory:
            effective_thread = thread_id or session_id or f"probe_{ts}"
            thread_config = {"configurable": {"thread_id": effective_thread}}
            reranked, router_result, _data_gaps, _data_integrity, _conflicted, answer_tokens = run_pipeline_with_memory(
                query, thread_config=thread_config
            )
            retrieval_elapsed = time.time() - t0
            t1 = time.time()
            answer = "".join(answer_tokens)
            generation_elapsed = time.time() - t1
        else:
            reranked, router_result, _data_gaps, _data_integrity, _conflicted, answer, _timing = run_pipeline(
                query, return_timing=True
            )
            total_elapsed = time.time() - t0
            retrieval_elapsed = total_elapsed - _timing.get("generator_node", 0) / 1000
            generation_elapsed = _timing.get("generator_node", 0) / 1000

        # Set trace output before context manager exits
        if trace is not None:
            trace.update(output=answer[:500])

    # --- Optional RAGAS evaluation (runs after trace finalized) ---
    if ragas and trace_id:
        contexts = [doc.get("content", "") for doc in reranked]
        run_evals(obs, EvalInputs(question=query, contexts=contexts, answer=answer, trace_id=trace_id))

    # --- Token estimates ---
    chunk_text = "".join(doc.get("content", "") for doc in reranked)
    est_in = _est_tokens(query + chunk_text)
    est_out = _est_tokens(answer)

    # --- Answer preview (first 120 chars, single line) ---
    answer_preview = answer[:120].replace("\n", " ")
    if len(answer) > 120:
        answer_preview += "…"

    # --- Derive display values ---
    courses_str = ", ".join(router_result.course_ids) if router_result and router_result.course_ids else "(none)"
    total_elapsed = retrieval_elapsed + generation_elapsed

    # --- Print summary ---
    print(f"\n[{index}/{total}] Q: {query}")
    print(f"  → function: {router_result.function if router_result else 'unknown'} | courses: {courses_str}")
    print(
        f"  → {len(reranked)} chunks | retrieval: {retrieval_elapsed:.1f}s | generation: {generation_elapsed:.1f}s | total: {total_elapsed:.1f}s"
    )
    print(f"  → ~{est_in} in-tokens | ~{est_out} out-tokens | {len(answer)} chars")
    print(f"  → A: {answer_preview}")
    if memory:
        effective_thread = thread_id or session_id or f"probe_{ts}"
        print(f"  [memory] thread: {effective_thread}")
    if trace_id:
        print(f"  → Trace: {_trace_url(trace_id)}")
    if ragas and trace_id:
        print("  → RAGAS: scoring in background (~30s), check Langfuse for scores")

    return {
        "query": query,
        "function": router_result.function if router_result else "unknown",
        "course_ids": router_result.course_ids if router_result else [],
        "n_chunks": len(reranked),
        "answer_len": len(answer),
        "answer_preview": answer_preview,
        "est_in_tokens": est_in,
        "est_out_tokens": est_out,
        "retrieval_elapsed": round(retrieval_elapsed, 2),
        "generation_elapsed": round(generation_elapsed, 2),
        "total_elapsed": round(total_elapsed, 2),
        "memory": memory,
        "trace_id": trace_id,
        "trace_url": _trace_url(trace_id) if trace_id else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run probe queries through the full TamuBot RAG pipeline with Langfuse tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Ad-hoc query string")
    group.add_argument(
        "--test-ids",
        type=int,
        nargs="+",
        metavar="N",
        help="1-based test case IDs from TEST_SUITE in eval_pipeline.py",
    )
    group.add_argument(
        "--suite",
        choices=["smoke", "all"],
        help="'smoke' = 3-query sanity check (hybrid_course + recursive + comparison); 'all' = full TEST_SUITE",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional Langfuse tag for before/after comparisons (e.g. 'generator_v2')",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also run async RAGAS faithfulness+relevancy scoring (~30s, appears in Langfuse)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use conversation-memory pipeline (run_pipeline_with_memory) for multi-turn simulation",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        dest="thread_id",
        help="Thread ID for memory pipeline — reuse across invocations to build conversation history",
    )
    args = parser.parse_args()

    # Smoke suite — fast sanity check covering the three main paths
    SMOKE_QUERIES = [
        "tell me about CSCE 638",  # hybrid_course
        "what should I take alongside CSCE 638?",  # recursive (5-step pipeline)
        "compare CSCE 638 and CSCE 670",  # hybrid_course → generate_comparison
    ]

    # Build the list of queries to run
    if args.query:
        queries: list[str] = [args.query]
    elif args.suite == "smoke":
        queries = SMOKE_QUERIES
    elif args.suite == "all":
        queries = [tc.query for tc in TEST_SUITE]
    else:
        queries = []
        suite_size = len(TEST_SUITE)
        for tid in args.test_ids:
            if 1 <= tid <= suite_size:
                queries.append(TEST_SUITE[tid - 1].query)
            else:
                print(
                    f"[WARNING] Test ID {tid} out of range (valid: 1–{suite_size})",
                    file=sys.stderr,
                )

    if not queries:
        print("No queries to run.", file=sys.stderr)
        sys.exit(1)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"probe_{session_ts}"
    total = len(queries)

    print(f"\nRunning {total} probe quer{'ies' if total != 1 else 'y'} (session: {session_id})")
    if args.memory:
        effective_thread = args.thread_id or session_id
        print(f"Memory: ON | thread: {effective_thread}")
    if args.tag:
        print(f"Tag: {args.tag}")
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        print("[WARNING] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — no tracing.")

    results: list[dict] = []
    for i, query in enumerate(queries, 1):
        try:
            result = run_probe(
                query=query,
                tag=args.tag,
                session_id=session_id,
                ragas=args.ragas,
                index=i,
                total=total,
                memory=args.memory,
                thread_id=args.thread_id,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[{i}/{total}] ERROR for query: {query!r}")
            print(f"  → {type(exc).__name__}: {exc}")

    print(f"\nDone. {len(results)}/{total} queries completed.")


if __name__ == "__main__":
    main()
