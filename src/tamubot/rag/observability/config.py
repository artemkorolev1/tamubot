"""Observability configuration — single dataclass controlling tracing + eval per-request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class ObservabilityConfig:
    """Per-request observability settings.

    Controls trace naming, tags, and which eval blocks run.
    """

    trace_name: str = "tamubot.request"
    tags: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    eval_blocks: list[str] = field(default_factory=list)  # empty = no evals
    eval_async: bool = True  # background thread vs synchronous
    eval_retry: bool = True  # retry once on failure
    enable_generator: bool = True  # False for retrieval-only evals


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------


def prod_config(session_id: Optional[str] = None) -> ObservabilityConfig:
    """App.py — tracing only, no evals."""
    return ObservabilityConfig(
        trace_name="tamubot.request",
        tags=["prod"],
        session_id=session_id,
    )


def probe_config(
    tag: Optional[str] = None,
    session_id: Optional[str] = None,
    ragas: bool = False,
) -> ObservabilityConfig:
    """run_probe.py — tracing + optional RAGAS (async)."""
    tags = ["probe"]
    if tag:
        tags.append(tag)
    eval_blocks = ["faithfulness", "answer_relevancy"] if ragas else []
    return ObservabilityConfig(
        trace_name="tamubot.probe",
        tags=tags,
        session_id=session_id,
        eval_blocks=eval_blocks,
        eval_async=True,
    )


def benchmark_config(
    experiment_name: str = "",
    ragas: bool = False,
) -> ObservabilityConfig:
    """run_benchmark.py — tracing + synchronous RAGAS."""
    eval_blocks = ["faithfulness", "answer_relevancy"] if ragas else []
    return ObservabilityConfig(
        trace_name="tamubot.benchmark",
        tags=["benchmark", experiment_name] if experiment_name else ["benchmark"],
        metadata={"experiment": experiment_name},
        eval_blocks=eval_blocks,
        eval_async=False,
    )


def chunking_config(
    experiment: str = "",
    run_name: str = "",
    ragas: bool = False,
) -> ObservabilityConfig:
    """eval_chunking.py — retrieval-only + optional retrieval RAGAS (sync)."""
    eval_blocks = ["context_precision", "context_recall"] if ragas else []
    return ObservabilityConfig(
        trace_name="tamubot.benchmark",
        tags=["chunking_eval", experiment, run_name],
        metadata={"experiment": experiment, "run_name": run_name},
        eval_blocks=eval_blocks,
        eval_async=False,
        enable_generator=False,
    )


def eval_config(
    experiment: str,
    run_name: str,
    with_generation: bool,
    metrics: list[str],
) -> ObservabilityConfig:
    """run_eval.py — unified config: caller picks generation on/off and metric list."""
    return ObservabilityConfig(
        trace_name="tamubot.benchmark",
        tags=["eval", experiment, run_name] + (["with_generation"] if with_generation else ["retrieval_only"]),
        metadata={"experiment": experiment, "run_name": run_name, "with_generation": with_generation},
        eval_blocks=list(metrics),
        eval_async=False,
        enable_generator=with_generation,
    )
