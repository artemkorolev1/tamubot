"""rag.observability — unified observability package.

Replaces rag/tools/langfuse.py with a config-driven approach.
"""

from .config import (
    ObservabilityConfig,
    benchmark_config,
    chunking_config,
    eval_config,
    probe_config,
    prod_config,
)
from .evals import EvalInputs, available_metrics, resolve_metrics, run_evals
from .state_capture import serialize_state
from .tracing import get_langfuse, trace_context

__all__ = [
    "ObservabilityConfig",
    "prod_config",
    "probe_config",
    "benchmark_config",
    "chunking_config",
    "eval_config",
    "trace_context",
    "get_langfuse",
    "EvalInputs",
    "run_evals",
    "available_metrics",
    "resolve_metrics",
    "serialize_state",
]
