"""Langfuse singleton + trace lifecycle helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .config import ObservabilityConfig

logger = logging.getLogger("tamubot.observability")

# Stores the context manager returned by start_as_current_observation
# so finalize_trace can exit it and detach the OTEL context.
_active_ctx_managers: dict[int, Any] = {}

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_langfuse_client = None


def get_langfuse():
    """Lazy singleton. Returns None if Langfuse credentials are not configured."""
    global _langfuse_client
    if _langfuse_client is None:
        from tamubot.core import config

        if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
            return None
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_BASE_URL,
                flush_interval=0.5,
            )
            logger.info("Langfuse SDK client initialised.")
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
            return None
    return _langfuse_client


# ---------------------------------------------------------------------------
# Trace lifecycle
# ---------------------------------------------------------------------------


def create_trace(
    obs_config: ObservabilityConfig,
    query: str,
    trace_id: Optional[str] = None,
) -> tuple[Optional[object], Optional[str]]:
    """Create a Langfuse root observation (trace). Returns (span, trace_id) or (None, None).

    Uses start_as_current_observation() so the OTEL context is set — any
    @observe-decorated functions called while this trace is active will
    automatically nest as child spans instead of creating separate traces.

    Args:
        obs_config: Observability settings (trace name, tags, etc.)
        query: The user query (used as trace input).
        trace_id: Optional existing trace ID to resume.  When provided, the new
                  observation is appended to the existing trace instead of
                  creating a new one (useful for multi-step flows like advisory
                  SCP collection → pipeline execution).
    """
    lf = get_langfuse()
    if lf is None:
        return None, None
    try:
        merged_meta = {
            **(obs_config.metadata or {}),
            "session_id": obs_config.session_id,
        }
        kwargs: dict[str, Any] = dict(
            name=obs_config.trace_name,
            input=query,
            metadata=merged_meta,
            end_on_exit=False,
        )
        if obs_config.session_id:
            kwargs["session_id"] = obs_config.session_id
        if trace_id is not None:
            kwargs["trace_context"] = {"trace_id": trace_id}

        ctx_manager = lf.start_as_current_observation(**kwargs)
        span = ctx_manager.__enter__()
        resolved_trace_id = span.trace_id

        # Store the context manager so finalize_trace can __exit__ it
        _active_ctx_managers[id(span)] = ctx_manager

        # Set tags on the trace via SDK internal API (only public way in v4)
        if obs_config.tags:
            try:
                lf._create_trace_tags_via_ingestion(
                    trace_id=resolved_trace_id,
                    tags=obs_config.tags,
                )
            except Exception:
                pass  # tags are nice-to-have, not critical
        return span, resolved_trace_id
    except Exception as e:
        logger.warning(f"Langfuse trace creation failed: {e}")
        return None, None


def finalize_trace(trace, output: str) -> None:
    """Update trace output, end span, exit OTEL context, and flush. No-op if trace is None."""
    if trace is None:
        return
    try:
        trace.update(output=output)
        trace.end()
    except Exception:
        pass

    # Exit the context manager to detach the OTEL context
    ctx_manager = _active_ctx_managers.pop(id(trace), None)
    if ctx_manager is not None:
        try:
            ctx_manager.__exit__(None, None, None)
        except Exception:
            pass

    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
