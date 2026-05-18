"""Langfuse singleton + trace lifecycle helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Optional

if TYPE_CHECKING:
    from .config import ObservabilityConfig

logger = logging.getLogger("tamubot.observability")

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
# Trace lifecycle — context manager
# ---------------------------------------------------------------------------


@contextmanager
def trace_context(
    obs_config: ObservabilityConfig,
    query: str,
    trace_id: Optional[str] = None,
) -> Generator[tuple[Optional[object], Optional[str]], None, None]:
    """Context manager for Langfuse trace lifecycle.

    Yields (span, trace_id) or (None, None) if Langfuse is not configured.
    Guarantees OTEL context cleanup on exit — safe for sequential loops.

    Args:
        obs_config: Observability settings (trace name, tags, etc.)
        query: The user query (used as trace input).
        trace_id: Optional existing trace ID to resume (for multi-step flows).
    """
    lf = get_langfuse()
    if lf is None:
        yield None, None
        return

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
    if trace_id is not None:
        kwargs["trace_context"] = {"trace_id": trace_id}

    # Set up the observation. If setup fails, yield (None, None) so the caller
    # can still proceed without tracing. Critically, do NOT wrap the user-code
    # yield in this except — otherwise exceptions thrown into the generator
    # (e.g. Streamlit's StopException) would be caught and we'd try to yield
    # again, triggering "generator didn't stop after throw()".
    try:
        observation_cm = lf.start_as_current_observation(**kwargs)
        span = observation_cm.__enter__()
    except Exception as e:
        logger.warning(f"Langfuse trace creation failed: {e}")
        yield None, None
        return

    attr_ctx = None
    try:
        resolved_trace_id = span.trace_id

        # Propagate session_id via OTEL attributes so all child spans inherit it.
        if obs_config.session_id:
            try:
                from langfuse._client.propagation import _propagate_attributes

                attr_ctx = _propagate_attributes(session_id=obs_config.session_id)
                attr_ctx.__enter__()
            except Exception as e:
                logger.warning(f"Session propagation setup failed: {e}")
                attr_ctx = None

        # Set tags on the trace (nice-to-have).
        if obs_config.tags:
            try:
                lf._create_trace_tags_via_ingestion(
                    trace_id=resolved_trace_id,
                    tags=obs_config.tags,
                )
            except Exception:
                pass

        yield span, resolved_trace_id
    finally:
        # End span
        try:
            span.end()
        except Exception as e:
            logger.warning(f"span.end() failed: {e}")

        # Exit attribute propagation before observation context exits
        if attr_ctx is not None:
            try:
                attr_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Attribute context exit failed: {e}")

        # Exit the observation context manager
        try:
            observation_cm.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Observation context exit failed: {e}")

        # Flush buffered spans to Langfuse
        try:
            lf.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")
