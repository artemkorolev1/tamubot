"""Retry/backoff wrappers for vision LLM calls.

Adapted from RAG-Anything raganything/resilience.py — LightRAG-specific retry
hooks dropped; callable-agnostic. Uses tenacity for the underlying retry
machinery so behaviour matches existing patterns in tamubot.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


def retry_call(
    fn: Callable,
    *,
    attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 16.0,
    retry_on: Optional[Tuple[Type[BaseException], ...]] = None,
):
    """Wrap fn with exponential-backoff retry.

    Default retries on any Exception. Pass a specific tuple via retry_on to
    narrow (e.g. (httpx.HTTPError, TimeoutError)).
    """
    exception_types = retry_on if retry_on is not None else (Exception,)

    @retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_types),
        reraise=True,
    )
    def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return _wrapped
