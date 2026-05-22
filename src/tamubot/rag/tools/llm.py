"""Unified LLM client — routes calls to TAMU AI gateway or Google Gemini direct.

Public API:
    call_llm(messages, ...)   → LLMResult    # blocking, handles SSE accumulation
    stream_llm(messages, ...) → Iterator[str] # yields text tokens

The backend is selected automatically based on config.USE_TAMU_API:
  - True  → TAMU AI gateway (OpenAI-compat, always stream=True due to SSE quirk)
  - False → Google Gemini direct (google-genai SDK)

Guardrails (applied automatically to every call):
  - Output token hard cap: truncates output exceeding LLM_MAX_OUTPUT_TOKENS
  - Input token guard: warns on soft limit, raises on hard limit
  - Retry with exponential backoff for transient errors (429, 503, timeout)
  - Per-call timeout: kills calls exceeding LLM_TIMEOUT_SECONDS

Canonical location: rag/tools/llm.py
"""

import logging
import time
from typing import Iterator, NamedTuple

from google.genai import types

from tamubot.core import config

_logger = logging.getLogger("tamubot")


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _count_tokens_approx(text: str) -> int:
    """Approximate token count using tiktoken cl100k_base (GPT-4 tokeniser).

    Used for the TAMU gateway path where SSE does not expose usage metadata.
    The cl100k_base encoding is a reasonable proxy for Gemini-family models
    (~±10% typical deviation).  Falls back to len/4 if tiktoken is unavailable.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _count_messages_tokens(messages: list[dict]) -> int:
    """Sum approximate token counts across all message contents."""
    return sum(_count_tokens_approx(m.get("content", "")) for m in messages)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class InputTooLargeError(Exception):
    """Raised when input tokens exceed the hard limit."""


def _check_input_guard(messages: list[dict]) -> None:
    """Check input token count against soft/hard limits.

    Logs a warning at soft limit, raises InputTooLargeError at hard limit.
    """
    token_count = _count_messages_tokens(messages)
    if token_count > config.LLM_INPUT_TOKEN_HARD_LIMIT:
        raise InputTooLargeError(
            f"Input tokens ({token_count:,}) exceed hard limit "
            f"({config.LLM_INPUT_TOKEN_HARD_LIMIT:,}). Refusing to send."
        )
    if token_count > config.LLM_INPUT_TOKEN_SOFT_LIMIT:
        _logger.warning(
            "Input tokens (%d) exceed soft limit (%d) — response quality may degrade",
            token_count,
            config.LLM_INPUT_TOKEN_SOFT_LIMIT,
        )


def _truncate_output(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate *text* to at most *max_tokens* tokens.

    Returns ``(text, was_truncated)``.  When truncated the cut is moved back
    to the nearest paragraph / sentence boundary so the response doesn't end
    mid-sentence.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
    except Exception:
        estimated = len(text) // 4
        if estimated <= max_tokens:
            return text, False
        char_limit = max_tokens * 4
        cut = text[:char_limit]
        half = len(cut) // 2
        for sep in ("\n\n", "\n", ". "):
            idx = cut.rfind(sep, half)
            if idx != -1:
                cut = cut[: idx + len(sep)]
                break
        return cut.rstrip() + "\n\n*[Response truncated — output limit reached]*", True

    if len(token_ids) <= max_tokens:
        return text, False

    truncated = enc.decode(token_ids[:max_tokens])
    half = len(truncated) // 2
    for sep in ("\n\n", "\n", ". "):
        idx = truncated.rfind(sep, half)
        if idx != -1:
            truncated = truncated[: idx + len(sep)]
            break
    return truncated.rstrip() + "\n\n*[Response truncated — output limit reached]*", True


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is a transient error worth retrying."""
    # OpenAI SDK errors (TAMU gateway)
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503):
            return True
    except ImportError:
        pass
    # google-genai SDK errors
    exc_name = type(exc).__name__
    if "ResourceExhausted" in exc_name or "ServiceUnavailable" in exc_name:
        return True
    if "DeadlineExceeded" in exc_name or "Timeout" in exc_name:
        return True
    # stdlib timeout
    if isinstance(exc, TimeoutError):
        return True
    return False


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LLMResult(NamedTuple):
    """Return type for call_llm().  Usage fields are approximate (tiktoken) on TAMU path."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None


def _resolved_model(model: str | None) -> str:
    """Pick the model name for a single call.

    Per-call ``model`` wins; otherwise falls back to the global TAMU_MODEL or
    GENERATION_MODEL based on USE_TAMU_API.
    """
    if model:
        return model
    return config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL


def call_llm(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    json_mode: bool = False,
    json_schema: dict | None = None,
    response_schema=None,
    thinking_budget: int = 0,
    model: str | None = None,
) -> LLMResult:
    """Make a blocking LLM call, returning an LLMResult with the full response text.

    Guardrails applied automatically:
      - Input token guard (soft warn / hard reject)
      - Retry with exponential backoff on transient errors
      - Per-call timeout
      - Output token hard cap (truncation)

    Args:
        messages:        List of {"role": ..., "content": ...} dicts.
        temperature:     Sampling temperature (0.0 = deterministic).
        max_tokens:      Max output tokens (use >= 4096; thinking tokens consume
                         budget first — 20 tokens → empty response).
        json_mode:       Request JSON-object output.
        json_schema:     Structured-output schema dict (TAMU json_schema format).
        response_schema: Pydantic model for Gemini structured output.
        thinking_budget: Thinking tokens for Gemini (0 = disabled).

    Returns:
        LLMResult with .text (str) and optional token-usage fields.

    Raises:
        InputTooLargeError: If input exceeds the hard token limit.
        LLMTimeoutError: If the call exceeds LLM_TIMEOUT_SECONDS.
    """
    _check_input_guard(messages)

    last_exc: Exception | None = None
    max_retries = config.LLM_MAX_RETRIES

    for attempt in range(1 + max_retries):
        if attempt > 0:
            backoff = min(2**attempt, 30)
            _logger.warning(
                "LLM call retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                backoff,
                last_exc,
            )
            time.sleep(backoff)

        try:
            result = _call_with_timeout(
                messages,
                temperature,
                max_tokens,
                json_mode,
                json_schema,
                response_schema,
                thinking_budget,
                model,
            )
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries:
                continue
            raise

        # Apply output token hard cap
        output_cap = config.LLM_MAX_OUTPUT_TOKENS
        capped_text, was_truncated = _truncate_output(result.text, output_cap)
        if was_truncated:
            _logger.warning(
                "LLM output truncated: %d → %d tokens (cap=%d)",
                _count_tokens_approx(result.text),
                output_cap,
                output_cap,
            )
        return LLMResult(
            text=capped_text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
        )

    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]


def stream_llm(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    thinking_budget: int = 0,
    usage_out: list | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Streaming LLM call — yields text tokens as they arrive.

    Guardrails applied:
      - Input token guard
      - Retry with exponential backoff on transient errors
      - Output token hard cap (stops consuming stream early)

    Args:
        messages:        List of {"role": ..., "content": ...} dicts.
        temperature:     Sampling temperature.
        max_tokens:      Max output tokens.
        thinking_budget: Thinking tokens for Gemini (0 = disabled).
        usage_out:       Optional list; populated with [input, output, thinking] after stream.

    Yields:
        str: Text tokens as they arrive (stops at output cap).

    Raises:
        InputTooLargeError: If input exceeds the hard token limit.
    """
    _check_input_guard(messages)

    last_exc: Exception | None = None
    max_retries = config.LLM_MAX_RETRIES

    for attempt in range(1 + max_retries):
        if attempt > 0:
            backoff = min(2**attempt, 30)
            _logger.warning(
                "LLM stream retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                backoff,
                last_exc,
            )
            time.sleep(backoff)

        try:
            yield from _stream_with_cap(
                messages,
                temperature,
                max_tokens,
                thinking_budget,
                usage_out,
                model,
            )
            return  # success — exit retry loop
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries:
                continue
            raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Internal dispatch with timeout
# ---------------------------------------------------------------------------


def _call_with_timeout(
    messages,
    temperature,
    max_tokens,
    json_mode,
    json_schema,
    response_schema,
    thinking_budget,
    model,
) -> LLMResult:
    """Dispatch to the correct backend, with optional timeout."""
    timeout = config.LLM_TIMEOUT_SECONDS
    if timeout <= 0:
        return _call_raw(
            messages,
            temperature,
            max_tokens,
            json_mode,
            json_schema,
            response_schema,
            thinking_budget,
            model,
        )

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _call_raw,
            messages,
            temperature,
            max_tokens,
            json_mode,
            json_schema,
            response_schema,
            thinking_budget,
            model,
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise LLMTimeoutError(f"LLM call exceeded timeout of {timeout}s") from None


def _call_raw(
    messages,
    temperature,
    max_tokens,
    json_mode,
    json_schema,
    response_schema,
    thinking_budget,
    model,
) -> LLMResult:
    """Dispatch to TAMU or Gemini backend (no guardrails)."""
    resolved = _resolved_model(model)
    if config.USE_TAMU_API:
        return _call_tamu(messages, temperature, max_tokens, json_mode, json_schema, resolved)
    return _call_gemini(messages, temperature, max_tokens, json_mode, response_schema, thinking_budget, resolved)


def _stream_with_cap(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    usage_out: list | None,
    model: str | None,
) -> Iterator[str]:
    """Stream tokens with output cap — stops consuming once limit is reached."""
    output_cap = config.LLM_MAX_OUTPUT_TOKENS
    accumulated_tokens = 0
    truncated = False

    resolved = _resolved_model(model)
    if config.USE_TAMU_API:
        inner = _stream_tamu(messages, temperature, max_tokens, usage_out=usage_out, model=resolved)
    else:
        inner = _stream_gemini(messages, temperature, max_tokens, thinking_budget, usage_out=usage_out, model=resolved)

    for token_text in inner:
        chunk_tokens = _count_tokens_approx(token_text)
        if accumulated_tokens + chunk_tokens > output_cap and not truncated:
            # Yield whatever fits and stop
            _logger.warning(
                "LLM stream truncated at ~%d tokens (cap=%d)",
                accumulated_tokens,
                output_cap,
            )
            yield "\n\n*[Response truncated — output limit reached]*"
            truncated = True
            # Drain the rest of the stream silently to avoid broken pipe
            for _ in inner:
                pass
            return
        accumulated_tokens += chunk_tokens
        yield token_text


# ---------------------------------------------------------------------------
# TAMU AI gateway paths (OpenAI-compatible SDK)
# ---------------------------------------------------------------------------


def _call_tamu(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    json_schema: dict | None,
    model: str,
) -> LLMResult:
    """Blocking call via TAMU gateway.  Always uses stream=True (gateway SSE quirk)."""
    tamu = config.get_tamu_client()
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    input_tokens = _count_messages_tokens(messages)
    stream = tamu.chat.completions.create(**kwargs)
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    return LLMResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=_count_tokens_approx(text),
    )


def _stream_tamu(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    usage_out: list | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Streaming call via TAMU gateway — yields tokens."""
    tamu = config.get_tamu_client()
    input_tokens = _count_messages_tokens(messages) if usage_out is not None else None
    stream = tamu.chat.completions.create(
        model=model or config.TAMU_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    output_parts: list[str] = [] if usage_out is not None else None  # type: ignore[assignment]
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            if output_parts is not None:
                output_parts.append(token)
            yield token
    if usage_out is not None and output_parts is not None:
        usage_out.extend(
            [
                input_tokens,
                _count_tokens_approx("".join(output_parts)),
                0,  # no thinking tokens on TAMU path
            ]
        )


# ---------------------------------------------------------------------------
# Google Gemini direct paths (google-genai SDK)
# ---------------------------------------------------------------------------


def _extract_messages(messages: list[dict]) -> tuple[str | None, str]:
    """Split messages into (system_instruction, user_content)."""
    if not messages:
        return None, ""
    system_instruction = None
    user_messages = messages
    if messages[0]["role"] == "system":
        system_instruction = messages[0]["content"]
        user_messages = messages[1:]
    return system_instruction, user_messages[-1]["content"] if user_messages else ""


def _build_genai_config(
    system_instruction: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    response_schema,
    thinking_budget: int,
) -> types.GenerateContentConfig:
    kwargs: dict = dict(
        temperature=temperature,
        max_output_tokens=max_tokens,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    elif json_mode:
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def _call_gemini(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    response_schema,
    thinking_budget: int,
    model: str,
) -> LLMResult:
    """Blocking call via Google Gemini direct."""
    system_instruction, contents = _extract_messages(messages)
    cfg = _build_genai_config(system_instruction, temperature, max_tokens, json_mode, response_schema, thinking_budget)
    client = config.get_genai_client()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=cfg,
    )
    text = response.text or ""
    usage = response.usage_metadata
    return LLMResult(
        text=text,
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
        thinking_tokens=getattr(usage, "thoughts_token_count", None) or 0,
    )


def _stream_gemini(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    usage_out: list | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Streaming call via Google Gemini direct — yields tokens."""
    system_instruction, contents = _extract_messages(messages)
    cfg = _build_genai_config(system_instruction, temperature, max_tokens, False, None, thinking_budget)
    client = config.get_genai_client()
    last_chunk = None
    for chunk in client.models.generate_content_stream(
        model=model or config.GENERATION_MODEL,
        contents=contents,
        config=cfg,
    ):
        last_chunk = chunk
        if chunk.text:
            yield chunk.text
    # Populate usage from final chunk metadata (Gemini only)
    if usage_out is not None and last_chunk is not None:
        meta = getattr(last_chunk, "usage_metadata", None)
        if meta:
            usage_out.extend(
                [
                    getattr(meta, "prompt_token_count", None),
                    getattr(meta, "candidates_token_count", None),
                    getattr(meta, "thoughts_token_count", None) or 0,
                ]
            )
