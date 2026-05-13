"""Generator (Outlet LLM) — Stage 3 of the 3-stage RAG pipeline.

Takes reranked retrieval results and generates a grounded, cited response
using Gemini Flash with function-adaptive system prompts.
"""

import logging

from langfuse import get_client as _lf_get_client
from langfuse import observe

from tamubot.core import config
from tamubot.rag.gates import validate_citations_with_trace
from tamubot.rag.prompts import (
    _BASE_SYSTEM,
    _FUNCTION_PROMPTS,
    _FUNCTION_TEMPERATURES,
    _SEMANTIC_TYPE_PROMPTS,
    COMPARISON_SYSTEM,
)
from tamubot.rag.tools.context import collapse_whitespace, format_context_xml
from tamubot.rag.tools.llm import call_llm, stream_llm
from tamubot.rag.utils import OOS_FALLBACK as _OUT_OF_SCOPE_RESPONSE

_logger = logging.getLogger("tamubot")

# Hard cap on generator output tokens.  The TAMU gateway (Gemini 2.5 Flash
# via SSE) ignores max_tokens and requires max_tokens>=4096 or the response is
# empty, so we enforce the limit at the application layer instead.
_MAX_OUTPUT_TOKENS = 2000


def _truncate_to_token_limit(text: str, max_tokens: int = _MAX_OUTPUT_TOKENS) -> tuple[str, bool]:
    """Truncate *text* to at most *max_tokens* tokens (cl100k_base).

    Returns ``(text, was_truncated)``.  When truncated the cut is moved back
    to the nearest paragraph / sentence boundary in the second half of the
    allowed window so the response doesn't end mid-sentence.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
    except Exception:
        # Fallback: rough 4-chars-per-token estimate
        estimated = len(text) // 4
        if estimated <= max_tokens:
            return text, False
        char_limit = max_tokens * 4
        cut = text[:char_limit]
        # Try to cut at a clean boundary in the second half
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
    # Move the cut back to a paragraph / sentence boundary in the second half
    half = len(truncated) // 2
    for sep in ("\n\n", "\n", ". "):
        idx = truncated.rfind(sep, half)
        if idx != -1:
            truncated = truncated[: idx + len(sep)]
            break
    return truncated.rstrip() + "\n\n*[Response truncated — output limit reached]*", True


# ---------------------------------------------------------------------------
# Function-adaptive system prompt assembly
# ---------------------------------------------------------------------------


def build_system_prompt(
    function: str,
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
) -> str:
    """Build a function-adaptive system prompt.

    Args:
        function: Router function type (e.g., "hybrid_course", "recursive").
        course_ids: List of course IDs referenced in the query.
        intent_type: Advisory intent type from the router (e.g. "CAREER").
    """
    parts = [_BASE_SYSTEM]
    parts.append(_FUNCTION_PROMPTS.get(function, _FUNCTION_PROMPTS["semantic_general"]))

    # Multi-course comparison overlay
    if course_ids and len(course_ids) > 1:
        parts.append(
            "The user is comparing multiple courses. "
            "For each aspect, describe each course separately, then highlight key differences. "
            "Ensure you cover each course mentioned. "
            "Use [Source N] citations for each piece of information."
        )

    # Advisory overlay for hybrid/semantic functions
    if intent_type and intent_type in _SEMANTIC_TYPE_PROMPTS:
        parts.append(_SEMANTIC_TYPE_PROMPTS[intent_type])

    if course_ids:
        parts.append(f"Courses referenced: {', '.join(course_ids)}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@observe(as_type="generation", name="pipeline.generator")
def generate(
    results: list[dict],
    question: str,
    function: str = "semantic_general",
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
    data_gaps: list[tuple[str, str]] | None = None,
    data_integrity: bool = True,
    history_context: str | None = None,
) -> str:
    """Generate a grounded response with citations using Gemini 2.0 Flash.

    Args:
        results:        Reranked retrieval results (list of chunk dicts).
        question:       The user's original question.
        function:       Retrieval function from the router (e.g. "hybrid_course").
        course_ids:     Extracted course IDs for context.
        intent_type:    Advisory intent type from the router (e.g. "CAREER").
        data_gaps:      [(course_id, category)] pairs missing from DB (recursive only).
        data_integrity: False if any data gaps were found; triggers disclaimer.

    Returns:
        Generated answer string with [Source N] citations.
    """
    # out_of_scope gets a canned response without an LLM call
    if function == "out_of_scope":
        return _OUT_OF_SCOPE_RESPONSE

    # Route multi-course known-course queries to streaming comparison.
    # Recursive queries may have multiple anchor course IDs but need pairing framing, not comparison.
    if course_ids and len(course_ids) > 1 and function != "recursive":
        return "".join(generate_comparison(results, question, course_ids))

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(function, course_ids, intent_type)
    if history_context:
        user_message = (
            f"{context_xml}\n\n"
            f"<conversation_history>\n{history_context}\n</conversation_history>\n\n"
            f"Question: {question}"
        )
    else:
        user_message = f"{context_xml}\n\nQuestion: {question}"

    # Determine thinking budget based on function type and intent.
    # Advisory queries (intent_type set) on hybrid_course also benefit from thinking.
    thinking_budget = (
        config.THINKING_BUDGET_SEMANTIC
        if function in ["recursive", "semantic_general"] or intent_type is not None
        else config.THINKING_BUDGET_METADATA
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    _lf_get_client().update_current_generation(
        model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
        input=messages,
    )
    llm_result = None
    try:
        llm_result = call_llm(
            messages=messages,
            temperature=_FUNCTION_TEMPERATURES.get(function, 0.1),
            max_tokens=4096,
            thinking_budget=thinking_budget,
        )
        text = llm_result.text
    except Exception:
        raise
    _lf_get_client().update_current_generation(
        output=text,
        usage_details={
            "input": llm_result.input_tokens or 0,
            "output": llm_result.output_tokens or 0,
        }
        if llm_result and llm_result.input_tokens is not None
        else None,
    )
    text = collapse_whitespace(text)

    text, was_truncated = _truncate_to_token_limit(text)
    if was_truncated:
        _logger.warning("generate: output truncated at %d tokens (function=%s)", _MAX_OUTPUT_TOKENS, function)

    # Prepend data integrity disclaimer when DB chunks are missing
    if not data_integrity and data_gaps:
        gap_lines = "\n".join(f"- {cid} / {cat}" for cid, cat in data_gaps)
        disclaimer = f"⚠️ Note: The following data was not found in the syllabus database:\n{gap_lines}\n\n"
        text = disclaimer + text

    # Gate 1: Validate citations in response
    validate_citations_with_trace(text, function)

    # Gate 2 (groundedness scoring) intentionally disabled — uses LLM on every query.

    return text


@observe(as_type="generation", name="pipeline.generator.comparison")
def generate_comparison(
    results: list[dict],
    question: str,
    course_ids: list[str],
):
    """Stream a multi-course comparison as free-form markdown.

    Uses stream_llm (no constrained decoding) so tokens arrive immediately.

    Args:
        results:    Reranked retrieval results (list of chunk dicts).
        question:   The user's original question.
        course_ids: List of course IDs being compared (len > 1).

    Yields:
        str: Text tokens as they arrive.
    """
    context_xml = format_context_xml(results)
    courses_list = ", ".join(course_ids)
    user_message = f"{context_xml}\n\nQuestion: {question}\n\nCompare the following courses: {courses_list}."
    messages = [
        {"role": "system", "content": COMPARISON_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    _lf_get_client().update_current_generation(
        model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
        input=messages,
    )
    usage_out: list = []
    for token in stream_llm(
        messages=messages,
        temperature=0.2,
        max_tokens=4096,
        usage_out=usage_out,
    ):
        yield token
    if usage_out:
        _lf_get_client().update_current_generation(
            usage_details={"input": usage_out[0] or 0, "output": usage_out[1] or 0},
        )


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------


@observe(as_type="generation", name="pipeline.generator")
def generate_stream(
    results: list[dict],
    question: str,
    function: str = "semantic_general",
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
    data_gaps: list[tuple[str, str]] | None = None,
    data_integrity: bool = True,
    conflicted_course_ids: list[str] | None = None,
    history_context: str | None = None,
):
    """Streaming variant of generate(). Yields text chunks as they arrive.

    Single-course queries stream tokens directly from Gemini using
    generate_content_stream(). Multi-course comparison queries also stream
    via generate_comparison() using free-form markdown (no constrained decoding).

    Args:
        results:        Reranked retrieval results (list of chunk dicts).
        question:       The user's original question.
        function:       Retrieval function from the router.
        course_ids:     Extracted course IDs for context.
        intent_type:    Advisory intent type from the router (e.g. "CAREER").
        data_gaps:      [(course_id, category)] pairs missing from DB (recursive only).
        data_integrity: False if any data gaps were found; triggers disclaimer.

    Yields:
        str: Text chunks as they arrive from the model.
    """
    # out_of_scope: yield canned response immediately
    if function == "out_of_scope":
        yield _OUT_OF_SCOPE_RESPONSE
        return

    # Multi-course known-course comparison: stream free-form markdown directly.
    # Recursive queries with multiple anchors stream normally using their own prompts.
    if course_ids and len(course_ids) > 1 and function != "recursive":
        yield from generate_comparison(results, question, course_ids)
        return

    # Prepend data integrity disclaimer before streaming begins
    if not data_integrity and data_gaps:
        gap_lines = "\n".join(f"- {cid} / {cat}" for cid, cat in data_gaps)
        disclaimer = f"⚠️ Note: The following data was not found in the syllabus database:\n{gap_lines}\n\n"
        yield disclaimer

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(function, course_ids, intent_type)
    if history_context:
        user_message = (
            f"{context_xml}\n\n"
            f"<conversation_history>\n{history_context}\n</conversation_history>\n\n"
            f"Question: {question}"
        )
    else:
        user_message = f"{context_xml}\n\nQuestion: {question}"

    thinking_budget = (
        config.THINKING_BUDGET_SEMANTIC
        if function in ["recursive", "semantic_general"] or intent_type is not None
        else config.THINKING_BUDGET_METADATA
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    _lf_get_client().update_current_generation(
        model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
        input=messages,
    )
    full_text_parts: list[str] = []
    usage_out: list = []
    was_truncated = False

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    token_count = 0
    for token in stream_llm(
        messages=messages,
        temperature=_FUNCTION_TEMPERATURES.get(function, 0.1),
        max_tokens=4096,
        thinking_budget=thinking_budget,
        usage_out=usage_out,
    ):
        if enc is not None:
            token_count += len(enc.encode(token))
            if token_count > _MAX_OUTPUT_TOKENS:
                was_truncated = True
                break
        full_text_parts.append(token)
        yield token

    if was_truncated:
        _logger.warning(
            "generate_stream: output truncated at %d tokens (function=%s)",
            _MAX_OUTPUT_TOKENS,
            function,
        )
        yield "\n\n*[Response truncated — output limit reached]*"
        full_text_parts.append("\n\n*[Response truncated — output limit reached]*")

    if usage_out:
        _lf_get_client().update_current_generation(
            usage_details={"input": usage_out[0] or 0, "output": usage_out[1] or 0},
        )

    # Post-stream: run Gate 1 citation check
    complete_text = "".join(full_text_parts)
    validate_citations_with_trace(complete_text, function)

    # Gate 2 (groundedness scoring) intentionally disabled — uses LLM on every query.
