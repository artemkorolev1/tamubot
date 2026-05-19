"""Query router — extracts structured variables from user questions and orchestrates retrieval + reranking.

Stage 1 of the 3-stage RAG pipeline: Router/Inlet → Retrieval+Rerank → Generator/Outlet.

Uses Gemini 2.5 Flash for structured variable extraction (course IDs, categories,
intent type) with query rewriting/expansion.  The retrieval function is derived
mechanically from the extracted variables — there is no intent classification step.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from langfuse import get_client as _lf_get_client
from langfuse import observe

from tamubot.core import config
from tamubot.rag.prompts import ROUTER_PROMPT
from tamubot.rag.tools.llm import call_llm
from tamubot.rag.utils import compute_dynamic_k, normalize_course_id  # noqa: F401 (re-exported)

# ---------------------------------------------------------------------------
# Derivation helpers (pure Python, no LLM)
# ---------------------------------------------------------------------------


def _derive_retrieval_mode(
    course_ids: list[str],
    recursive_search: bool,
) -> str:
    """Derive retrieval_mode from course presence and recursive_search flag.

    - No course IDs → "semantic" (full-corpus vector search, no course filter)
    - recursive_search=True → "hybrid" (two-stage: anchor fetch + corpus-wide discovery)
    - course IDs, no recursive_search → "hybrid_course" (query-driven hybrid per course)
    """
    if not course_ids:
        return "semantic"
    if recursive_search:
        return "hybrid"
    return "hybrid_course"


def _derive_function(
    course_ids: list[str],
    recursive_search: bool,
    intent_type: Optional[str],
    use_summary: bool = False,
) -> str:
    """Derive the retrieval function name from extracted variables.

    Function matrix (v4 — 5 functions):
    course_ids  recursive_search  use_summary  intent_type  → function
    empty       any               any          not None     → semantic_general
    empty       any               any          None         → out_of_scope
    present     True              any          any          → recursive
    present     False             True         any          → course_summary
    present     False             False        any          → hybrid_course
    """
    if not course_ids:
        return "semantic_general" if intent_type is not None else "out_of_scope"
    if recursive_search:
        return "recursive"
    return "course_summary" if use_summary else "hybrid_course"


# ---------------------------------------------------------------------------
# RouterResult — extracted variables + derived fields
# ---------------------------------------------------------------------------


@dataclass
class RouterResult:
    """Structured output from the query router."""

    # Extracted by router LLM
    course_ids: list[str] = field(default_factory=list)
    intent_type: Optional[str] = None  # None = out_of_scope only
    recursive_search: bool = False
    use_summary: bool = False
    rewritten_query: str = ""
    section: Optional[str] = None

    # Derived in Python — auto-computed in __post_init__ if left empty
    retrieval_mode: str = ""
    function: str = ""

    def __post_init__(self):
        if not self.function:
            self.function = _derive_function(
                self.course_ids,
                self.recursive_search,
                self.intent_type,
                self.use_summary,
            )
        if not self.retrieval_mode:
            self.retrieval_mode = _derive_retrieval_mode(self.course_ids, self.recursive_search)

    @property
    def requires_retrieval(self) -> bool:
        return bool(self.course_ids) or self.intent_type is not None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@observe(as_type="generation", name="node.router")
def classify_query(
    query: str,
    prior_course_ids: Optional[list[str]] = None,
    prior_context: Optional[str] = None,
) -> "RouterResult":
    """Extract structured variables from a user query using Gemini Flash.

    Args:
        query:            The raw user question.
        prior_course_ids: Course IDs from the previous turn, prepended as a
                          context hint so the LLM can resolve pronouns like
                          "it" or "that course".
        prior_context:    Full prior-turn context string (query, courses, categories),
                          takes precedence over prior_course_ids if provided.
    """
    if prior_context:
        hint = f"[Context: {prior_context}]\n"
        query = hint + query
    elif prior_course_ids:
        hint = f"[Context: previous turn mentioned courses: {', '.join(prior_course_ids)}]\n"
        query = hint + query
    prompt = ROUTER_PROMPT.format(query=query)

    _lf_get_client().update_current_generation(
        model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
        input=[{"role": "user", "content": prompt}],
        metadata={"prior_context": prior_context or ""},
    )

    llm_result = None
    try:
        llm_result = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
            json_mode=True,
            thinking_budget=512,
        )
        raw_text = llm_result.text
    except Exception:
        raise

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError, ValueError, AttributeError:
        result = RouterResult(rewritten_query=query)
        _lf_get_client().update_current_generation(
            output={"parse_error": True, "function": result.function},
            usage_details={
                "input": llm_result.input_tokens or 0,
                "output": llm_result.output_tokens or 0,
            }
            if llm_result and llm_result.input_tokens is not None
            else None,
        )
        return result

    # Normalize course IDs
    raw_ids = data.get("course_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    course_ids = [normalize_course_id(c) for c in raw_ids if c]

    valid_intent_types = {"ACADEMIC", "CAREER", "DIFFICULTY", "PLANNING", "ADMINISTRATIVE", "GENERAL"}
    intent_type = data.get("intent_type")
    if intent_type not in valid_intent_types:
        intent_type = None

    result = RouterResult(
        course_ids=course_ids,
        intent_type=intent_type,
        recursive_search=bool(data.get("recursive_search", False)),
        use_summary=bool(data.get("use_summary", False)),
        rewritten_query=data.get("rewritten_query", query),
        section=data.get("section"),
        # function and retrieval_mode auto-derived in __post_init__
    )

    _lf_get_client().update_current_generation(
        output=data,
        usage_details={
            "input": llm_result.input_tokens or 0,
            "output": llm_result.output_tokens or 0,
        }
        if llm_result and llm_result.input_tokens is not None
        else None,
        metadata={
            "function": result.function,
            "retrieval_mode": result.retrieval_mode,
            "rewritten_query": result.rewritten_query,
        },
    )

    return result


def deduplicate_chunks(results: list[dict]) -> list[dict]:
    """Keep only the highest-scored chunk per (course_id, category) pair.

    The reranker returns results sorted best-first, so the first occurrence
    of each pair is the most relevant.  Deduplication prevents near-duplicate
    chunks (e.g. same GRADING content across 3 sections) from inflating the
    context and triggering Gemini's whitespace padding bug.
    """
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for doc in results:
        key = (doc.get("course_id", ""), doc.get("chunk_index", -1))
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped
