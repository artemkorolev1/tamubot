"""Query router — extracts structured variables from user questions and orchestrates retrieval + reranking.

Stage 1 of the 3-stage RAG pipeline: Router/Inlet → Retrieval+Rerank → Generator/Outlet.

Uses Gemini 2.5 Flash for structured variable extraction (course IDs, categories,
intent type) with query rewriting/expansion.  The retrieval function is derived
mechanically from the extracted variables — there is no intent classification step.

Post-LLM repairs (course-ID grounding, flag combos, subquery dedupe/fallback) run
deterministically in ``_validate_and_repair`` — no extra LLM call. Every repair is
logged to the Langfuse generation metadata so eval state dumps can record what
was changed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from langfuse import get_client as _lf_get_client
from langfuse import observe

from tamubot.core import config
from tamubot.rag import course_index
from tamubot.rag.prompts import ROUTER_PROMPT
from tamubot.rag.tools.llm import call_llm
from tamubot.rag.utils import (
    compute_dynamic_k,  # noqa: F401 (re-exported)
    normalize_course_id,
    normalize_query,
)

VALID_INTENT_TYPES: frozenset[str] = frozenset(
    {"ACADEMIC", "CAREER", "DIFFICULTY", "PLANNING", "ADMINISTRATIVE", "GENERAL"}
)

MAX_SUBQUERIES = 4

# Discovery questions ("which course teaches X?") mention course titles as
# topics, not as the subject of the question. Title-substring recovery is
# harmful here — it pins the query to a single course and prevents
# semantic_general from surfacing related courses.
_DISCOVERY_QUERY_RE = re.compile(
    r"\b(which|what)\s+(course|courses|class|classes)\b",
    re.IGNORECASE,
)

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
) -> str:
    """Derive the retrieval function name from extracted variables.

    Function matrix (v4 — 4 functions):
    course_ids  recursive_search  intent_type  → function
    empty       any               not None     → semantic_general
    empty       any               None         → out_of_scope
    present     True              any          → recursive
    present     False             any          → hybrid_course
    """
    if not course_ids:
        return "semantic_general" if intent_type is not None else "out_of_scope"
    if recursive_search:
        return "recursive"
    return "hybrid_course"


# ---------------------------------------------------------------------------
# Validator / repair — post-LLM, no network calls except COURSE_INDEX (Mongo,
# loaded lazily once per process). Returns the cleaned dict plus a list of
# applied repairs for telemetry.
# ---------------------------------------------------------------------------


def _validate_and_repair(raw: dict, query: str) -> tuple[dict, dict]:
    """Repair LLM output deterministically.

    Steps (in order):
      1. Normalize raw course_ids; drop any not in COURSE_INDEX. Recover from
         a title substring match against the query when all are dropped.
      2. If ``recursive_search=True`` but no course_ids survive, flip to False.
      3. Dedupe ``subqueries`` (normalize_query) and cap at MAX_SUBQUERIES.
      4. Fallback chain so subqueries is never empty:
            [rewritten_query] → [original query].
      5. Validate ``intent_type`` against the whitelist (else None).
      6. Set ``rewritten_query = subqueries[0]`` for back-compat.

    Returns ``(cleaned, telemetry)`` where ``telemetry`` carries
    ``dropped_course_ids`` and ``repairs_applied`` for state-dump capture.
    """
    repairs: list[str] = []
    dropped_course_ids: list[str] = []

    # --- 1. course_ids: normalize + ground against index ---
    raw_ids = raw.get("course_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    normalized = [normalize_course_id(c) for c in raw_ids if c]
    kept: list[str] = []
    for cid in normalized:
        if cid and course_index.is_known_course_id(cid):
            if cid not in kept:
                kept.append(cid)
        elif cid:
            dropped_course_ids.append(cid)
    if dropped_course_ids:
        repairs.append("dropped_unreal_course_ids")

    # Last-ditch: if all dropped (or LLM emitted none) and the query mentions a
    # known course title, recover one course_id. Skipped for discovery questions
    # ("which course teaches X?") where the title appears as a topic.
    if not kept:
        if _DISCOVERY_QUERY_RE.search(query):
            repairs.append("recovery_skipped_discovery_query")
        else:
            recovered = course_index.match_title(query)
            if recovered:
                kept.append(recovered)
                repairs.append("course_id_recovered_from_title")

    # --- 2. impossible flag combo: recursive without anchor ---
    recursive_search = bool(raw.get("recursive_search", False))
    if recursive_search and not kept:
        recursive_search = False
        repairs.append("rs_flipped_to_false")

    # --- 3 + 4. subqueries: dedupe, cap, never-empty ---
    rewritten_query = raw.get("rewritten_query") or ""
    raw_subqueries = raw.get("subqueries")
    if isinstance(raw_subqueries, str):
        raw_subqueries = [raw_subqueries]
    candidates: list[str] = [s for s in (raw_subqueries or []) if isinstance(s, str) and s.strip()]
    if rewritten_query:
        # Guarantee the canonical rewrite is the first entry (back-compat).
        candidates.insert(0, rewritten_query)
    candidates = [s.strip() for s in candidates]

    # Dedupe by normalized form.
    seen: set[str] = set()
    subqueries: list[str] = []
    for s in candidates:
        key = normalize_query(s)
        if key and key not in seen:
            seen.add(key)
            subqueries.append(s)
    if len(candidates) > len(subqueries):
        repairs.append("subqueries_deduped")
    if len(subqueries) > MAX_SUBQUERIES:
        subqueries = subqueries[:MAX_SUBQUERIES]
        repairs.append("subqueries_capped")

    if not subqueries:
        fallback = rewritten_query or query
        subqueries = [fallback]
        repairs.append("subqueries_fallback")

    # --- 5. intent_type whitelist ---
    intent_type = raw.get("intent_type")
    if intent_type not in VALID_INTENT_TYPES:
        if intent_type is not None:
            repairs.append("intent_type_rejected")
        intent_type = None

    # --- 6. rewritten_query := subqueries[0] for back-compat ---
    cleaned = {
        "course_ids": kept,
        "intent_type": intent_type,
        "recursive_search": recursive_search,
        "rewritten_query": subqueries[0],
        "section": raw.get("section"),
        "subqueries": subqueries,
    }
    telemetry = {
        "dropped_course_ids": dropped_course_ids,
        "repairs_applied": repairs,
    }
    return cleaned, telemetry


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
    rewritten_query: str = ""
    section: Optional[str] = None
    subqueries: list[str] = field(default_factory=list)

    # Derived in Python — auto-computed in __post_init__ if left empty
    retrieval_mode: str = ""
    function: str = ""

    # Repair telemetry — populated by classify_query when applicable
    dropped_course_ids: list[str] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.subqueries:
            self.subqueries = [self.rewritten_query] if self.rewritten_query else []
        elif not self.rewritten_query:
            self.rewritten_query = self.subqueries[0]
        if not self.function:
            self.function = _derive_function(
                self.course_ids,
                self.recursive_search,
                self.intent_type,
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
    original_query = query
    if prior_context:
        hint = f"[Context: {prior_context}]\n"
        query = hint + query
    elif prior_course_ids:
        hint = f"[Context: previous turn mentioned courses: {', '.join(prior_course_ids)}]\n"
        query = hint + query
    prompt = ROUTER_PROMPT.format(query=query)

    router_model = config.ROUTER_MODEL or (config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL)
    _lf_get_client().update_current_generation(
        model=router_model,
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
            model=config.ROUTER_MODEL,
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
            metadata={"parse_error": True},
        )
        return result

    cleaned, telemetry = _validate_and_repair(data, original_query)

    result = RouterResult(
        course_ids=cleaned["course_ids"],
        intent_type=cleaned["intent_type"],
        recursive_search=cleaned["recursive_search"],
        rewritten_query=cleaned["rewritten_query"],
        section=cleaned["section"],
        subqueries=cleaned["subqueries"],
        dropped_course_ids=telemetry["dropped_course_ids"],
        repairs_applied=telemetry["repairs_applied"],
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
            "subqueries": result.subqueries,
            "dropped_course_ids": telemetry["dropped_course_ids"],
            "repairs_applied": telemetry["repairs_applied"],
        },
    )

    return result


def deduplicate_chunks(results: list[dict]) -> list[dict]:
    """Keep only the highest-scored chunk per (course_id, chunk_index) pair.

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
