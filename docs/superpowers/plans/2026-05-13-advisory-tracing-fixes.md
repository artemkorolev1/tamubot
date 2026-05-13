# Advisory Pipeline Tracing Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three observability defects in the advisory pipeline: orphaned traces, missing session IDs, and oversized recursive answers.

**Architecture:** The advisory pipeline (`advisory/pipeline.py`) calls into the headless RAG graph, whose nodes use `@observe` decorators from Langfuse. These decorators auto-nest under the current OTEL context — but only if a parent trace was created via `start_as_current_observation()`. The fix creates a parent trace inside `run_advisory_pipeline()` so all child spans nest correctly regardless of whether the caller (Streamlit or standalone) sets up OTEL context. For the 102K answer issue, we cap the recursive prompt to limit discovery recommendations.

**Tech Stack:** Langfuse SDK v4, LangGraph, Python, TAMU AI Gateway (Gemini 2.5 Flash)

---

### Task 1: Add trace lifecycle to advisory pipeline

The core issue: `run_advisory_pipeline()` accepts a `trace` param but ignores it. Child `@observe` spans have no parent OTEL context and create orphan top-level traces. Fix: when a `trace` is passed from Streamlit, it already has OTEL context set — do nothing extra. When no trace exists (standalone/test calls), create one internally.

**Files:**
- Modify: `src/tamubot/advisory/pipeline.py`

- [ ] **Step 1: Import tracing helpers and add trace management**

In `src/tamubot/advisory/pipeline.py`, add the trace lifecycle around the graph invocation. When the caller passes a trace (Streamlit path), the OTEL context is already set by `create_trace()` in Streamlit — we just need to ensure it's still active. When no trace exists (standalone), create one internally so `@observe` spans nest.

```python
"""Advisory pipeline entry point.

Convenience wrapper around the advisory orchestrator graph,
analogous to ``rag.graph.pipeline.run_pipeline()``.
"""

from __future__ import annotations

import logging
from typing import Optional

from tamubot.advisory.graph import build_advisory_graph
from tamubot.advisory.state import OrchestratorState

_logger = logging.getLogger("tamubot")

_advisory_graph = None


def _get_advisory_graph():
    global _advisory_graph
    if _advisory_graph is None:
        _advisory_graph = build_advisory_graph()
    return _advisory_graph


def run_advisory_pipeline(
    query: str,
    scp: dict,
    router_result: dict,
    trace=None,
    session_id: str = "",
    history: Optional[list[dict]] = None,
) -> tuple[str, Optional[str]]:
    """Run the advisory orchestrator pipeline.

    Args:
        query: Raw user query.
        scp: Student Context Profile dict with keys:
             scp_program, scp_completed_courses, scp_target_semester, scp_goal.
        router_result: RouterResult fields as dict (from the existing RAG router).
        trace: Langfuse trace (when called from Streamlit, OTEL context is
               already active; when None, this function creates its own trace).
        session_id: Session identifier.
        history: Conversation history.

    Returns:
        (answer, error) — error is None on success.
    """
    initial_state: OrchestratorState = {
        "query": query,
        "session_id": session_id,
        "history": history or [],
        "scp_program": scp.get("scp_program"),
        "scp_completed_courses": scp.get("scp_completed_courses", []),
        "scp_target_semester": scp.get("scp_target_semester"),
        "scp_goal": scp.get("scp_goal"),
        "scp_validated": True,
        "router_result": router_result,
        "rag_answer": "",
        "final_answer": "",
        "error": None,
    }

    _logger.info(
        "advisory pipeline: program=%s, target=%s, courses=%d",
        scp.get("scp_program"),
        scp.get("scp_target_semester"),
        len(scp.get("scp_completed_courses", [])),
    )

    # Create a standalone trace when no caller-provided OTEL context exists.
    # When Streamlit calls us, `trace` is set and OTEL context is already
    # active from create_trace() — skip to avoid double-tracing.
    own_trace = None
    if trace is None:
        from tamubot.rag.observability import create_trace
        from tamubot.rag.observability.config import ObservabilityConfig

        obs = ObservabilityConfig(
            trace_name="tamubot.advisory",
            tags=["advisory"],
            session_id=session_id or None,
        )
        own_trace, _ = create_trace(obs, query)

    try:
        result = _get_advisory_graph().invoke(initial_state)
        answer = result.get("final_answer", "")
        error = result.get("error")
    finally:
        # Finalize only the trace we created (not the caller's)
        if own_trace is not None:
            from tamubot.rag.observability import finalize_trace

            finalize_trace(own_trace, output=answer if not error else f"[error] {error}")

    return answer, error
```

- [ ] **Step 2: Verify traces nest correctly**

Run:
```bash
python -c "
from tamubot.advisory.pipeline import run_advisory_pipeline

answer, error = run_advisory_pipeline(
    query='What CSCE graduate courses focus on AI?',
    scp={'scp_program': 'Computer Science MS', 'scp_completed_courses': [], 'scp_target_semester': 'Fall 2025', 'scp_goal': 'AI focus'},
    router_result={'function': 'semantic_general', 'retrieval_mode': 'semantic', 'course_ids': [], 'rewritten_query': 'CSCE graduate courses AI focus', 'intent_type': 'PLANNING', 'requires_retrieval': True, 'recursive_search': False, 'section': None},
    session_id='test-trace-nesting',
)
print(f'Error: {error}')
print(f'Answer length: {len(answer)}')
"
```

Then check Langfuse:
```bash
python -c "
from tamubot.core import config
import requests

base_url = config.LANGFUSE_BASE_URL.rstrip('/')
auth = (config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY)
resp = requests.get(f'{base_url}/api/public/traces', params={'limit': 5}, auth=auth)
for t in resp.json().get('data', []):
    print(f'{t[\"name\"]} | session={t.get(\"sessionId\")} | tags={t.get(\"tags\")}')
    # Check child observations
    obs_resp = requests.get(f'{base_url}/api/public/observations', params={'traceId': t['id'], 'limit': 20}, auth=auth)
    for o in obs_resp.json().get('data', []):
        print(f'  child: {o[\"name\"]} | type={o.get(\"type\")}')
"
```

Expected: One `tamubot.advisory` trace with `session=test-trace-nesting`, `tags=["advisory"]`, and child observations for `pipeline.retrieval.search.semantic`, `pipeline.retrieval.rerank`, `pipeline.generator`.

- [ ] **Step 3: Commit**

```bash
git add src/tamubot/advisory/pipeline.py
git commit -m "fix: add trace lifecycle to advisory pipeline — orphan spans now nest under parent trace"
```

---

### Task 2: Enforce application-layer token limit on generator output

The TAMU gateway (Gemini 2.5 Flash via SSE) does not enforce `max_tokens` — query 2 produced 102K chars. The TAMU gateway also requires `max_tokens>=4096` or the response is empty, so we cannot reduce the API parameter. Fix: enforce a 2000-token hard cap at the application layer in both `generate()` and `generate_stream()`, using tiktoken for counting. Set a `was_truncated` flag (logged + returned via state) when the cap is hit. Also tighten the recursive prompt to encourage conciseness.

**Files:**
- Modify: `src/tamubot/rag/generator.py`
- Modify: `src/tamubot/rag/prompts.py`

- [ ] **Step 1: Add token-counting truncation helper to generator.py**

At the top of `src/tamubot/rag/generator.py` (after imports), add:

```python
import logging

_logger = logging.getLogger("tamubot")

# Hard cap on generator output tokens. The TAMU gateway ignores max_tokens,
# so we enforce at the application layer. ~2000 tokens ≈ 6-8K chars.
_MAX_OUTPUT_TOKENS = 2000


def _truncate_to_token_limit(text: str, max_tokens: int = _MAX_OUTPUT_TOKENS) -> tuple[str, bool]:
    """Truncate text to max_tokens. Returns (text, was_truncated)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text, False
        truncated = enc.decode(tokens[:max_tokens])
        # Cut at last paragraph or sentence boundary for clean output
        for sep in ("\n\n", "\n", ". "):
            idx = truncated.rfind(sep)
            if idx > len(truncated) // 2:
                truncated = truncated[: idx + len(sep)]
                break
        truncated += "\n\n*[Response truncated — output limit reached]*"
        return truncated, True
    except Exception:
        # Fallback: rough char estimate (4 chars/token)
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text, False
        truncated = text[:char_limit].rsplit("\n", 1)[0]
        truncated += "\n\n*[Response truncated — output limit reached]*"
        return truncated, True
```

- [ ] **Step 2: Apply truncation in generate() (non-streaming path)**

In `generate()`, after `text = collapse_whitespace(text)` (around line 148), add:

```python
    text = collapse_whitespace(text)

    # Enforce token limit (TAMU gateway ignores max_tokens)
    text, was_truncated = _truncate_to_token_limit(text)
    if was_truncated:
        _logger.warning("generate: output truncated at %d tokens (function=%s)", _MAX_OUTPUT_TOKENS, function)
```

- [ ] **Step 3: Apply truncation in generate_stream() (streaming path)**

In `generate_stream()`, replace the token-yielding loop and post-stream section (lines ~284-304) with token-counted streaming:

```python
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
        _logger.warning("generate_stream: output truncated at %d tokens (function=%s)", _MAX_OUTPUT_TOKENS, function)
        yield "\n\n*[Response truncated — output limit reached]*"
        full_text_parts.append("\n\n*[Response truncated — output limit reached]*")

    if usage_out:
        _lf_get_client().update_current_generation(
            usage_details={"input": usage_out[0] or 0, "output": usage_out[1] or 0},
        )

    # Post-stream: run Gate 1 citation check
    complete_text = "".join(full_text_parts)
    validate_citations_with_trace(complete_text, function)
```

- [ ] **Step 4: Tighten the recursive prompt for conciseness**

In `src/tamubot/rag/prompts.py`, update the `recursive` entry in `_FUNCTION_PROMPTS`:

```python
    "recursive": (
        "The student asked about courses in relation to a specific anchor course. "
        "Context includes both the anchor course and related discovered courses. "
        "Answer the student's original question directly: "
        "for discovery questions (what to take after/with/similar to X), recommend the "
        "discovered courses using the anchor only as background context — do not recommend "
        "the anchor course itself as an answer to a discovery query. "
        "For comparison questions (compare X with Y), present a structured comparison of both. "
        "Limit discovery recommendations to at most 3 courses — depth over breadth. "
        "Keep your response under 1500 words. Be concise: summarize key points rather than "
        "reproducing syllabus content verbatim."
    ),
```

- [ ] **Step 5: Commit**

```bash
git add src/tamubot/rag/generator.py src/tamubot/rag/prompts.py
git commit -m "fix: enforce 2000-token output cap in generator — application-layer truncation with flag"
```

---

### Task 3: Propagate session_id to Langfuse traces

All traces (including Streamlit `tamubot.request`) show `sessionId: None`. The `ObservabilityConfig` has `session_id` but `create_trace()` only passes it inside `metadata` — it's never set as the Langfuse trace's `session_id` field. Fix: pass `session_id` to the Langfuse `start_as_current_observation()` call.

**Files:**
- Modify: `src/tamubot/rag/observability/tracing.py`

- [ ] **Step 1: Add session_id to trace creation**

In `src/tamubot/rag/observability/tracing.py`, in `create_trace()`, add the `session_id` kwarg to the `start_as_current_observation()` call:

Current code (around line 80-89):
```python
        kwargs: dict = dict(
            name=obs_config.trace_name,
            input=query,
            metadata=merged_meta,
            end_on_exit=False,
        )
        if trace_id is not None:
            kwargs["trace_context"] = {"trace_id": trace_id}
```

Change to:
```python
        kwargs: dict = dict(
            name=obs_config.trace_name,
            input=query,
            metadata=merged_meta,
            end_on_exit=False,
        )
        if obs_config.session_id:
            kwargs["session_id"] = obs_config.session_id
        if trace_id is not None:
            kwargs["trace_context"] = {"trace_id": trace_id}
```

- [ ] **Step 2: Verify session_id appears in traces**

Run a quick pipeline call and check:
```bash
python -c "
from tamubot.rag.observability import create_trace, finalize_trace
from tamubot.rag.observability.config import ObservabilityConfig

obs = ObservabilityConfig(trace_name='test.session', tags=['test'], session_id='session-123')
trace, tid = create_trace(obs, 'test query')
finalize_trace(trace, output='test output')

from tamubot.core import config
import requests
base_url = config.LANGFUSE_BASE_URL.rstrip('/')
auth = (config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY)
resp = requests.get(f'{base_url}/api/public/traces', params={'limit': 1}, auth=auth)
t = resp.json()['data'][0]
print(f'Name: {t[\"name\"]} | Session: {t.get(\"sessionId\")}')
"
```

Expected: `Name: test.session | Session: session-123`

- [ ] **Step 3: Commit**

```bash
git add src/tamubot/rag/observability/tracing.py
git commit -m "fix: propagate session_id to Langfuse traces — was only stored in metadata"
```
