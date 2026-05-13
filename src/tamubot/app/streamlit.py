import importlib
import logging
import os
import sys
import traceback

import streamlit as st

# Force-reload RAG logic modules so Streamlit hot-reload picks up code changes
# without a full process restart.  Skip stateful infrastructure modules
# (config, mongo, voyage, observability, graph infra) to preserve DB
# connections, caches, and the in-memory checkpointer that holds conversation
# history across Streamlit reruns.
_SKIP_RELOAD = frozenset(
    {
        "tamubot.core.config",
        "tamubot.rag.tools.mongo",
        "tamubot.rag.tools.voyage",
        "tamubot.rag.tools.mem0",
        "tamubot.rag.observability",
        "tamubot.rag.observability.tracing",
        "tamubot.rag.observability.config",
        "tamubot.rag.observability.evals",
        "tamubot.rag.graph.pipeline",
        "tamubot.rag.graph.builder",
        "tamubot.rag.graph.checkpointer",
    }
)

for name, mod in sorted(sys.modules.items()):
    if name.startswith("tamubot.rag.") and name not in _SKIP_RELOAD and mod is not None:
        importlib.reload(mod)

from tamubot.core import config
from tamubot.rag.observability import prod_config

# Intent types that trigger the advisory orchestrator
ADVISORY_INTENTS = {"PLANNING", "CAREER"}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tamubot")

st.set_page_config(page_title="TamuBot", page_icon="🤖", layout="wide")

st.title("🤖 TamuBot — Texas A&M Academic Assistant")
st.markdown("Ask questions about courses, syllabi, degree requirements, and university policies.")

if config.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = f"TamuBot-{config.APP_MODE}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

USE_MONGODB = config.RETRIEVAL_BACKEND == "mongodb"

_session_manager = None

if USE_MONGODB:
    from tamubot.rag.graph.pipeline import run_pipeline_with_memory
    from tamubot.rag.graph.session import SessionManager
    from tamubot.rag.tools.mongo import get_syllabus_urls

    _session_manager = SessionManager()
else:
    from typing import Any, List

    import vertexai
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_google_vertexai import ChatVertexAI
    from vertexai.preview import rag


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Backend:** {'MongoDB Atlas' if USE_MONGODB else 'Vertex AI RAG'}")
    st.write(f"**Model:** {config.MODEL_NAME}")

    if USE_MONGODB:
        st.info("Using MongoDB Atlas hybrid search + Voyage AI reranking")
    else:
        st.write(f"**Project ID:** {config.PROJECT_ID}")
        st.write(f"**RAG Region:** {config.RETRIEVAL_REGION}")
        st.write(f"**LLM Region:** {config.GENERATION_REGION}")
        st.info("Using Vertex AI Managed RAG Service")


# ---------------------------------------------------------------------------
# Vertex AI legacy path (SYSTEM_PROMPT used only by Vertex)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are TamuBot, an academic assistant for Texas A&M University.
Answer the question based only on the following context. If the context does not contain
enough information, say so clearly rather than guessing.

Context:
{context}

Question: {question}
"""

if not USE_MONGODB:

    class VertexRagRetriever(BaseRetriever):
        project_id: str
        location: str
        rag_corpus_resource_name: str

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
            try:
                vertexai.init(project=self.project_id, location=self.location)
                response = rag.retrieval_query(
                    rag_resources=[rag.RagResource(rag_corpus=self.rag_corpus_resource_name)],
                    text=query,
                    similarity_top_k=5,
                )
                documents = []
                if hasattr(response, "contexts") and hasattr(response.contexts, "contexts"):
                    for context in response.contexts.contexts:
                        documents.append(
                            Document(
                                page_content=context.text,
                                metadata={"source": context.source_uri, "score": context.score},
                            )
                        )
                return documents
            except Exception as e:
                st.error(f"Error retrieving documents: {e}")
                return []

    @st.cache_resource
    def get_rag_chain():
        try:
            llm = ChatVertexAI(
                model=config.MODEL_NAME, temperature=0.2, project=config.PROJECT_ID, location=config.GENERATION_REGION
            )
            retriever = VertexRagRetriever(
                project_id=config.PROJECT_ID,
                location=config.RETRIEVAL_REGION,
                rag_corpus_resource_name=config.RAG_CORPUS_RESOURCE_NAME,
            )
            template = SYSTEM_PROMPT
            prompt = ChatPromptTemplate.from_template(template)
            generation_chain = prompt | llm
            return generation_chain, retriever
        except Exception as e:
            st.error(f"Error initializing RAG Chain: {e}")
            return None, None

    rag_chain, retriever = get_rag_chain()


# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Advisory SCP form — rendered at top level so form submission is always processed
# ---------------------------------------------------------------------------


def _semester_options() -> list[str]:
    """Generate semester options for the next 2 years."""
    import datetime

    now = datetime.date.today()
    options = []
    for year in range(now.year, now.year + 3):
        for term in ("Spring", "Summer", "Fall"):
            options.append(f"{term} {year}")
    return options


# The SCP form must be rendered on EVERY rerun while it's active, not just when
# prompt is set.  Otherwise Streamlit's form-submission rerun can't find the
# form widget and the submitted data is lost.
prompt = None  # will be set below by form handler or chat_input

if st.session_state.get("scp_form_active") and not st.session_state.get("scp_validated"):
    from tamubot.advisory.program_registry import PROGRAM_COURSES

    st.info(
        "This looks like an academic planning question. "
        "Please share some details so I can give you a personalized answer."
    )
    with st.form("scp_form"):
        st.subheader("Student Context Profile")
        program = st.selectbox(
            "Degree program",
            options=[""] + list(PROGRAM_COURSES.keys()),
            index=0,
            help="Select your degree program",
        )
        completed = st.text_input(
            "Completed courses (comma-separated)",
            placeholder="e.g. CSCE 121, CSCE 221, MATH 251",
        )
        semester = st.selectbox(
            "Target semester",
            options=[""] + _semester_options(),
            index=0,
        )
        goal = st.text_input(
            "Academic goal (optional)",
            placeholder="e.g. graduate on time, prepare for grad school",
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.scp_program = program or None
        st.session_state.scp_completed_courses = (
            [c.strip().upper() for c in completed.split(",") if c.strip()] if completed else []
        )
        st.session_state.scp_target_semester = semester or None
        st.session_state.scp_goal = goal or None
        st.session_state.scp_validated = True
        # Re-inject the pending query so the advisory pipeline runs
        prompt = st.session_state.get("scp_pending_query")
    else:
        # Form visible but not yet submitted — stop here
        st.stop()

# ---------------------------------------------------------------------------
# Advisory: pick up pending query after SCP form validation
# ---------------------------------------------------------------------------

_advisory_router = st.session_state.get("scp_router_result")
_advisory_trace_id = st.session_state.get("scp_trace_id")

# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------

_is_scp_rerun = prompt is not None  # True when form was just submitted

if not prompt:
    prompt = st.chat_input("Ask about courses, syllabi, or degree requirements...")

if prompt and not _is_scp_rerun:
    # New user input — append to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

if prompt:
    with st.chat_message("assistant"):
        if USE_MONGODB:
            # --- MongoDB 3-stage pipeline: Route → Retrieve+Rerank → Generate ---
            from tamubot.rag.observability import trace_context

            obs = prod_config(session_id=str(id(st.session_state)))

            # Resolve thread_config early (needed by cache check and pipeline)
            # Use session_state to store thread_id so it's stable across Streamlit reruns
            # (id(st.session_state) changes each rerun, breaking the checkpointer)
            if "thread_id" not in st.session_state:
                import uuid as _uuid

                st.session_state.thread_id = str(_uuid.uuid4())
            thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Initialize Mem0Manager once per session
            if config.MEM0_ENABLED and "mem0_manager" not in st.session_state:
                try:
                    from tamubot.rag.tools import mem0 as mem0_registry
                    from tamubot.rag.tools.mem0 import Mem0Manager

                    _thread_id = thread_config.get("configurable", {}).get("thread_id", "")
                    st.session_state.mem0_manager = Mem0Manager(_thread_id)
                    mem0_registry.register(_thread_id, st.session_state.mem0_manager)
                except Exception as _mem0_err:
                    import logging as _log

                    _log.getLogger("tamubot").warning(f"mem0 initialization failed (non-fatal): {_mem0_err}")

            # trace_context guarantees OTEL cleanup even when st.stop()/st.rerun()
            # raise exceptions. On advisory rerun, resume_trace_id appends to the
            # existing trace so the SCP-collection → pipeline sequence is one trace.
            with trace_context(obs, prompt, trace_id=_advisory_trace_id) as (lf_trace, _trace_id):
                # --- answer cache check (skip full pipeline on exact-match hit) ---
                if config.SESSION_CACHE_ENABLED:
                    from tamubot.rag.graph.pipeline import get_current_state
                    from tamubot.rag.utils import normalize_query as _norm

                    _current = get_current_state(thread_config)
                    _cached_answer = _current.get("answer_cache", {}).get(_norm(prompt))
                    if _cached_answer:
                        answer_placeholder = st.empty()
                        answer_placeholder.markdown(_cached_answer)
                        if lf_trace is not None:
                            lf_trace.update(output=_cached_answer)
                        st.session_state.messages.append({"role": "assistant", "content": _cached_answer})
                        st.stop()

                # --- Step 1: Classify query ---
                # If we're resuming after SCP form submission, use the stored router
                # result instead of re-classifying (avoids a second LLM call that
                # might return a different intent type).
                from tamubot.rag.router import classify_query

                router_result = None
                rr_dict = _advisory_router  # None on first pass, dict on SCP rerun

                if rr_dict is not None:
                    # Rerun after SCP form — router result was stored in session_state
                    logger.info("Advisory rerun: using stored router result, skipping re-classification")
                else:
                    with st.spinner("Classifying your question..."):
                        try:
                            router_result = classify_query(prompt)
                            logger.info(
                                f"Router: function={router_result.function}, mode={router_result.retrieval_mode},"
                                f" courses={router_result.course_ids}, intent={router_result.intent_type}"
                            )
                            # Pre-serialize to dict for both advisory and standard paths
                            rr_dict = {
                                "function": router_result.function,
                                "course_ids": router_result.course_ids,
                                "retrieval_mode": router_result.retrieval_mode,
                                "intent_type": router_result.intent_type,
                                "rewritten_query": router_result.rewritten_query,
                                "recursive_search": router_result.recursive_search,
                                "requires_retrieval": router_result.requires_retrieval,
                                "section": router_result.section,
                            }
                        except Exception as e:
                            logger.error(f"Router failed: {traceback.format_exc()}")
                            st.error(f"Classification failed: {e}")

                # --- Step 2: Advisory dispatch (before running any pipeline) ---
                is_advisory = rr_dict is not None and rr_dict.get("intent_type") in ADVISORY_INTENTS

                if is_advisory and not st.session_state.get("scp_validated"):
                    # Activate the SCP form (rendered at top level) and stop.
                    # Store the query, router result, and trace_id so the form
                    # submission rerun can resume the advisory pipeline.
                    st.session_state.scp_form_active = True
                    st.session_state.scp_pending_query = prompt
                    st.session_state.scp_router_result = rr_dict
                    st.session_state.scp_trace_id = _trace_id
                    if lf_trace is not None:
                        lf_trace.update(output="[SCP form shown — awaiting student profile]")
                    st.rerun()

                if is_advisory and st.session_state.get("scp_validated"):
                    # Run advisory pipeline with collected SCP
                    from tamubot.advisory.pipeline import run_advisory_pipeline

                    scp = {
                        "scp_program": st.session_state.get("scp_program"),
                        "scp_completed_courses": st.session_state.get("scp_completed_courses", []),
                        "scp_target_semester": st.session_state.get("scp_target_semester"),
                        "scp_goal": st.session_state.get("scp_goal"),
                    }
                    session_id = thread_config.get("configurable", {}).get("thread_id", "")
                    answer = ""
                    try:
                        with st.spinner("Generating personalized advisory answer..."):
                            advisory_answer, advisory_error = run_advisory_pipeline(
                                query=prompt,
                                scp=scp,
                                router_result=rr_dict,
                                trace=lf_trace,
                                session_id=session_id,
                            )
                        answer = advisory_answer or ""
                        if advisory_error:
                            st.warning(f"Advisory pipeline error: {advisory_error}")
                    except Exception as e:
                        logger.error(f"Advisory pipeline failed: {traceback.format_exc()}")
                        st.error(f"Advisory pipeline failed: {e}")

                    answer_placeholder = st.empty()
                    answer_placeholder.markdown(answer)
                    if lf_trace is not None:
                        lf_trace.update(output=answer or "[advisory error]")
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Clear SCP form state so the next query starts fresh
                    for key in (
                        "scp_form_active",
                        "scp_pending_query",
                        "scp_router_result",
                        "scp_trace_id",
                        "scp_validated",
                    ):
                        st.session_state.pop(key, None)

                else:
                    # --- Step 3: Standard RAG pipeline (non-advisory queries) ---
                    answer_tokens: list[str] = []
                    source_docs: list[dict] = []
                    with st.spinner("Retrieving and generating..."):
                        try:
                            result = run_pipeline_with_memory(prompt, thread_config=thread_config)
                            source_docs, router_result, data_gaps, data_integrity, conflicted_ids, answer_tokens = (
                                result
                            )
                        except Exception as e:
                            logger.error(f"Retrieval failed: {traceback.format_exc()}")
                            st.error(f"Retrieval failed: {e}")

                    answer = ""
                    answer_placeholder = st.empty()
                    for token in answer_tokens:
                        answer += token
                        answer_placeholder.markdown(answer + "▌")
                    answer_placeholder.markdown(answer)
                    logger.info(f"Generation complete, answer length: {len(answer)}")

                    # Render syllabus links for all retrieved courses
                    if source_docs:
                        course_ids = list({doc["course_id"] for doc in source_docs if doc.get("course_id")})
                        try:
                            url_map = get_syllabus_urls(course_ids)
                        except Exception:
                            url_map = {}
                        if url_map:
                            links = "  ".join(f"[{cid} Syllabus]({url})" for cid, url in sorted(url_map.items()))
                            answer_placeholder.markdown(answer + "\n\n---\n**Syllabi:** " + links)
                            answer += "\n\n---\n**Syllabi:** " + links

                    # Set trace output before context manager exits
                    if lf_trace is not None:
                        lf_trace.update(output=answer)

                    if source_docs:
                        with st.expander("View Source Documents", expanded=False):
                            if router_result:
                                mode_label = router_result.retrieval_mode
                                sem = f" | Intent: {router_result.intent_type}" if router_result.intent_type else ""
                                st.caption(
                                    f"Function: **{router_result.function}** | "
                                    f"Mode: {mode_label}{sem} | "
                                    f"Courses: {', '.join(router_result.course_ids) or 'none'}"
                                )
                            for i, doc in enumerate(source_docs):
                                label = doc.get("course_id", doc.get("policy_name", "Unknown"))
                                st.write(f"**Source {i + 1}:** {label}")
                                if doc.get("category"):
                                    st.write(f"*Category: {doc['category']}*")
                                content = doc.get("content", doc.get("policy_name", ""))
                                st.info(content[:500] + ("..." if len(content) > 500 else ""))
                                st.write("---")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

        else:
            # --- Vertex AI legacy path ---
            source_docs = []
            with st.spinner("Retrieving information..."):
                try:
                    source_docs = retriever.invoke(prompt)
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")

            answer = ""
            if rag_chain and source_docs:
                with st.spinner("Generating answer..."):
                    try:
                        vertexai.init(project=config.PROJECT_ID, location=config.GENERATION_REGION)
                        response = rag_chain.invoke(
                            {"context": "\n\n".join([d.page_content for d in source_docs]), "question": prompt}
                        )
                        answer = response.content
                    except Exception as e:
                        st.warning("Generative model unavailable. Showing retrieved documents.")
                        st.caption(f"Error: {str(e)[:100]}...")
                        answer = "**Relevant documents found:**"
            elif not source_docs:
                answer = "No relevant information found in the knowledge base."

            st.markdown(answer)

            if source_docs:
                with st.expander("View Source Documents", expanded=False):
                    for i, doc in enumerate(source_docs):
                        st.write(f"**Source {i + 1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"*Score: {doc.metadata.get('score', 'N/A')}*")
                        st.info(doc.page_content)
                        st.write("---")

            st.session_state.messages.append({"role": "assistant", "content": answer})
