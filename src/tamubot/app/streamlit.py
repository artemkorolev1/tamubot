import importlib
import logging
import os
import sys
import traceback

import streamlit as st

# Force-reload RAG logic modules so Streamlit hot-reload picks up code changes
# without a full process restart.  Skip stateful infrastructure modules
# (config, mongo, voyage, observability) to preserve DB connections and caches.
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
    }
)

for name, mod in sorted(sys.modules.items()):
    if name.startswith("tamubot.rag.") and name not in _SKIP_RELOAD and mod is not None:
        importlib.reload(mod)

from tamubot.core import config
from tamubot.rag.observability import create_trace, finalize_trace, prod_config

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
# Chat input handling
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about courses, syllabi, or degree requirements..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if USE_MONGODB:
            # --- MongoDB 3-stage pipeline: Route → Retrieve+Rerank → Generate ---

            # Create a parent Langfuse trace for this request
            obs = prod_config(session_id=str(id(st.session_state)))
            lf_trace, _trace_id = create_trace(obs, query=prompt)

            source_docs = []
            router_result = None
            data_gaps: list = []
            data_integrity = True
            conflicted_ids: list = []

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

            # --- answer cache check (skip full pipeline on exact-match hit) ---
            if config.SESSION_CACHE_ENABLED:
                from tamubot.rag.graph.pipeline import get_current_state
                from tamubot.rag.utils import normalize_query as _norm

                _current = get_current_state(thread_config)
                _cached_answer = _current.get("answer_cache", {}).get(_norm(prompt))
                if _cached_answer:
                    answer_placeholder = st.empty()
                    answer_placeholder.markdown(_cached_answer)
                    finalize_trace(lf_trace, output=_cached_answer)
                    st.session_state.messages.append({"role": "assistant", "content": _cached_answer})
                    st.stop()

            answer_tokens: list[str] = []
            source_docs: list[dict] = []
            router_result = None
            with st.spinner("Routing, retrieving, and generating..."):
                try:
                    result = run_pipeline_with_memory(prompt, trace=lf_trace, thread_config=thread_config)
                    source_docs, router_result, data_gaps, data_integrity, conflicted_ids, answer_tokens = result
                    logger.info(
                        f"Router: function={router_result.function}, mode={router_result.retrieval_mode},"
                        f" courses={router_result.course_ids}, docs={len(source_docs)}"
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

            # Close the parent trace and flush all buffered spans
            finalize_trace(lf_trace, output=answer)

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
