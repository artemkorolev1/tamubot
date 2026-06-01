import os
import threading
import time

from dotenv import load_dotenv

load_dotenv(override=True)

# LLM_PROVIDER switch: "tamu" or "gemini" (default).
# Set in .env to choose which backend RAG LLM calls use.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# --- GCP / Vertex AI (legacy, kept as fallback) ---
PROJECT_ID = os.getenv("PROJECT_ID", "glossy-surge-486017-g8")
RETRIEVAL_REGION = os.getenv("RETRIEVAL_REGION", "us-south1")
GENERATION_REGION = os.getenv("GENERATION_REGION", "us-central1")
RAG_CORPUS_RESOURCE_NAME = os.getenv(
    "RAG_CORPUS_RESOURCE_NAME", "projects/glossy-surge-486017-g8/locations/us-south1/ragCorpora/2305843009213693952"
)

# --- MongoDB Atlas ---
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "tamubot")

# --- Voyage AI (embeddings + reranking) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2")

# --- Google AI (Gemini for generation + router) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "gemini-2.5-flash-lite")

# --- Google AI rate limiter ---
GOOGLE_API_RPM: int = int(os.getenv("GOOGLE_API_RPM", "30"))

# --- TAMU AI API (OpenAI-compatible gateway; data privacy + institutional billing) ---
TAMU_API_KEY = os.getenv("TAMU_API_KEY")
TAMU_BASE_URL = os.getenv("TAMU_BASE_URL", "https://chat-api.tamu.ai/openai")
TAMU_MODEL = os.getenv("TAMU_MODEL", "protected.gemini-2.5-flash")
# Derived from LLM_PROVIDER switch in .env ("tamu" or "gemini").
# ingestion_pipeline/process_syllabi.py is excluded (uses PDF multimodal input).
USE_TAMU_API: bool = LLM_PROVIDER == "tamu"

# Per-stage model overrides. Each defaults to None → call_llm falls back to
# TAMU_MODEL (on TAMU path) or GENERATION_MODEL (on Gemini path).
ROUTER_MODEL: str | None = os.getenv("ROUTER_MODEL") or None
GENERATOR_MODEL: str | None = os.getenv("GENERATOR_MODEL") or None
HISTORY_UPDATE_MODEL: str | None = os.getenv("HISTORY_UPDATE_MODEL") or None
RECURSIVE_ROUTER_MODEL: str | None = os.getenv("RECURSIVE_ROUTER_MODEL") or None
OUT_OF_SCOPE_MODEL: str | None = os.getenv("OUT_OF_SCOPE_MODEL") or None

# --- LLM guardrails ---
# Hard cap on output tokens (application-layer; TAMU gateway ignores max_tokens).
LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8000"))
# Max retries on transient API errors (429, 503, timeout).
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
# Input token soft limit — log warning when exceeded, raise above hard limit.
LLM_INPUT_TOKEN_SOFT_LIMIT: int = int(os.getenv("LLM_INPUT_TOKEN_SOFT_LIMIT", "30000"))
LLM_INPUT_TOKEN_HARD_LIMIT: int = int(os.getenv("LLM_INPUT_TOKEN_HARD_LIMIT", "60000"))
# Per-call timeout in seconds (0 = no timeout).
LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

# --- Thinking token budgets for Gemini 2.5 Flash ---
# hybrid_course (factual): deterministic extraction, no thinking needed
THINKING_BUDGET_METADATA = 0
# recursive, semantic_general, and hybrid_course with advisory intent use thinking
THINKING_BUDGET_SEMANTIC = 1024

# --- Temperature constants for function-based stochasticity ---
# Deterministic (factual extraction): 0.0
TEMP_DETERMINISTIC = 0.0
# Synthesis (advisory reasoning): 0.2 for linguistic fluidity
TEMP_SYNTHESIS = 0.2

# --- Retrieval tuning (global fallbacks for low-confidence paths) ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# category_confidence threshold: if < this, inject Verbal Uncertainty Calibration.
CATEGORY_CONFIDENCE_THRESHOLD: float = 0.7

# Per-function retrieval config: retrieve_k = candidates sent to hybrid search per course,
# rerank_k = final results kept after cross-course reranking.
# For multi-course queries these are scaled by n_courses via compute_dynamic_k.
FUNCTION_RETRIEVAL_CONFIG: dict[str, dict[str, int]] = {
    # Per-course filtered hybrid search (vector + BM25), then cross-course rerank
    # v4 chunks are ~3x smaller (median 100 tokens) → 2x scaling from v3 values
    "hybrid_course": {"retrieve_k": 40, "rerank_k": 15},
    # Corpus-wide vector search — not scaled by course count
    "semantic_general": {"retrieve_k": 50, "rerank_k": 20},
    # Two-stage: anchor fetch → corpus-wide discovery
    "recursive": {"retrieve_k": 30, "rerank_k": 10},
    # No retrieval
    "out_of_scope": {"retrieve_k": 0, "rerank_k": 0},
}

# Alias used by router.compute_dynamic_k for per-course scaling.
PER_COURSE_K = FUNCTION_RETRIEVAL_CONFIG

# Global caps for scaled multi-course retrieval.
MAX_RETRIEVE_K: int = 100
MAX_RERANK_K: int = 35

# Maximum unique discovery courses to recommend in recursive path (after schedule filter).
RECURSIVE_MAX_RECOMMENDED_COURSES: int = 3

# Stratified selection: chunks per (course_id, header_path) slot after reranking.
CHUNKS_PER_SLOT: int = 2
# Fallback when no specific header_path given: top-N per unique course_id.
STRATIFIED_FALLBACK_PER_COURSE: int = 12

# --- Reranker score threshold ---
# Drops chunks below a fixed score after reranking. Always active.
# Lowered from 0.35 for v4's smaller chunks which get lower absolute reranker scores.
RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.25"))
RERANK_SCORE_MIN_CHUNKS: int = int(os.getenv("RERANK_SCORE_MIN_CHUNKS", "2"))

# --- Reranker knee-point filter ---
# When enabled, cuts low-signal chunks using a score-gap heuristic (off by default).
RERANK_KNEE_ENABLED: bool = os.getenv("RERANK_KNEE_ENABLED", "false").lower() == "true"
RERANK_KNEE_ABS_THRESHOLD: float = float(os.getenv("RERANK_KNEE_ABS_THRESHOLD", "0.15"))
RERANK_KNEE_MIN_CHUNKS: int = int(os.getenv("RERANK_KNEE_MIN_CHUNKS", "2"))
RERANK_KNEE_MIN_GAP_FALLBACK: float = float(os.getenv("RERANK_KNEE_MIN_GAP_FALLBACK", "0.05"))

# --- Retrieval backend ---
# "mongodb" (default) or "vertex" (legacy fallback)
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "mongodb")

# --- Course-summary primer ---
# When True, hybrid_course also fetches the per-course summary as a non-citable
# <overview> primer prepended to the context XML. The primer is NOT counted as
# a retrieved chunk (invisible to Ragas precision/recall).
SUMMARY_AS_PRIMER: bool = os.getenv("SUMMARY_AS_PRIMER", "true").lower() == "true"

# --- v6 pipeline feature flags ---
# Master switch for v6 ingestion pipeline (additive + vector-tagged boilerplate).
USE_V6_PIPELINE: bool = os.getenv("USE_V6_PIPELINE", "false").lower() == "true"
# When True, retrieval includes chunks tagged is_boilerplate=True (TAMU-policy
# queries). When False (default), boilerplate is excluded at query time.
# Back-compat: pre-v6 chunks without the field are always kept (handled by $ne: True).
INCLUDE_BOILERPLATE: bool = os.getenv("INCLUDE_BOILERPLATE", "false").lower() == "true"
INCLUDE_DUPLICATE: bool = os.getenv("INCLUDE_DUPLICATE", "false").lower() == "true"
# Cosine-sim threshold for the v6 tag stage; calibrated in Step 3 (target precision >=0.98).
V6_TAG_THRESHOLD: float = float(os.getenv("V6_TAG_THRESHOLD", "0.92"))

# --- v6b pipeline feature flags (RAG-Anything-based parallel track) ---
# Master switch. False by default; turn on to materialize pipeline_v6b assets.
USE_V6B_PIPELINE: bool = os.getenv("USE_V6B_PIPELINE", "false").lower() == "true"
# Threshold for "table-heavy" routing decision (Phase 2). table_density = (tables_detected_count
# * 2 + total_table_cells / 100) / page_count; route to MinerU when >= threshold.
V6B_TABLE_DENSITY_THRESHOLD: float = float(os.getenv("V6B_TABLE_DENSITY_THRESHOLD", "3.0"))
# Hard cap on vision-LLM calls per silver_modal materialization. Default 10 matches the
# project rule against unannounced API spend; bump per-run via env var when scaling.
V6B_MODAL_CALL_BUDGET: int = int(os.getenv("V6B_MODAL_CALL_BUDGET", "10"))
# When False (default in Phase 1), silver_modal asset is a no-op: blocks pass through
# unmodified. Flip to True once budget is approved per pilot run.
V6B_MODAL_ENABLED: bool = os.getenv("V6B_MODAL_ENABLED", "false").lower() == "true"
# When False (default), silver_ingest is a dry-run: chunks are embedded and serialized to
# disk but never written to Atlas. Flip to True to actually upsert chunks_v4.
V6B_INGEST_ENABLED: bool = os.getenv("V6B_INGEST_ENABLED", "false").lower() == "true"

# --- NuExtract structured-extraction backend ---
# "in_process" (transformers+fla on the local GPU, default) or "http" (vLLM
# sidecar, OpenAI-compatible). The in-process path stays the default + fallback.
NUEXTRACT_BACKEND = os.getenv("NUEXTRACT_BACKEND", "in_process").strip().lower()
NUEXTRACT_SERVER_URL = os.getenv("NUEXTRACT_SERVER_URL", "http://localhost:8000/v1")
NUEXTRACT_MODEL = os.getenv("NUEXTRACT_MODEL", "numind/NuExtract3")

# --- Observability ---
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
APP_MODE = os.getenv("APP_MODE", "test")

# --- Langfuse ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


# --- Google API rate limiter ---
class _GoogleRateLimiter:
    """Sliding-window rate limiter: enforces at most `rpm` calls per 60 seconds."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._window: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a call slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._window = [t for t in self._window if t >= cutoff]
                if len(self._window) < self._rpm:
                    self._window.append(now)
                    return
                wait = self._window[0] + 60.0 - now
            time.sleep(max(wait, 0.1))


_google_rate_limiter = _GoogleRateLimiter(GOOGLE_API_RPM)

# --- Shared google-genai client (lazy singleton) ---
_genai_client = None


def get_genai_client():
    """Return a shared google.genai.Client instance, creating it on first call.

    Each call acquires a rate-limit slot (GOOGLE_API_RPM calls/minute).
    """
    _google_rate_limiter.acquire()
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _genai_client


# --- Shared TAMU OpenAI-compatible client (lazy singleton) ---
_tamu_client = None


def get_tamu_client():
    """Return a shared openai.OpenAI client pointed at the TAMU AI gateway."""
    global _tamu_client
    if _tamu_client is None:
        from openai import OpenAI

        _tamu_client = OpenAI(api_key=TAMU_API_KEY, base_url=TAMU_BASE_URL)
    return _tamu_client


# ---------------------------------------------------------------------------
# v4 pipeline feature flags
# ---------------------------------------------------------------------------
USE_V4_PIPELINE: bool = os.getenv("USE_V4_PIPELINE", "true").lower() == "true"
V4_CHECKPOINTER_BACKEND: str = os.getenv("V4_CHECKPOINTER_BACKEND", "memory")
V4_MAX_HISTORY_TURNS: int = int(os.getenv("V4_MAX_HISTORY_TURNS", "6"))

# --- mem0 integration ---
MEM0_ENABLED: bool = os.getenv("MEM0_ENABLED", "false").lower() == "true"
MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")
SESSION_CACHE_ENABLED: bool = os.getenv("SESSION_CACHE_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Public-deployment guards (Railway)
# ---------------------------------------------------------------------------
# Master switch. False locally / in tests so no Lakera key or Atlas counter is
# needed in dev. Set GUARD_ENABLED=true in the Railway service env.
GUARD_ENABLED: bool = os.getenv("GUARD_ENABLED", "false").lower() == "true"

# Cost guard — per-browser-session turn cap + global daily turn budget.
SESSION_TURN_CAP: int = int(os.getenv("SESSION_TURN_CAP", "20"))
DAILY_TURN_BUDGET: int = int(os.getenv("DAILY_TURN_BUDGET", "500"))
# Collection holding one document per UTC date: {"_id": "YYYY-MM-DD", "turns": int}
USAGE_COLLECTION: str = os.getenv("USAGE_COLLECTION", "usage")

# Safety guard — Lakera Guard prompt-injection / jailbreak detection.
LAKERA_GUARD_API_KEY = os.getenv("LAKERA_GUARD_API_KEY")
LAKERA_BASE_URL = os.getenv("LAKERA_BASE_URL", "https://api.lakera.ai")
LAKERA_TIMEOUT_SECONDS: float = float(os.getenv("LAKERA_TIMEOUT_SECONDS", "5.0"))
