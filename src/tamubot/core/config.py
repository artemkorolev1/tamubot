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

# --- LLM guardrails ---
# Hard cap on output tokens (application-layer; TAMU gateway ignores max_tokens).
LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2000"))
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
    "hybrid_course": {"retrieve_k": 20, "rerank_k": 7},
    # Corpus-wide vector search — not scaled by course count
    "semantic_general": {"retrieve_k": 30, "rerank_k": 10},
    # Two-stage: anchor fetch → corpus-wide discovery
    "recursive": {"retrieve_k": 15, "rerank_k": 5},
    # No retrieval
    "out_of_scope": {"retrieve_k": 0, "rerank_k": 0},
}

# Alias used by router.compute_dynamic_k for per-course scaling.
PER_COURSE_K = FUNCTION_RETRIEVAL_CONFIG

# Global caps for scaled multi-course retrieval.
MAX_RETRIEVE_K: int = 60
MAX_RERANK_K: int = 20

# Maximum unique discovery courses to recommend in recursive path (after schedule filter).
RECURSIVE_MAX_RECOMMENDED_COURSES: int = 3

# Stratified selection: chunks per (course_id, category) slot after reranking.
CHUNKS_PER_SLOT: int = 2
# Fallback when no specific categories given: top-N per unique course_id.
STRATIFIED_FALLBACK_PER_COURSE: int = 6

# --- Reranker score threshold ---
# Drops chunks below a fixed score after reranking. Always active.
RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.35"))
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
