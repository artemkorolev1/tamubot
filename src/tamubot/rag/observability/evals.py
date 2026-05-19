"""Eval block system — declarative, registry-based RAGAS evaluation with retry + failure scoring."""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import ObservabilityConfig

logger = logging.getLogger("tamubot.observability")

# ---------------------------------------------------------------------------
# Critic LLM factory (singletons)
# ---------------------------------------------------------------------------

_critic_llm = None
_critic_embeddings = None


def get_critic_llm():
    """Singleton TAMU gateway ChatOpenAI wrapped for RAGAS."""
    global _critic_llm
    if _critic_llm is None:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        from tamubot.core import config

        _critic_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=config.TAMU_MODEL,
                api_key=config.TAMU_API_KEY,
                base_url=config.TAMU_BASE_URL,
                temperature=0,
            )
        )
    return _critic_llm


def get_critic_embeddings():
    """Singleton Voyage AI embeddings wrapped for RAGAS."""
    global _critic_embeddings
    if _critic_embeddings is None:
        import voyageai
        from langchain_core.embeddings import Embeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        from tamubot.core import config

        class _VoyageEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
                return client.embed(texts, model="voyage-3", input_type="document").embeddings

            def embed_query(self, text: str) -> list[float]:
                client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
                return client.embed([text], model="voyage-3", input_type="query").embeddings[0]

        _critic_embeddings = LangchainEmbeddingsWrapper(_VoyageEmbeddings())
    return _critic_embeddings


# ---------------------------------------------------------------------------
# EvalInputs + EvalBlock base
# ---------------------------------------------------------------------------


@dataclass
class EvalInputs:
    """Data container for eval block inputs."""

    question: str
    contexts: list[str]
    answer: str = ""
    reference: str = ""
    trace_id: Optional[str] = None


class EvalBlock(ABC):
    """Base class for declarative eval blocks."""

    name: str
    required_fields: tuple[str, ...]

    @abstractmethod
    def compute(self, inputs: EvalInputs) -> dict[str, float]:
        """Run evaluation and return metric_name → score."""

    def score_failure(self, inputs: EvalInputs, error: Exception) -> dict[str, float]:
        """Return failure scores (default: -1 for each metric)."""
        return {self.name: -1.0}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, EvalBlock] = {}


def register_block(block: EvalBlock) -> None:
    """Register an eval block by name."""
    _REGISTRY[block.name] = block


def _ensure_registry_loaded() -> None:
    """Import ragas_blocks to auto-register blocks on first use."""
    if not _REGISTRY:
        import tamubot.rag.observability.ragas_blocks  # noqa: F401


def available_metrics() -> list[str]:
    """Return all registered eval block names."""
    _ensure_registry_loaded()
    return sorted(_REGISTRY.keys())


def resolve_metrics(selection: Optional[str], default: list[str]) -> list[str]:
    """Resolve a user --metrics selection string against the registry.

    Accepts: None (returns default), "all" (returns every registered block),
    or a comma-separated list. Unknown names raise ValueError listing what's available.
    """
    _ensure_registry_loaded()
    if selection is None:
        return list(default)
    sel = selection.strip()
    if sel.lower() == "all":
        return available_metrics()
    requested = [s.strip() for s in sel.split(",") if s.strip()]
    unknown = [m for m in requested if m not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available: {available_metrics()}")
    return requested


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _disable_litellm_budget() -> None:
    """Disable litellm's $5 budget cap that trips on TAMU gateway pricing."""
    try:
        import litellm

        litellm.max_budget = None
    except Exception:
        pass


def _run_evals_sync(
    obs_config: ObservabilityConfig,
    inputs: EvalInputs,
) -> dict[str, float]:
    """Run all configured eval blocks synchronously. Returns merged scores dict."""
    from .tracing import get_langfuse

    _ensure_registry_loaded()
    _disable_litellm_budget()

    all_scores: dict[str, float] = {}

    for block_name in obs_config.eval_blocks:
        block = _REGISTRY.get(block_name)
        if block is None:
            logger.warning(f"Eval block '{block_name}' not found in registry")
            continue

        # Check required fields
        for field_name in block.required_fields:
            if not getattr(inputs, field_name, None):
                logger.info(f"Skipping eval block '{block_name}': missing field '{field_name}'")
                break
        else:
            # All required fields present — run block
            scores = _run_one_block(block, inputs, obs_config.eval_retry)
            all_scores.update(scores)

            # Post scores to Langfuse
            lf = get_langfuse()
            if lf and inputs.trace_id:
                for metric_name, value in scores.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        try:
                            lf.create_score(
                                trace_id=inputs.trace_id,
                                name=metric_name,
                                value=float(value),
                                comment="RAGAS automated evaluation" if value >= 0 else "RAGAS evaluation failed",
                            )
                        except Exception as e:
                            logger.warning(f"Failed to post score '{metric_name}': {e}")

    return all_scores


def _run_one_block(
    block: EvalBlock,
    inputs: EvalInputs,
    retry: bool,
) -> dict[str, float]:
    """Run a single eval block with optional retry."""
    try:
        return block.compute(inputs)
    except Exception as first_error:
        if retry:
            try:
                return block.compute(inputs)
            except Exception as second_error:
                logger.warning(f"Eval block '{block.name}' failed after retry: {second_error}")
                return block.score_failure(inputs, second_error)
        else:
            logger.warning(f"Eval block '{block.name}' failed: {first_error}")
            return block.score_failure(inputs, first_error)


def run_evals(
    obs_config: ObservabilityConfig,
    inputs: EvalInputs,
) -> dict[str, float]:
    """Run configured eval blocks. Async if obs_config.eval_async, else synchronous.

    Returns scores dict (empty if async — scores posted to Langfuse in background).
    """
    if not obs_config.eval_blocks:
        return {}

    if obs_config.eval_async:
        thread = threading.Thread(
            target=_run_evals_sync,
            args=(obs_config, inputs),
            daemon=True,
        )
        thread.start()
        return {}  # scores posted in background
    else:
        return _run_evals_sync(obs_config, inputs)
