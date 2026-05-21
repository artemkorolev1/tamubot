"""rag — RAG pipeline public API.

Import only from this module, not from submodules.
"""

from tamubot.rag.graph.pipeline import get_current_state, run_pipeline, run_pipeline_with_memory
from tamubot.rag.models import ChunkDoc, CourseDoc, PolicyDoc
from tamubot.rag.observability import get_langfuse
from tamubot.rag.router import RouterResult
from tamubot.rag.state.pipeline_state import ConversationState, PipelineState

__all__ = [
    # Pipeline entry points
    "run_pipeline",
    "run_pipeline_with_memory",
    "get_current_state",
    # Data types
    "ChunkDoc",
    "CourseDoc",
    "PolicyDoc",
    "PipelineState",
    "ConversationState",
    "RouterResult",
    # Observability
    "get_langfuse",
]
