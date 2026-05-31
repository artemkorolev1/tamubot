"""Factory for LangGraph checkpointers.

Canonical location: rag/graph/checkpointer.py

"memory" → MemorySaver (default, no persistence across process restarts)
"sqlite" → SqliteSaver (persists across restarts, stored next to checkpointer.py)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tamubot.core import config


def make_checkpointer(backend: Optional[str] = None):
    """Create a LangGraph checkpointer based on config.V4_CHECKPOINTER_BACKEND.

    Args:
        backend: Override backend ("memory" | "sqlite"). Defaults to config.V4_CHECKPOINTER_BACKEND.

    Returns:
        A LangGraph checkpointer (MemorySaver or SqliteSaver)
    """
    backend = backend or config.V4_CHECKPOINTER_BACKEND

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = str(Path(__file__).parent / "sessions.db")
            return SqliteSaver(db_path)  # type: ignore[arg-type]
        except ImportError:
            # langgraph-checkpoint-sqlite not installed; fall back to memory
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
