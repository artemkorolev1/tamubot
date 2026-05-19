"""Shared helpers for eval runners — golden-set filtering, env pre-arg, Langfuse dataset upsert, link lookup."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger("tamubot.evals.runner_common")


# ---------------------------------------------------------------------------
# Env / pre-arg parsing (must run before `from tamubot.rag.*` imports)
# ---------------------------------------------------------------------------


def pre_arg(flag: str) -> Optional[str]:
    """Read --flag VALUE from sys.argv without invoking argparse. Returns None if absent."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def set_collection_env(chunks_collection: Optional[str], chunk_tag: Optional[str]) -> None:
    """Set CHUNKS_COLLECTION / VECTOR_INDEX / TEXT_INDEX / CHUNK_TAG_FILTER env vars.

    rag/tools/mongo.py reads these at import time, so call this BEFORE any
    `from tamubot.rag.*` import in the runner.
    """
    if chunks_collection:
        suffix = chunks_collection.removeprefix("chunks_")
        os.environ.setdefault("CHUNKS_COLLECTION", chunks_collection)
        os.environ.setdefault("VECTOR_INDEX", f"vector_index_{suffix}")
        os.environ.setdefault("TEXT_INDEX", f"text_index_{suffix}")
    if chunk_tag:
        os.environ.setdefault("CHUNK_TAG_FILTER", chunk_tag)


# ---------------------------------------------------------------------------
# Golden-set filtering
# ---------------------------------------------------------------------------


def parse_id_filter(ids_arg: Optional[str]) -> Optional[set[int]]:
    """Parse --ids "3,7,12" into {3, 7, 12}. Returns None when arg is empty / absent."""
    if not ids_arg:
        return None
    out: set[int] = set()
    for tok in ids_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            raise ValueError(f"--ids value {tok!r} is not an integer")
    return out or None


def filter_items_by_ids(items: list[dict], ids: Optional[set[int]]) -> list[dict]:
    """Keep only items whose `id` field (cast to int) is in `ids`. Pass-through when ids is None."""
    if ids is None:
        return items
    kept = [it for it in items if _safe_int(it.get("id")) in ids]
    missing = ids - {_safe_int(it.get("id")) for it in items}
    if missing:
        logger.warning(f"--ids referenced row(s) not present in golden set: {sorted(missing)}")
    return kept


def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except TypeError, ValueError:
        return None


# ---------------------------------------------------------------------------
# Langfuse: dataset upsert + item id lookup
# ---------------------------------------------------------------------------


def stable_item_id(question: str) -> str:
    """16-char MD5 prefix — stable per question, used as fallback dataset_item_id."""
    return hashlib.md5(question.encode()).hexdigest()[:16]


def upsert_langfuse_dataset(
    lf,
    golden_items: list[dict],
    dataset_name: str,
    description: Optional[str] = None,
) -> None:
    """Create dataset (if absent) and upsert items keyed by question text."""
    try:
        lf.create_dataset(
            name=dataset_name,
            description=description
            or "TamuBot eval dataset — golden questions with router expectations and reference answers.",
        )
    except Exception:
        pass  # already exists

    existing = _list_dataset_question_set(lf, dataset_name)

    uploaded = 0
    for item in golden_items:
        question = item.get("question", item.get("query", ""))
        if not question or question in existing:
            continue
        try:
            lf.create_dataset_item(
                dataset_name=dataset_name,
                input=question,
                expected_output=item.get("reference_answer") or "",
                metadata={
                    "id": item.get("id"),
                    "expected_function": item.get("expected_function"),
                    "source_course_id": item.get("source_course_id"),
                    "source_category": item.get("source_category"),
                    "stratum": item.get("stratum"),
                },
            )
            uploaded += 1
        except Exception as e:
            logger.warning(f"Dataset item upsert failed for '{question[:40]}': {e}")

    print(f"  Langfuse dataset '{dataset_name}': {uploaded}/{len(golden_items)} new items.")


def load_dataset_item_ids(lf, dataset_name: str) -> dict[str, str]:
    """Return {question_text: langfuse_item_id} for a dataset."""
    out: dict[str, str] = {}
    try:
        page = 1
        while True:
            resp = lf.api.dataset_items.list(dataset_name=dataset_name, page=page, limit=100)
            for it in resp.data:
                q = it.input if isinstance(it.input, str) else ""
                if q:
                    out[q] = it.id
            if len(resp.data) < 100:
                break
            page += 1
    except Exception as e:
        logger.warning(f"Could not load dataset item IDs for '{dataset_name}': {e}")
    return out


def _list_dataset_question_set(lf, dataset_name: str) -> set[str]:
    return set(load_dataset_item_ids(lf, dataset_name).keys())


def tag_trace_superseded(lf, trace_id: str) -> None:
    """Best-effort: add `superseded` tag to a prior trace when its question is re-run."""
    if not trace_id:
        return
    try:
        existing = lf.api.trace.get(trace_id)
        tags = list(getattr(existing, "tags", []) or [])
        if "superseded" in tags:
            return
        tags.append("superseded")
        lf.api.trace.update(trace_id=trace_id, tags=tags)
    except Exception as e:
        logger.info(f"Could not tag trace {trace_id} superseded: {e}")


def find_existing_run_items(lf, dataset_name: str, run_name: str) -> dict[str, str]:
    """Return {dataset_item_id: trace_id} for an existing dataset run, or {} on miss."""
    out: dict[str, str] = {}
    try:
        run = lf.api.datasets.get_run(dataset_name=dataset_name, run_name=run_name)
        for item in getattr(run, "dataset_run_items", []) or []:
            item_id = getattr(item, "dataset_item_id", None)
            trace_id = getattr(item, "trace_id", None)
            if item_id and trace_id:
                out[item_id] = trace_id
    except Exception as e:
        logger.info(f"No prior dataset run '{run_name}' on '{dataset_name}': {e}")
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"
