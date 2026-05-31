"""Thread-pool batch parser driver.

Adapted from RAG-Anything raganything/batch_parser.py — tqdm progress bar
dropped (we log per-file via Dagster instead), upstream's coupling to
RAGAnything orchestrator removed. Just a simple parallel map over a Parser.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tamubot.vendor.raganything.parser import Parser

log = logging.getLogger(__name__)


def batch_parse(
    parser: Parser,
    pdf_paths: List[Path],
    output_dir: Optional[Path] = None,
    max_workers: int = 4,
    on_file_done: Optional[Callable[[Path, List[Dict[str, Any]], Optional[Exception]], None]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse many PDFs in parallel.

    Returns {pdf_path_str: block_list}. Files that error map to []; the
    exception is passed to on_file_done for logging. Callers that need to
    fail loud should check for [] values or use on_file_done.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    def _one(pdf_path: Path) -> tuple[Path, List[Dict[str, Any]], Optional[Exception]]:
        try:
            blocks = parser.parse_pdf(
                pdf_path,
                output_dir=str(output_dir) if output_dir else None,
            )
            return pdf_path, blocks, None
        except Exception as exc:
            log.warning("parse failed for %s: %s", pdf_path, exc)
            return pdf_path, [], exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, p) for p in pdf_paths]
        for fut in as_completed(futures):
            pdf_path, blocks, exc = fut.result()
            results[str(pdf_path)] = blocks
            if on_file_done is not None:
                on_file_done(pdf_path, blocks, exc)

    return results
