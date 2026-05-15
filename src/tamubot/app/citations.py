"""Post-generation rewrite of [Source N, p.X] citations into clickable markdown links.

The generator emits citations as plain text in the form `[Source N]` or
`[Source N, p.X]`. The Streamlit app rewrites these into markdown links that
point at the corresponding syllabus PDF (served from .streamlit/static/syllabi/
via Streamlit's static-serving feature) with a #page=X deep-link fragment.

Resolution rules:
  - source_docs[N-1] supplies the PDF stem via source_file.
  - If source_file is missing, the citation is left as plain text [S N, p.X].
  - If the regex matches [Source N] with no page, link is generated without
    the #page= fragment.
"""

from __future__ import annotations

import re
from typing import Iterable

_CITATION_RE = re.compile(r"\[Source\s+(\d+)(?:,\s*p\.(\d+))?\]")

STATIC_PREFIX = "/app/static/syllabi"


def _resolve_pdf_stem(doc: dict) -> str | None:
    stem = doc.get("source_file")
    if not stem:
        return None
    # source_file is stored as either the .pdf path, the .json sidecar from
    # ingestion, or the bare stem. Strip either suffix.
    for suffix in (".pdf", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or None


def rewrite_citations(answer: str, source_docs: Iterable[dict]) -> str:
    """Rewrite [Source N, p.X] into clickable markdown links pointing at the PDF."""
    docs = list(source_docs)

    def replace(match: re.Match) -> str:
        n_str, page_str = match.group(1), match.group(2)
        n = int(n_str)
        if not (1 <= n <= len(docs)):
            return match.group(0)
        stem = _resolve_pdf_stem(docs[n - 1])
        page_suffix = f", p.{page_str}" if page_str else ""
        label = f"\\[S{n}{page_suffix}\\]"
        if stem is None:
            return label
        url = f"{STATIC_PREFIX}/{stem}.pdf"
        if page_str:
            url += f"#page={page_str}"
        return f"[{label}]({url})"

    return _CITATION_RE.sub(replace, answer)
