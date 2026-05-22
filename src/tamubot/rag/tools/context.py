"""Context assembly for the generator stage.

Provides format_context_xml() which converts reranked retrieval results into
an XML-tagged context string with primacy-recency bracketing to combat
Lost-in-the-Middle attention degradation.
"""

import html
import re
from typing import Optional


def format_context_xml(results: list[dict], primer: Optional[str] = None) -> str:
    """Format retrieval results as XML-tagged chunks for the generator.

    Implements primacy-recency bracketing to combat Lost-in-the-Middle attention degradation:
    - Rank 1 chunk → Context start (primacy position)
    - Rank 2 chunk → Context end (recency/nearest query position)
    - Ranks 3–N → Middle (descending rank order)

    Each chunk gets metadata attributes so the LLM can cite sources precisely.

    If `primer` is non-empty, it is prepended verbatim inside an `<overview>`
    block with no `source=` attribute and no `<chunk>` wrapper, so the generator
    cannot construct a `[Source N]` citation for it.
    """
    parts: list[str] = []
    if primer:
        parts.append("<overview>")
        parts.append(primer)
        parts.append("</overview>")

    if not results:
        parts.append("<context>\nNo relevant documents found.\n</context>")
        return "\n".join(parts)

    # Apply primacy-recency reordering: [rank_1, ranks_3_to_N, rank_2]
    # No reorder needed for 1-2 results; for 3+, rank 1 at start, rank 2 at end.
    if len(results) <= 2:
        ordered_results = results
        rank_mapping = list(range(1, len(results) + 1))
    else:
        ordered_results = [results[0]] + results[2:] + [results[1]]
        rank_mapping = [1] + list(range(3, len(results) + 1)) + [2]

    parts.append("<context>")
    for position, (rank, doc) in enumerate(zip(rank_mapping, ordered_results), 1):
        # source= attribute uses original rank for citation purposes
        attrs = [f'source="{rank}"', f'id="{rank}"']
        if doc.get("course_id"):
            attrs.append(f'course="{doc["course_id"]}"')
        if doc.get("section"):
            attrs.append(f'section="{doc["section"]}"')
        if doc.get("header_path"):
            attrs.append(f'header="{doc["header_path"]}"')
        if doc.get("instructor_name"):
            attrs.append(f'instructor="{doc["instructor_name"]}"')
        if doc.get("term"):
            attrs.append(f'term="{doc["term"]}"')
        if doc.get("page") is not None:
            attrs.append(f'page="{doc["page"]}"')
        if doc.get("source"):
            attrs.append(f'origin="{html.escape(str(doc["source"]))}"')

        attr_str = " ".join(attrs)
        title = doc.get("title", "") or doc.get("header_text", "") or doc.get("header_path", "")
        content = doc.get("content", doc.get("policy_name", ""))

        # XML escape special characters in content
        content_escaped = html.escape(content)
        title_escaped = html.escape(title) if title else ""

        parts.append(f"<chunk {attr_str}>")
        if title_escaped:
            parts.append(f"<title>{title_escaped}</title>")
        parts.append(f"<content>{content_escaped}</content>")
        parts.append("</chunk>")
    parts.append("</context>")
    return "\n".join(parts)


def collapse_whitespace(text: str) -> str:
    """Collapse 3+ consecutive spaces to a single space.

    Gemini sometimes pads markdown table cells with excessive whitespace.
    """
    return re.sub(r" {3,}", " ", text)


def strip_thinking_blocks(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from generated text.

    The system prompt instructs the model to write a Chain-of-Verification
    quote into a <thinking> block before answering. These blocks must be
    stripped before the response is shown to the user.
    """
    return re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).strip()
