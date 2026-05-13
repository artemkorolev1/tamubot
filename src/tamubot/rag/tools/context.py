"""Context assembly for the generator stage.

Provides format_context_xml() which converts reranked retrieval results into
an XML-tagged context string with primacy-recency bracketing to combat
Lost-in-the-Middle attention degradation.
"""

import html
import re


def format_context_xml(results: list[dict]) -> str:
    """Format retrieval results as XML-tagged chunks for the generator.

    Implements primacy-recency bracketing to combat Lost-in-the-Middle attention degradation:
    - Rank 1 chunk → Context start (primacy position)
    - Rank 2 chunk → Context end (recency/nearest query position)
    - Ranks 3–N → Middle (descending rank order)

    Each chunk gets metadata attributes so the LLM can cite sources precisely.
    """
    if not results:
        return "<context>\nNo relevant documents found.\n</context>"

    # Apply primacy-recency reordering: [rank_1, ranks_3_to_N, rank_2]
    if len(results) == 1:
        ordered_results = results
        rank_mapping = [1]
    elif len(results) == 2:
        ordered_results = [results[0], results[1]]
        rank_mapping = [1, 2]
    else:
        # Rank 1 at start, Rank 2 at end, Ranks 3-N in middle (descending order)
        ordered_results = [results[0]] + results[2:] + [results[1]]
        rank_mapping = [1] + list(range(3, len(results) + 1)) + [2]

    # Prepend course summaries if available — skip when chunks ARE summaries (course_summary path)
    from tamubot.rag.tools.mongo import get_course_summaries

    all_summary_chunks = all(doc.get("source") == "course_summary" for doc in results)
    course_ids = list({doc.get("course_id") for doc in results if doc.get("course_id")})
    summaries = get_course_summaries(course_ids) if course_ids and not all_summary_chunks else {}

    parts = []
    if summaries:
        parts.append("<course_summaries>")
        for cid, summary in sorted(summaries.items()):
            parts.append(f'<summary course="{html.escape(cid)}">{html.escape(summary)}</summary>')
        parts.append("</course_summaries>")

    parts.append("<context>")
    for position, (rank, doc) in enumerate(zip(rank_mapping, ordered_results), 1):
        # source= attribute uses original rank for citation purposes
        attrs = [f'source="{rank}"', f'id="{rank}"']
        if doc.get("course_id"):
            attrs.append(f'course="{doc["course_id"]}"')
        if doc.get("section"):
            attrs.append(f'section="{doc["section"]}"')
        # Prefer v4 header_path, fall back to v3 category
        if doc.get("header_path"):
            attrs.append(f'header="{doc["header_path"]}"')
        elif doc.get("category"):
            attrs.append(f'category="{doc["category"]}"')
        if doc.get("instructor_name"):
            attrs.append(f'instructor="{doc["instructor_name"]}"')
        if doc.get("term"):
            attrs.append(f'term="{doc["term"]}"')

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
