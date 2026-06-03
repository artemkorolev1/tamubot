from __future__ import annotations

import re
from dataclasses import dataclass, field


def _tokens_approx(text: str) -> int:
    return max(0, round(len(text) / 4))


def _has_table(content: str) -> bool:
    return bool(re.search(r"\|.*\|", content))


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts


_SENTENCE_RE = re.compile(r"(?<=\. )(?=[A-Z])")


def _split_on_whitespace(text: str, max_chars: int) -> list[str]:
    """Hard split on whitespace boundaries — guaranteed to produce pieces ≤ max_chars."""
    result = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        result.append(text[:cut].strip())
        text = text[cut:].strip()
    if text.strip():
        result.append(text.strip())
    return result


def _split_long_text(text: str, max_tok: int) -> list[str]:
    """Split a single oversized text block into pieces under max_tok."""
    max_chars = max_tok * 4

    # Try line boundaries first
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) > 1:
        packed = _pack_parts(lines, max_chars)
        # Re-split any individual packed chunk still over max_chars
        # (e.g. a single line that is itself huge)
        result = []
        for chunk in packed:
            if len(chunk) > max_chars:
                result.extend(_split_on_whitespace(chunk, max_chars))
            else:
                result.append(chunk)
        return result

    # Try sentence boundaries
    sentences = _SENTENCE_RE.split(text)
    if len(sentences) > 1:
        return _pack_parts(sentences, max_chars)

    # Last resort: split on whitespace near max_chars boundary
    return _split_on_whitespace(text, max_chars)


def _pack_parts(parts: list[str], max_chars: int) -> list[str]:
    """Greedily pack parts into blocks under max_chars."""
    result: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for part in parts:
        if buf_len + len(part) > max_chars and buf:
            result.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(part)
        buf_len += len(part)
    if buf:
        result.append("\n".join(buf))
    return result


# ── Section tree ─────────────────────────────────────────────────────────────


@dataclass
class SectionNode:
    header: str = ""
    level: int = 0
    body: str = ""
    children: list[SectionNode] = field(default_factory=list)

    def total_tokens(self) -> int:
        t = _tokens_approx(self._full_text())
        return t

    def _full_text(self) -> str:
        parts = []
        if self.header:
            parts.append(f"{'#' * self.level} {self.header}")
        if self.body:
            parts.append(self.body)
        for child in self.children:
            parts.append(child._full_text())
        return "\n\n".join(parts)


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")


def _parse_sections(markdown: str) -> SectionNode:
    root = SectionNode(header="", level=0, body="", children=[])
    stack: list[SectionNode] = [root]
    body_lines: list[str] = []

    def _flush_body():
        text = "\n".join(body_lines).strip()
        if text:
            stack[-1].body = text
        body_lines.clear()

    for line in markdown.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            _flush_body()
            level = len(m.group(1))
            node = SectionNode(header=m.group(2).strip(), level=level)
            # Find parent: pop stack until we find a node with level < this one
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        else:
            body_lines.append(line)

    _flush_body()
    return root


# ── Chunking ─────────────────────────────────────────────────────────────────


def _header_path(ancestors: list[str]) -> str:
    return " > ".join(a for a in ancestors if a)


def _emit_chunk(
    content: str,
    header_path: str,
    split_reason: str,
    chunks: list[dict],
):
    content = content.strip()
    if not content:
        return
    chunks.append(
        {
            "chunk_index": len(chunks),
            "content": content,
            "header_path": header_path,
            "token_count": _tokens_approx(content),
            "has_table": _has_table(content),
            "split_reason": split_reason,
        }
    )


def _node_text(node: SectionNode, include_children: bool = True) -> str:
    parts = []
    if node.header:
        parts.append(f"{'#' * node.level} {node.header}")
    if node.body:
        parts.append(node.body)
    if include_children:
        for child in node.children:
            parts.append(_node_text(child))
    return "\n\n".join(parts)


def _walk(
    node: SectionNode,
    ancestors: list[str],
    max_tok: int,
    min_tok: int,
    chunks: list[dict],
    log: dict,
):
    total = node.total_tokens()
    hp = _header_path(ancestors)

    if total <= max_tok:
        # Emit as one chunk
        text = _node_text(node)
        _emit_chunk(text, hp, "section", chunks)
        log["kept_as_is"] += 1
        return

    if node.children:
        # Emit this node's own body first (if any)
        if node.body.strip():
            body_text = ""
            if node.header:
                body_text = f"{'#' * node.level} {node.header}\n\n{node.body}"
            else:
                body_text = node.body
            if _tokens_approx(body_text) > max_tok:
                _paragraph_fallback(body_text, hp, max_tok, chunks, log)
            elif body_text.strip():
                _emit_chunk(body_text, hp, "sub-header", chunks)

        # Recurse into children
        for child in node.children:
            _walk(child, ancestors + [child.header], max_tok, min_tok, chunks, log)
        log["split"] += 1
        return

    # Leaf node too big — paragraph fallback
    text = _node_text(node)
    _paragraph_fallback(text, hp, max_tok, chunks, log)


def _paragraph_fallback(
    text: str,
    header_path: str,
    max_tok: int,
    chunks: list[dict],
    log: dict,
):
    paragraphs = _split_paragraphs(text)
    current_parts: list[str] = []
    current_tok = 0

    for para in paragraphs:
        ptok = _tokens_approx(para)
        if ptok > max_tok:
            # Flush buffer first
            if current_parts:
                _emit_chunk("\n\n".join(current_parts), header_path, "paragraph-fallback", chunks)
                current_parts = []
                current_tok = 0
            # Split the oversized paragraph
            for sub in _split_long_text(para, max_tok):
                _emit_chunk(sub, header_path, "long-paragraph-split", chunks)
            log["long_paragraph_split"] = log.get("long_paragraph_split", 0) + 1
            continue
        if current_tok + ptok > max_tok and current_parts:
            _emit_chunk("\n\n".join(current_parts), header_path, "paragraph-fallback", chunks)
            current_parts = []
            current_tok = 0
        current_parts.append(para)
        current_tok += ptok

    if current_parts:
        _emit_chunk("\n\n".join(current_parts), header_path, "paragraph-fallback", chunks)

    log["paragraph_fallback"] += 1


def _merge_small(chunks: list[dict], min_tok: int, max_tok: int = 600) -> tuple[list[dict], int]:
    if not chunks:
        return chunks, 0
    merged: list[dict] = []
    merge_count = 0

    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        combined_tok = merged[-1]["token_count"] + chunk["token_count"]
        can_merge = combined_tok <= max_tok
        if can_merge and chunk["token_count"] < min_tok and merged[-1]["header_path"] == chunk["header_path"]:
            merged[-1]["content"] += "\n\n" + chunk["content"]
            merged[-1]["token_count"] = _tokens_approx(merged[-1]["content"])
            merged[-1]["has_table"] = merged[-1]["has_table"] or chunk["has_table"]
            merged[-1]["split_reason"] = "merged"
            merge_count += 1
        elif can_merge and merged[-1]["token_count"] < min_tok:
            merged[-1]["content"] += "\n\n" + chunk["content"]
            merged[-1]["token_count"] = _tokens_approx(merged[-1]["content"])
            merged[-1]["has_table"] = merged[-1]["has_table"] or chunk["has_table"]
            merged[-1]["split_reason"] = "merged"
            merge_count += 1
        else:
            merged.append(chunk)

    # Re-index
    for i, c in enumerate(merged):
        c["chunk_index"] = i

    return merged, merge_count


def chunk_markdown(
    markdown: str,
    max_chunk_tokens: int = 600,
    min_chunk_tokens: int = 50,
    split_level: int = 3,
) -> list[dict]:
    root = _parse_sections(markdown)
    log: dict = {"kept_as_is": 0, "split": 0, "paragraph_fallback": 0}
    chunks: list[dict] = []

    if not root.children and root.body.strip():
        # No headers at all — use paragraph fallback on entire document
        _paragraph_fallback(root.body, "", max_chunk_tokens, chunks, log)
    else:
        # Emit root body if any
        if root.body.strip():
            if _tokens_approx(root.body) > max_chunk_tokens:
                _paragraph_fallback(root.body, "", max_chunk_tokens, chunks, log)
            else:
                _emit_chunk(root.body, "", "section", chunks)

        for child in root.children:
            _walk(child, [child.header], max_chunk_tokens, min_chunk_tokens, chunks, log)

    chunks, merge_count = _merge_small(chunks, min_chunk_tokens, max_chunk_tokens)
    return chunks


def chunk_with_log(markdown: str, **kwargs) -> tuple[list[dict], dict]:
    root = _parse_sections(markdown)
    max_tok = kwargs.get("max_chunk_tokens", 600)
    min_tok = kwargs.get("min_chunk_tokens", 50)

    log = {"kept_as_is": 0, "split": 0, "paragraph_fallback": 0, "long_paragraph_split": 0, "merged": 0}
    chunks: list[dict] = []

    total_sections = _count_sections(root)

    if not root.children and root.body.strip():
        _paragraph_fallback(root.body, "", max_tok, chunks, log)
    else:
        if root.body.strip():
            if _tokens_approx(root.body) > max_tok:
                _paragraph_fallback(root.body, "", max_tok, chunks, log)
            else:
                _emit_chunk(root.body, "", "section", chunks)

        for child in root.children:
            _walk(child, [child.header], max_tok, min_tok, chunks, log)

    chunks, merge_count = _merge_small(chunks, min_tok, max_tok)
    log["merged"] = merge_count
    log["total_sections"] = total_sections

    return chunks, log


def _count_sections(node: SectionNode) -> int:
    count = 1 if node.header else 0
    for child in node.children:
        count += _count_sections(child)
    return count


# ── Semantic chunking ───────────────────────────────────────────────────────

# Headers whose content is already captured in course_metadata — drop as chunks
_METADATA_HEADERS_LOWER = {
    "course information",
    "instructor details",
    "instructor information",
    "course details",
    "course prerequisites",
    "preferred contact method",
    "meeting times:",
    "socials",
}

_TITLE_HEADER_RE = re.compile(r"^[A-Z]{3,4}\s+\d{3}\s+(Syllabus|Course\s+Syllabus)$", re.IGNORECASE)
_SYLLABUS_TITLE_RE = re.compile(r"^Course\s+Syllabus:\s+[A-Z]{3,4}\s+\d{3}\b", re.IGNORECASE)

# Strip leading section numbers like "1.1 ", "2.3.1 " before matching
_LEADING_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\s+")


def _is_metadata_section(header: str) -> bool:
    """Check if a section header corresponds to metadata already extracted."""
    h = header.strip()
    # Strip leading section numbers (e.g. "1.1 Course Information" → "Course Information")
    h_norm = _LEADING_SECTION_NUM_RE.sub("", h)
    if h_norm.lower() in _METADATA_HEADERS_LOWER:
        return True
    if _TITLE_HEADER_RE.match(h) or _TITLE_HEADER_RE.match(h_norm):
        return True
    if _SYLLABUS_TITLE_RE.match(h) or _SYLLABUS_TITLE_RE.match(h_norm):
        return True
    return False


def chunk_semantic(
    markdown: str,
    flag_threshold: int = 500,
    min_chunk_tokens: int = 15,
) -> tuple[list[dict], dict]:
    """Semantic chunking: one chunk per top-level section, no max size cap.

    - Sections matching metadata headers are dropped (already in course_metadata).
    - Chunks under *min_chunk_tokens* are merged into the previous chunk.
    - Chunks over *flag_threshold* are flagged but **not** split.
    - Subsections stay with their parent (semantic unity).
    """
    root = _parse_sections(markdown)
    chunks: list[dict] = []
    log: dict = {
        "total_sections": _count_sections(root),
        "kept": 0,
        "dropped_metadata": 0,
        "dropped_tiny": 0,
        "merged_tiny": 0,
        "flagged_oversized": 0,
    }

    def _refresh_flags(chunk: dict) -> None:
        """Recompute a chunk's flags after a merge grew it — otherwise OVERSIZED
        (and HAS_TABLE) can go stale once tiny chunks are glued on."""
        text = chunk["content"]
        flags: list[str] = []
        if chunk["token_count"] > flag_threshold:
            flags.append("OVERSIZED")
        if _has_table(text):
            flags.append("HAS_TABLE")
        if not chunk.get("header_path"):
            flags.append("NO_HEADER")
        chunk["flags"] = flags

    def _emit(text: str, hp: str, header: str) -> None:
        tok = _tokens_approx(text)
        if tok < min_chunk_tokens:
            # Try to merge into previous chunk instead of dropping
            if chunks:
                chunks[-1]["content"] += "\n\n" + text
                chunks[-1]["token_count"] = _tokens_approx(chunks[-1]["content"])
                chunks[-1]["has_table"] = chunks[-1]["has_table"] or _has_table(text)
                _refresh_flags(chunks[-1])
                log["merged_tiny"] += 1
            else:
                log["dropped_tiny"] += 1
            return
        flags: list[str] = []
        if tok > flag_threshold:
            flags.append("OVERSIZED")
            log["flagged_oversized"] += 1
        if _has_table(text):
            flags.append("HAS_TABLE")
        if not header:
            flags.append("NO_HEADER")
        chunks.append(
            {
                "chunk_index": len(chunks),
                "content": text,
                "header_path": hp,
                "token_count": tok,
                "has_table": _has_table(text),
                "flags": flags,
                "split_reason": "semantic",
            }
        )
        log["kept"] += 1

    def _process_node(node: SectionNode, ancestors: list[str]) -> None:
        hp = _header_path(ancestors)

        if node.header and _is_metadata_section(node.header):
            log["dropped_metadata"] += 1
            # Still recurse into children — they may be real content sections
            # nested under a metadata/title header
            for child in node.children:
                _process_node(child, ancestors + [child.header])
            return

        text = _node_text(node, include_children=True).strip()
        tok = _tokens_approx(text)

        if tok < min_chunk_tokens:
            # Merge into previous chunk instead of dropping
            if chunks:
                chunks[-1]["content"] += "\n\n" + text
                chunks[-1]["token_count"] = _tokens_approx(chunks[-1]["content"])
                chunks[-1]["has_table"] = chunks[-1]["has_table"] or _has_table(text)
                _refresh_flags(chunks[-1])
                log["merged_tiny"] += 1
            else:
                log["dropped_tiny"] += 1
            return

        # If oversized and has children, recurse into subsections
        if tok > flag_threshold and node.children:
            # Emit the node's own body (without children) if substantial
            own_text = _node_text(node, include_children=False).strip()
            if _tokens_approx(own_text) >= min_chunk_tokens:
                _emit(own_text, hp, node.header)
            elif _tokens_approx(own_text) > 0 and chunks:
                # Merge tiny parent body into previous chunk
                chunks[-1]["content"] += "\n\n" + own_text
                chunks[-1]["token_count"] = _tokens_approx(chunks[-1]["content"])
                chunks[-1]["has_table"] = chunks[-1]["has_table"] or _has_table(own_text)
                _refresh_flags(chunks[-1])
                log["merged_tiny"] += 1
            # Recurse into each child
            for child in node.children:
                _process_node(child, ancestors + [child.header])
            return

        # Oversized leaf section (no subsections to recurse into): split on
        # paragraph/sentence boundaries so a flat, merge-glued block (e.g. a
        # bold-styled "header" the bronze stage failed to promote, gluing
        # several policies together) doesn't emit one giant chunk. Childless
        # means there is no semantic substructure to preserve, so splitting
        # here doesn't violate the section-unity contract.
        if tok > flag_threshold and not node.children:
            for sub in _split_long_text(text, flag_threshold):
                _emit(sub.strip(), hp, node.header)
            return

        _emit(text, hp, node.header)
        return

    # Root body (text before any header)
    if root.body.strip():
        tok = _tokens_approx(root.body.strip())
        if tok >= min_chunk_tokens:
            flags = []
            if tok > flag_threshold:
                flags.append("OVERSIZED")
                log["flagged_oversized"] += 1
            if _has_table(root.body):
                flags.append("HAS_TABLE")
            # Root body precedes any header → headerless, same as _emit's contract.
            flags.append("NO_HEADER")
            chunks.append(
                {
                    "chunk_index": 0,
                    "content": root.body.strip(),
                    "header_path": "",
                    "token_count": tok,
                    "has_table": _has_table(root.body),
                    "flags": flags,
                    "split_reason": "semantic",
                }
            )
            log["kept"] += 1

    # Each top-level child → one semantic chunk
    for child in root.children:
        _process_node(child, [child.header])

    # Re-index
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks, log
