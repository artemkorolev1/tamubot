"""Text quality validators: replacement chars (U+FFFD) and letter-drops.

Letter-drop dictionary mirrors pipeline_v6c.run_bakeoff._LETTER_DROP_TOKENS.
"""

from __future__ import annotations

import re

from tamubot.ingestion.validation.types import CheckOutcome

REPLACEMENT_CHAR = "�"

_LETTER_DROP_TOKENS = (
    r"\bColege\b",
    r"\bSylabus\b",
    r"\bMeting\b",
    r"\bAditional\b",
    r"\bTextbok\b",
    r"\bwil\b",
    r"\bfal\b",
    r"\binstaling\b",
)
_LETTER_DROP_RE = re.compile("|".join(_LETTER_DROP_TOKENS))


def check_no_replacement_chars(text: str) -> CheckOutcome:
    """Pass iff text contains zero U+FFFD characters."""
    count = text.count(REPLACEMENT_CHAR)
    return CheckOutcome(
        passed=count == 0,
        metadata={"replacement_char_count": count},
    )


# OCR f-ligature loss (STAT_651 class): the PDF text layer drops the fi/fl/ff/ffi/ffl
# glyphs, so high-frequency university-policy words arrive mangled ("office"->"ofce",
# "confidentiality"->"confdentiality"). These damaged forms are vanishingly rare as
# legitimate words, so a curated allow-list of the common policy-vocabulary casualties
# is a low-false-positive proxy for "this doc's OCR damage is degrading dedup/BP
# matching". The util/text_normalize fold makes matching robust to this; this check is
# the visibility signal so a heavily-damaged stem gets a human glance. Case-insensitive.
_LIGATURE_DAMAGE_TOKENS = (
    r"ofce", r"ofcial", r"ofcer", r"ofcers",
    r"confdential", r"confdentiality", r"confdentially",
    r"signifcant", r"signifcantly",
    r"difcult", r"difculty", r"difculties",
    r"beneft", r"benefts",
    r"fnal", r"fnally",
    r"specifc", r"specifcally", r"specifed",
    r"certifcate", r"certifcation",
    r"notifcation", r"notifed", r"notifes",
    r"fexible", r"fexibility",
    r"fnancial",
    r"afliated", r"afliation",
    r"sufcient", r"sufciently",
    r"efcient", r"efciency",
    r"profcient", r"profciency",
    r"classifcation", r"classifed",
    r"modifcation", r"modifed",
    r"qualifcation", r"qualifed",
)
_LIGATURE_DAMAGE_RE = re.compile(r"\b(?:" + "|".join(_LIGATURE_DAMAGE_TOKENS) + r")\b", re.IGNORECASE)


def count_ligature_damage(text: str, threshold: int = 2) -> CheckOutcome:
    """Count OCR f-ligature-damaged policy words (STAT_651 class).

    A curated allow-list of common university-policy words in their ligature-dropped
    form ("ofce", "confdentiality", "signifcant", ...). Pass iff the count is
    ``<= threshold`` — a handful is tolerable (the signature fold handles matching),
    but a high count means the source PDF's text layer is badly damaged and any
    text-similarity (dedup, boilerplate, retrieval) on this stem is degraded. WARN.
    """
    matches = _LIGATURE_DAMAGE_RE.findall(text)
    return CheckOutcome(
        passed=len(matches) <= threshold,
        metadata={
            "ligature_damage_count": len(matches),
            "matches": matches[:20],
            "threshold": threshold,
        },
    )


def check_letter_drops(text: str, threshold: int = 0) -> CheckOutcome:
    """Pass iff letter-drop token count <= threshold (default 0)."""
    matches = _LETTER_DROP_RE.findall(text)
    return CheckOutcome(
        passed=len(matches) <= threshold,
        metadata={
            "letter_drop_count": len(matches),
            "matches": matches[:20],
            "threshold": threshold,
        },
    )


# A synthesized markdown link: [label](target) where target is a mailto: or http(s) URL.
# Catches the FID_HALLUCINATION class — bronze fabricated a clickable link for a value
# that was already present as plain text in some block.
_MD_LINK_RE = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<target>(?:mailto:|https?://)[^)\s]+)\)")


def _link_target_value(target: str) -> str:
    """The bare value a link target points at, stripped of the mailto:/scheme so it
    can be matched against plain text. ``mailto:jiang@tamu.edu`` -> ``jiang@tamu.edu``;
    ``https://cesg.tamu.edu/x`` -> ``cesg.tamu.edu/x`` (scheme + trailing slash removed)."""
    t = target.strip()
    if t.lower().startswith("mailto:"):
        return t[len("mailto:") :].strip()
    t = re.sub(r"^https?://", "", t, flags=re.IGNORECASE)
    return t.rstrip("/").strip()


def find_fabricated_links(blocks: list) -> CheckOutcome:
    """Detect synthesized markdown links whose target was *already present as plain
    text* somewhere in the document (the ``FID_HALLUCINATION`` signature — bronze
    fabricated a ``[label](mailto:X)`` / ``[label](http…)`` link for a value that
    already appeared verbatim, so the link adds nothing real and risks a wrong target).

    Pure over ``blocks: list[dict]``. For every text block, find markdown links to a
    ``mailto:`` / ``http(s)`` target; FLAG a link when its bare target value (scheme
    stripped) appears as a plain-text substring in *some* block's text that is NOT
    itself a markdown link to that same target. ``sample`` gives ``label -> target``
    examples. Shipped WARN — promote to blocking once the pass rate is confirmed
    across the golden set.
    """
    # Gather all link occurrences and the full plain-text corpus (links stripped, so a
    # link doesn't count as "plain text" evidence for itself).
    link_hits: list[tuple[str, str]] = []  # (label, bare_target)
    plain_parts: list[str] = []
    for b in blocks:
        t = b.get("text")
        if not isinstance(t, str) or not t:
            continue
        for m in _MD_LINK_RE.finditer(t):
            link_hits.append((m.group("label"), _link_target_value(m.group("target"))))
        # Plain text = this block with every markdown link removed, so a value that
        # only ever appears *inside* a link is not mistaken for pre-existing plain text.
        plain_parts.append(_MD_LINK_RE.sub(" ", t))
    plain_text = "\n".join(plain_parts)

    fabricated = 0
    samples: list[str] = []
    for label, value in link_hits:
        if value and value in plain_text:
            fabricated += 1
            if len(samples) < 20:
                samples.append(f"[{label}] -> {value}")
    return CheckOutcome(
        passed=fabricated == 0,
        metadata={
            "fabricated_link_count": fabricated,
            "total_links": len(link_hits),
            "sample_fabricated": samples,
        },
    )


def count_unanswered_labels(
    blocks: list,
    *,
    max_label_len: int = 35,
    max_label_words: int = 4,
) -> CheckOutcome:
    """Deterministic detector for the two-column reading-order split
    (``FID_HEADER_BROKEN``): a short ``Label:`` block whose value Docling orphaned
    far away, leaving the label with no value beside it. Pure — no PyMuPDF.

    A field label is a short text block (``<= max_label_len`` chars,
    ``<= max_label_words`` words, single line) ending in ``:``. It is *answered*
    when the next text block on the same page is a non-label value. A label
    followed by another bare label (the broken column-first run
    ``Course Number:`` / ``Course Title:`` / …) or by a heading is *unanswered*.
    After the adapter's two-column recovery a repaired label reads
    ``Course Number: ECEN 671`` — no trailing ``:`` — so it is not counted; a
    non-zero count means orphaning the recovery did not repair. Catches what
    ``text_coverage`` cannot (the values survive, just disconnected). Shipped WARN
    — calibrate on the corpus before promotion.
    """
    labels = 0
    unanswered = 0
    samples: list[str] = []
    for i, b in enumerate(blocks):
        if b.get("type") != "text":
            continue
        t = (b.get("text") or "").strip()
        if not t.endswith(":") or len(t) < 3 or len(t) > max_label_len:
            continue
        if "\n" in t or len(t.split()) > max_label_words:
            continue
        labels += 1
        page = b.get("page_idx") or 0
        nxt = None
        for ob in blocks[i + 1 :]:
            if (ob.get("page_idx") or 0) != page:
                break
            if ob.get("type") == "heading":
                break  # heading right after a label -> no value beside it
            if ob.get("type") == "text":
                nxt = (ob.get("text") or "").strip()
                break
        answered = bool(nxt) and not nxt.endswith(":")
        if not answered:
            unanswered += 1
            if len(samples) < 20:
                samples.append(t)
    return CheckOutcome(
        passed=unanswered == 0,
        metadata={
            "unanswered_labels": unanswered,
            "total_labels": labels,
            "sample_unanswered": samples,
        },
    )
