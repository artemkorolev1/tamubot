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
