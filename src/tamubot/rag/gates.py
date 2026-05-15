"""Post-generation validation gates.

Gate 1 (regex, synchronous): checks for [Source N] citation presence.
Gate 2 (LLM, asynchronous): RAGAS ResponseGroundedness scoring via observability.py.
"""

import re

# Functions that require citation validation
_FACTUAL_FUNCTIONS = frozenset(
    [
        "hybrid_course",
        "recursive",
        "semantic_general",
    ]
)


def validate_citations_gate1(response_text: str) -> bool:
    """Validate Gate 1: Check for presence of citations in response.

    Gate 1 (Regex): Verify presence of at least one [Source N] or [N] citation.
    Applied only to factual query types (metadata_*, hybrid_*).

    Args:
        response_text: Generated response text.

    Returns:
        True if at least one citation is found, False otherwise.
    """
    # Pattern matches [Source N], [Source N:], [Source N, p.X], or [N]
    citation_pattern = r"\[(?:Source\s+)?(\d+)[^\]]*\]"
    return re.search(citation_pattern, response_text) is not None


def validate_citations_with_trace(
    response_text: str,
    function: str,
) -> bool:
    """Validate citations and log result to current Langfuse observation.

    Args:
        response_text: Generated response text.
        function:      Retrieval function type.

    Returns:
        True if citations are present or validation is skipped.
    """
    if function not in _FACTUAL_FUNCTIONS:
        return True

    citation_valid = validate_citations_gate1(response_text)

    try:
        from langfuse import get_client as _lf_get_client

        _lf_get_client().score_current_trace(
            name="citation_gate1_pass",
            value=1 if citation_valid else 0,
            comment="Gate 1 validation: regex citation check",
        )
    except Exception:
        pass

    return citation_valid
