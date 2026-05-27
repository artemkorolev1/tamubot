"""Test the CheckOutcome dataclass shared by all validation helpers."""

from tamubot.ingestion.validation.types import CheckOutcome


def test_check_outcome_pass():
    outcome = CheckOutcome(passed=True, metadata={"count": 42})
    assert outcome.passed is True
    assert outcome.metadata == {"count": 42}


def test_check_outcome_fail():
    outcome = CheckOutcome(passed=False, metadata={"reason": "empty"})
    assert outcome.passed is False
    assert outcome.metadata["reason"] == "empty"


def test_check_outcome_default_metadata_empty_dict():
    outcome = CheckOutcome(passed=True)
    assert outcome.metadata == {}
