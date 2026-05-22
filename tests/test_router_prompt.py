"""Regression guard: ensure use_summary is fully purged from ROUTER_PROMPT."""

from tamubot.rag.prompts import ROUTER_PROMPT


def test_router_prompt_no_use_summary():
    assert "use_summary" not in ROUTER_PROMPT
    assert "USE SUMMARY" not in ROUTER_PROMPT
