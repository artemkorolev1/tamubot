"""Pre-pipeline guards for the public Railway deployment.

For each NEW user turn, before the RAG pipeline runs:
  1. Per-session turn cap (in-memory, free) — short-circuits cheapest first.
  2. Injection / jailbreak check via Lakera Guard /v2/guard.
  3. Global daily turn budget — atomic counter in MongoDB Atlas (circuit breaker).

Ordering guarantees a blocked turn (over-cap or malicious) never consumes the
daily budget. All checks are no-ops when config.GUARD_ENABLED is False, so local
dev and tests need no Lakera key or Atlas counter.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import MutableMapping

import httpx
from pymongo import ReturnDocument

from tamubot.core import config

logger = logging.getLogger("tamubot.guard")

_SESSION_KEY = "guard_turns"

_INJECTION_MSG = (
    "Sorry — I can't process that request. Please rephrase your question about "
    "Texas A&M courses, syllabi, or degree requirements."
)
_SESSION_MSG = "You've reached this session's question limit. Refresh the page to start a new session."
_BUDGET_MSG = "TamuBot has reached its daily usage limit for this demo. Please check back tomorrow."


@dataclass
class GuardDecision:
    allowed: bool
    message: str = ""
    reason: str = ""  # "" | "session_cap" | "injection" | "daily_budget"


def is_enabled() -> bool:
    return config.GUARD_ENABLED


def evaluate(text: str, session_state: MutableMapping, db) -> GuardDecision:
    """Run all guards for one new user turn."""
    if not config.GUARD_ENABLED:
        return GuardDecision(allowed=True)

    used = session_state.get(_SESSION_KEY, 0)
    if used >= config.SESSION_TURN_CAP:
        return GuardDecision(False, _SESSION_MSG, "session_cap")

    if _is_injection(text):
        return GuardDecision(False, _INJECTION_MSG, "injection")

    if _exceeds_daily_budget(db):
        return GuardDecision(False, _BUDGET_MSG, "daily_budget")

    session_state[_SESSION_KEY] = used + 1
    return GuardDecision(allowed=True)


def _is_injection(text: str) -> bool:
    """Return True if Lakera flags the prompt. Fail-open (allow) on any error."""
    try:
        resp = httpx.post(
            f"{config.LAKERA_BASE_URL}/v2/guard",
            headers={"Authorization": f"Bearer {config.LAKERA_GUARD_API_KEY}"},
            json={"messages": [{"role": "user", "content": text}]},
            timeout=config.LAKERA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return bool(resp.json().get("flagged", False))
    except Exception as exc:  # network error, timeout, bad key, malformed body
        logger.warning("Lakera guard unavailable, allowing prompt (fail-open): %s", exc)
        return False


def _exceeds_daily_budget(db) -> bool:
    """Atomically bump today's turn counter; return True if over the daily budget."""
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    doc = db[config.USAGE_COLLECTION].find_one_and_update(
        {"_id": today},
        {"$inc": {"turns": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["turns"] > config.DAILY_TURN_BUDGET
