"""NuExtractHTTPClient payload/parse logic (no network / GPU)."""

from __future__ import annotations

import json

from tamubot.ingestion.clients.nuextract_http_client import build_chat_payload, parse_chat_response


def test_build_chat_payload_embeds_template_and_greedy():
    p = build_chat_payload([{"type": "text", "text": "hello"}], model="numind/NuExtract3")
    assert p["model"] == "numind/NuExtract3"
    assert p["temperature"] == 0.0
    assert p["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    tmpl = json.loads(p["chat_template_kwargs"]["template"])
    assert tmpl["course_code"] == "verbatim-string"  # SYLLABUS_TEMPLATE leaf
    assert p["chat_template_kwargs"]["enable_thinking"] is False


def test_parse_chat_response_extracts_message_content():
    resp = {"choices": [{"message": {"content": '{"course_code": "STAT 608", "credit_hours": 3}'}}]}
    ex = parse_chat_response(resp)
    assert ex.course_code == "STAT 608"
    assert ex.credit_hours == 3


def test_parse_chat_response_tolerates_fenced_json():
    resp = {"choices": [{"message": {"content": '```json\n{"course_code": "CSCE 121"}\n```'}}]}
    assert parse_chat_response(resp).course_code == "CSCE 121"
