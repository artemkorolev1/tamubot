"""parse_extract output-parsing logic (no GPU / model required)."""

from __future__ import annotations

import pytest

from tamubot.ingestion.clients.nuextract_client import parse_extract, plan_batches


def test_parse_clean_json():
    ex = parse_extract('{"course_code": "STAT 608", "credit_hours": 3}')
    assert ex.course_code == "STAT 608"
    assert ex.credit_hours == 3


def test_parse_strips_json_fence():
    ex = parse_extract('```json\n{"course_code": "STAT 615"}\n```')
    assert ex.course_code == "STAT 615"


def test_parse_recovers_from_surrounding_prose():
    ex = parse_extract('Here you go: {"course_code": "CSCE 608"} hope that helps')
    assert ex.course_code == "CSCE 608"


def test_parse_ignores_extra_fields():
    ex = parse_extract('{"course_code": "X", "extra": "junk"}')
    assert ex.course_code == "X"


def test_parse_handles_null_subfields():
    ex = parse_extract('{"assessment_weights": [{"component": "Final", "weight_pct": null}]}')
    assert ex.assessment_weights[0].component == "Final"
    assert ex.assessment_weights[0].weight_pct is None


def test_parse_raises_on_no_json():
    with pytest.raises(ValueError):
        parse_extract("no json object here at all")


# --- plan_batches: length-bucketed dynamic batching ---


def test_plan_batches_empty():
    assert plan_batches([], max_batch_tokens=1000) == []


def test_plan_batches_single_item():
    assert plan_batches([100], max_batch_tokens=1000) == [[0]]


def test_plan_batches_packs_within_token_budget():
    # 3 docs * 100 tok = 300 <= 1000 → one batch
    assert plan_batches([100, 100, 100], max_batch_tokens=1000, max_batch_size=8) == [[0, 1, 2]]


def test_plan_batches_splits_when_token_product_exceeds_budget():
    # budget 250: [0,1] costs 2*100=200 ok; adding a third → 3*100=300 > 250 → new batch
    assert plan_batches([100, 100, 100], max_batch_tokens=250) == [[0, 1], [2]]


def test_plan_batches_long_doc_gets_its_own_batch():
    # sorted by length asc: idx1(10), idx2(10), idx0(1000)
    # [1,2] cost 2*10=20 ok; adding idx0 → 3*1000=3000 > 1200 → idx0 alone
    assert plan_batches([1000, 10, 10], max_batch_tokens=1200, max_batch_size=8) == [[1, 2], [0]]


def test_plan_batches_single_item_over_budget_still_placed():
    # a doc larger than the whole budget can't be split → its own batch
    assert plan_batches([5000], max_batch_tokens=1000) == [[0]]
    # and it doesn't drag a small doc with it
    assert plan_batches([5000, 10], max_batch_tokens=1000) == [[1], [0]]


def test_plan_batches_respects_max_batch_size():
    assert plan_batches([1, 1, 1, 1, 1], max_batch_tokens=10000, max_batch_size=2) == [[0, 1], [2, 3], [4]]


def test_plan_batches_partitions_every_index_exactly_once():
    lengths = [300, 50, 300, 50, 1000, 25]
    batches = plan_batches(lengths, max_batch_tokens=700, max_batch_size=3)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))


# --- get_extractor backend factory ---


def test_get_extractor_http_backend_returns_http_client(monkeypatch):
    from tamubot.core import config
    from tamubot.ingestion.clients import nuextract_client
    from tamubot.ingestion.clients.nuextract_http_client import NuExtractHTTPClient

    monkeypatch.setattr(config, "NUEXTRACT_BACKEND", "http")
    monkeypatch.setattr(config, "NUEXTRACT_SERVER_URL", "http://x:8000/v1")
    monkeypatch.setattr(config, "NUEXTRACT_MODEL", "numind/NuExtract3")

    ex = nuextract_client.get_extractor()  # must NOT load a GPU model
    assert isinstance(ex, NuExtractHTTPClient)
    assert ex.base_url == "http://x:8000/v1"
