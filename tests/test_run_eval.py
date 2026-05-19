"""Tests for tamubot.evals.run_eval helpers — pure unit tests, no external calls."""

import pytest


def test_default_metrics_with_generation():
    from tamubot.evals.run_eval import default_metrics

    m = default_metrics(with_generation=True)
    assert "faithfulness" in m
    assert "answer_relevancy" in m
    assert "context_precision" in m
    assert "context_recall" in m


def test_default_metrics_retrieval_only():
    from tamubot.evals.run_eval import default_metrics

    m = default_metrics(with_generation=False)
    assert "faithfulness" not in m
    assert "context_precision" in m
    assert "context_recall" in m


def test_parse_id_filter_basic():
    from tamubot.evals._runner_common import parse_id_filter

    assert parse_id_filter("3,7,12") == {3, 7, 12}
    assert parse_id_filter(" 1 , 2 , 3 ") == {1, 2, 3}


def test_parse_id_filter_empty_and_none():
    from tamubot.evals._runner_common import parse_id_filter

    assert parse_id_filter(None) is None
    assert parse_id_filter("") is None
    assert parse_id_filter(", ,") is None


def test_parse_id_filter_invalid_raises():
    from tamubot.evals._runner_common import parse_id_filter

    with pytest.raises(ValueError):
        parse_id_filter("3,abc,7")


def test_filter_items_by_ids_keeps_matches():
    from tamubot.evals._runner_common import filter_items_by_ids

    items = [{"id": 1, "question": "a"}, {"id": 2, "question": "b"}, {"id": 3, "question": "c"}]
    kept = filter_items_by_ids(items, {1, 3})
    assert [i["id"] for i in kept] == [1, 3]


def test_filter_items_by_ids_none_passes_through():
    from tamubot.evals._runner_common import filter_items_by_ids

    items = [{"id": 1}, {"id": 2}]
    assert filter_items_by_ids(items, None) == items


def test_filter_items_by_ids_warns_on_missing(caplog):
    import logging

    from tamubot.evals._runner_common import filter_items_by_ids

    items = [{"id": 1}, {"id": 2}]
    with caplog.at_level(logging.WARNING, logger="tamubot.evals.runner_common"):
        kept = filter_items_by_ids(items, {1, 99})
    assert kept == [{"id": 1}]
    assert any("99" in r.message for r in caplog.records)


def test_stable_item_id_is_deterministic():
    from tamubot.evals._runner_common import stable_item_id

    a = stable_item_id("what is CSCE 670?")
    b = stable_item_id("what is CSCE 670?")
    c = stable_item_id("what is CSCE 671?")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_resolve_metrics_default_passthrough():
    from tamubot.rag.observability import resolve_metrics

    assert resolve_metrics(None, ["faithfulness"]) == ["faithfulness"]


def test_resolve_metrics_all_returns_registry():
    from tamubot.rag.observability import available_metrics, resolve_metrics

    assert resolve_metrics("all", []) == available_metrics()


def test_resolve_metrics_subset():
    from tamubot.rag.observability import resolve_metrics

    out = resolve_metrics("faithfulness,context_recall", [])
    assert out == ["faithfulness", "context_recall"]


def test_resolve_metrics_unknown_raises():
    from tamubot.rag.observability import resolve_metrics

    with pytest.raises(ValueError, match="bogus_metric"):
        resolve_metrics("faithfulness,bogus_metric", [])
