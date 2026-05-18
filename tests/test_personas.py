"""Unit tests for tamubot.evals.personas.load_personas."""

from pathlib import Path

import pytest


def test_load_single_persona_from_yaml(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    yaml_text = (
        "personas:\n  - name: Test Persona\n    role_description: |\n      A student deciding between courses.\n"
    )
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)

    personas = load_personas(p)

    assert len(personas) == 1
    assert personas[0].name == "Test Persona"
    assert "A student deciding between courses." in personas[0].role_description


def test_load_personas_returns_ragas_persona_instances(tmp_path: Path) -> None:
    from ragas.testset.persona import Persona

    from tamubot.evals.personas import load_personas

    p = tmp_path / "p.yaml"
    p.write_text("personas:\n  - name: X\n    role_description: y\n")

    personas = load_personas(p)

    assert isinstance(personas[0], Persona)


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    with pytest.raises(FileNotFoundError):
        load_personas(tmp_path / "does_not_exist.yaml")


def test_empty_personas_list_raises_valueerror(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    p = tmp_path / "p.yaml"
    p.write_text("personas: []\n")

    with pytest.raises(ValueError, match="at least one persona"):
        load_personas(p)


def test_missing_personas_key_raises_valueerror(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    p = tmp_path / "p.yaml"
    p.write_text("other_key: value\n")

    with pytest.raises(ValueError, match="personas"):
        load_personas(p)


def test_load_multiple_personas(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    yaml_text = "personas:\n  - name: A\n    role_description: alpha\n  - name: B\n    role_description: beta\n"
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)

    personas = load_personas(p)

    assert [pp.name for pp in personas] == ["A", "B"]


# ---------------------------------------------------------------------------
# Prompt-nudge regression: build_query_distribution must attach a durable-
# attribute instruction to the SingleHopSpecificQuerySynthesizer.
# ---------------------------------------------------------------------------


def test_single_hop_synthesizer_carries_durable_attribute_nudge() -> None:
    """The single-hop synthesizer's instruction must mention durable attributes
    and explicitly steer away from term-bound deadlines."""
    from unittest.mock import MagicMock

    from tamubot.evals.generate_ragas_testset import build_query_distribution

    dist = build_query_distribution(llm=MagicMock(), preset="balanced_50_50")

    single_hop, weight = dist[0]
    assert weight == 0.50
    instruction = single_hop.get_prompts()["query_answer_generation_prompt"].instruction

    assert "durable course attributes" in instruction
    assert "term-bound deadlines" in instruction


def test_multi_hop_abstract_synthesizer_is_unmodified() -> None:
    """The multi-hop abstract synthesizer must NOT receive the nudge."""
    from unittest.mock import MagicMock

    from tamubot.evals.generate_ragas_testset import build_query_distribution

    dist = build_query_distribution(llm=MagicMock(), preset="balanced_50_50")

    multi_hop, weight = dist[1]
    assert weight == 0.50
    # Inspect every prompt on the multi-hop synthesizer; none should carry
    # the single-hop-specific nudge marker.
    for prompt in multi_hop.get_prompts().values():
        assert "durable course attributes" not in prompt.instruction
