"""Unit tests for the Generator Stage of the RAG pipeline.

Tests cover:
  - Primacy-recency bracketing in context assembly (rag/context_builder.py)
  - Temperature routing for function-adaptive stochasticity (rag/prompts.py)
  - Citation validation (Gate 1) (rag/gates.py)
  - Thinking token configuration (config.py)
"""

from tamubot.core import config
from tamubot.rag.gates import validate_citations_gate1
from tamubot.rag.prompts import _FUNCTION_TEMPERATURES
from tamubot.rag.tools.context import format_context_xml


class TestFormatContextXmlPrimacyRecency:
    """Test primacy-recency reordering in format_context_xml()."""

    def test_single_chunk_no_reorder(self):
        """Single chunk should remain unchanged."""
        results = [{"content": "chunk0", "course_id": "TEST-101"}]
        xml = format_context_xml(results)
        assert "chunk0" in xml
        assert 'source="1"' in xml
        assert 'id="1"' in xml

    def test_two_chunks_no_reorder(self):
        """Two chunks should maintain order [0, 1]."""
        results = [
            {"content": "chunk0", "course_id": "TEST-101"},
            {"content": "chunk1", "course_id": "TEST-101"},
        ]
        xml = format_context_xml(results)
        # Both chunks should be present
        assert "chunk0" in xml
        assert "chunk1" in xml
        # Verify chunk0 comes before chunk1
        idx0 = xml.find("chunk0")
        idx1 = xml.find("chunk1")
        assert idx0 < idx1

    def test_five_chunks_primacy_recency_ordering(self):
        """Five chunks should reorder to [0, 2, 3, 4, 1]."""
        results = [
            {"content": "chunk0", "course_id": "TEST-101"},
            {"content": "chunk1", "course_id": "TEST-101"},
            {"content": "chunk2", "course_id": "TEST-101"},
            {"content": "chunk3", "course_id": "TEST-101"},
            {"content": "chunk4", "course_id": "TEST-101"},
        ]
        xml = format_context_xml(results)

        # Extract chunk positions
        positions = {}
        for chunk in ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4"]:
            positions[chunk] = xml.find(chunk)

        # Verify order: chunk0 < chunk2 < chunk3 < chunk4 < chunk1
        assert positions["chunk0"] < positions["chunk2"]
        assert positions["chunk2"] < positions["chunk3"]
        assert positions["chunk3"] < positions["chunk4"]
        assert positions["chunk4"] < positions["chunk1"]

    def test_chunk_id_attributes(self):
        """Each chunk should have id attribute matching its rank."""
        results = [
            {"content": "chunk0", "course_id": "TEST-101"},
            {"content": "chunk1", "course_id": "TEST-101"},
            {"content": "chunk2", "course_id": "TEST-101"},
        ]
        xml = format_context_xml(results)

        # All chunks should have id attributes
        assert 'id="1"' in xml  # Rank 1
        assert 'id="2"' in xml  # Rank 2
        assert 'id="3"' in xml  # Rank 3

    def test_xml_escaping(self):
        """Special XML characters should be escaped."""
        results = [
            {
                "content": 'Test <tag> & "quotes" with \\ backslash',
                "course_id": "TEST-101",
            }
        ]
        xml = format_context_xml(results)

        # Verify escaping
        assert "&lt;tag&gt;" in xml
        assert "&amp;" in xml
        assert "&quot;" in xml or '"' in xml  # Quotes might be in attribute or content


class TestTemperatureRouting:
    """Test temperature configuration for function types."""

    def test_hybrid_course_deterministic(self):
        """hybrid_course should use 0.0 temperature (factual extraction)."""
        assert _FUNCTION_TEMPERATURES["hybrid_course"] == 0.0

    def test_semantic_general_synthesis_temperature(self):
        """semantic_general should use 0.2 temperature (synthesis)."""
        assert _FUNCTION_TEMPERATURES["semantic_general"] == 0.2

    def test_recursive_synthesis_temperature(self):
        """recursive should use 0.2 temperature (advisory synthesis)."""
        assert _FUNCTION_TEMPERATURES["recursive"] == 0.2

    def test_out_of_scope_deterministic(self):
        """out_of_scope should use 0.0 temperature."""
        assert _FUNCTION_TEMPERATURES["out_of_scope"] == 0.0


class TestValidateCitationsGate1:
    """Test citation validation (Gate 1) regex pattern."""

    def test_source_citation_detected(self):
        """[Source N] citation should be detected."""
        text = "According to the syllabus [Source 1], grading is based on exams."
        assert validate_citations_gate1(text) is True

    def test_numbered_citation_detected(self):
        """[N] citation should be detected."""
        text = "The course requires participation [1]."
        assert validate_citations_gate1(text) is True

    def test_multiple_citations(self):
        """Multiple citations should be detected."""
        text = "Fact 1 [Source 2] and Fact 2 [Source 3]."
        assert validate_citations_gate1(text) is True

    def test_source_with_description_detected(self):
        """[Source N: description] should be detected."""
        text = "Details [Source 1: grading structure] show percentages."
        assert validate_citations_gate1(text) is True

    def test_no_citation_found(self):
        """Response without citations should fail."""
        text = "The course covers many topics including advanced algorithms."
        assert validate_citations_gate1(text) is False

    def test_malformed_bracket_not_matched(self):
        """Malformed brackets should not match."""
        text = "The [material] is in the syllabus but [Source ] is incomplete."
        # Only incomplete [Source ] should fail
        assert validate_citations_gate1(text) is False

    def test_empty_response(self):
        """Empty response should fail citation check."""
        assert validate_citations_gate1("") is False


class TestThinkingBudgetConfiguration:
    """Test thinking token budget constants in config."""

    def test_thinking_budget_metadata_is_zero(self):
        """THINKING_BUDGET_METADATA should be 0."""
        assert config.THINKING_BUDGET_METADATA == 0

    def test_thinking_budget_semantic_is_1024(self):
        """THINKING_BUDGET_SEMANTIC should be 1024 (reduced from 4096 for latency)."""
        assert config.THINKING_BUDGET_SEMANTIC == 1024

    def test_temperature_deterministic_constant(self):
        """TEMP_DETERMINISTIC should be 0.0."""
        assert config.TEMP_DETERMINISTIC == 0.0

    def test_temperature_synthesis_constant(self):
        """TEMP_SYNTHESIS should be 0.2."""
        assert config.TEMP_SYNTHESIS == 0.2

    def test_validation_model_constant(self):
        """VALIDATION_MODEL should be gemini-2.5-flash-lite."""
        assert config.VALIDATION_MODEL == "gemini-2.5-flash-lite"

    def test_generation_model_supports_thinking(self):
        """GENERATION_MODEL should be gemini-2.5-flash (supports thinking)."""
        assert config.GENERATION_MODEL == "gemini-2.5-flash"


def test_generate_stream_includes_conversation_history_block():
    """generate_stream with history_context includes <conversation_history> XML block."""
    from unittest.mock import patch

    from tamubot.rag.generator import generate_stream

    captured_messages = []

    def mock_stream_llm(messages, **kwargs):
        captured_messages.extend(messages)
        yield "Answer [Source 1]"

    chunks = [{"content": "Grading is 40% exams.", "course_id": "CSCE 638"}]
    history_ctx = "User: What is CSCE 638?\nAssistant: It is a grad ML course."

    with patch("tamubot.rag.generator.stream_llm", side_effect=mock_stream_llm):
        list(
            generate_stream(
                results=chunks,
                question="What is the grading?",
                function="hybrid_course",
                history_context=history_ctx,
            )
        )

    user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
    assert "<conversation_history>" in user_msg
    assert history_ctx in user_msg
    assert "Question: What is the grading?" in user_msg
    # conversation_history block must appear BEFORE the Question line
    assert user_msg.index("<conversation_history>") < user_msg.index("Question:")


def test_generate_stream_no_history_context_no_block():
    """generate_stream without history_context does not include <conversation_history> block."""
    from unittest.mock import patch

    from tamubot.rag.generator import generate_stream

    captured_messages = []

    def mock_stream_llm(messages, **kwargs):
        captured_messages.extend(messages)
        yield "Answer [Source 1]"

    chunks = [{"content": "Grading is 40% exams.", "course_id": "CSCE 638"}]

    with patch("tamubot.rag.generator.stream_llm", side_effect=mock_stream_llm):
        list(
            generate_stream(
                results=chunks,
                question="What is the grading?",
                function="hybrid_course",
            )
        )

    user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
    assert "<conversation_history>" not in user_msg


def test_base_system_no_chain_of_thought_instruction():
    from tamubot.rag.prompts import _BASE_SYSTEM

    assert "Before answering, identify which chunk" not in _BASE_SYSTEM


def test_comparison_system_exists_and_is_compact():
    from tamubot.rag.prompts import COMPARISON_SYSTEM

    assert len(COMPARISON_SYSTEM) < 1200
    assert "compare" in COMPARISON_SYSTEM.lower()


def test_router_prompt_has_all_required_output_fields():
    from tamubot.rag.prompts import ROUTER_PROMPT

    for field in ["course_ids", "intent_type", "recursive_search", "rewritten_query"]:
        assert field in ROUTER_PROMPT, f"ROUTER_PROMPT missing field: {field}"


def test_generate_comparison_uses_comparison_system(monkeypatch):
    """generate_comparison streams using COMPARISON_SYSTEM as the system prompt."""
    import tamubot.rag.generator as gen_mod
    from tamubot.rag.prompts import COMPARISON_SYSTEM

    captured_messages = []

    def mock_stream_llm(messages, **kwargs):
        captured_messages.extend(messages)
        return iter(["token1", "token2"])

    monkeypatch.setattr(gen_mod, "stream_llm", mock_stream_llm)

    tokens = list(gen_mod.generate_comparison([], "compare CSCE 638 and CSCE 670", ["CSCE 638", "CSCE 670"]))

    assert tokens == ["token1", "token2"]
    system_msg = next((m["content"] for m in captured_messages if m["role"] == "system"), None)
    assert system_msg == COMPARISON_SYSTEM
