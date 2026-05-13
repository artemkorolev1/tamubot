"""Unit tests for LLM guardrails in rag/tools/llm.py.

All tests are pure unit tests — no real API calls, no tokens consumed.
Tests cover:
  - Output token hard cap (call_llm + stream_llm)
  - Input token guard (soft warn + hard reject)
  - Retry with exponential backoff on transient errors
  - Timeout guard
"""

from unittest.mock import patch

import pytest

from tamubot.rag.tools.llm import (
    InputTooLargeError,
    LLMResult,
    LLMTimeoutError,
    _check_input_guard,
    _is_retryable,
    _truncate_output,
    call_llm,
    stream_llm,
)

# ---------------------------------------------------------------------------
# _truncate_output
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    """Test the output truncation helper."""

    def test_short_text_unchanged(self):
        text = "Hello world."
        result, truncated = _truncate_output(text, max_tokens=100)
        assert result == text
        assert truncated is False

    def test_long_text_truncated(self):
        # ~250 tokens worth of text (each word is ~1 token)
        text = " ".join(f"word{i}" for i in range(500))
        result, truncated = _truncate_output(text, max_tokens=50)
        assert truncated is True
        assert "*[Response truncated — output limit reached]*" in result

    def test_truncation_respects_sentence_boundary(self):
        # Build text with clear sentence boundaries
        sentences = [f"Sentence number {i}. " for i in range(200)]
        text = "".join(sentences)
        result, truncated = _truncate_output(text, max_tokens=50)
        assert truncated is True
        # Should not end mid-word (apart from the truncation notice)
        body = result.split("\n\n*[Response truncated")[0]
        assert body.rstrip().endswith(".")

    def test_empty_text_not_truncated(self):
        result, truncated = _truncate_output("", max_tokens=100)
        assert result == ""
        assert truncated is False


# ---------------------------------------------------------------------------
# _check_input_guard
# ---------------------------------------------------------------------------


class TestInputGuard:
    """Test input token limit checking."""

    def test_small_input_passes(self):
        messages = [{"role": "user", "content": "Hello"}]
        # Should not raise
        _check_input_guard(messages)

    @patch("tamubot.rag.tools.llm.config")
    def test_hard_limit_raises(self, mock_config):
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 10
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 5
        # ~50 tokens
        messages = [{"role": "user", "content": " ".join(f"word{i}" for i in range(50))}]
        with pytest.raises(InputTooLargeError):
            _check_input_guard(messages)

    @patch("tamubot.rag.tools.llm.config")
    @patch("tamubot.rag.tools.llm._logger")
    def test_soft_limit_warns(self, mock_logger, mock_config):
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 5
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        messages = [{"role": "user", "content": " ".join(f"word{i}" for i in range(50))}]
        _check_input_guard(messages)
        mock_logger.warning.assert_called_once()
        assert "soft limit" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------


class TestIsRetryable:
    """Test transient error detection."""

    def test_timeout_error_retryable(self):
        assert _is_retryable(TimeoutError("timed out")) is True

    def test_value_error_not_retryable(self):
        assert _is_retryable(ValueError("bad value")) is False

    def test_generic_exception_not_retryable(self):
        assert _is_retryable(Exception("unknown")) is False


# ---------------------------------------------------------------------------
# call_llm — output cap
# ---------------------------------------------------------------------------


class TestCallLlmOutputCap:
    """Test that call_llm truncates oversized output."""

    @patch("tamubot.rag.tools.llm._call_with_timeout")
    @patch("tamubot.rag.tools.llm.config")
    def test_output_truncated_at_cap(self, mock_config, mock_call):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 20
        mock_config.LLM_MAX_RETRIES = 0
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        # Return a very long response
        long_text = " ".join(f"word{i}" for i in range(500))
        mock_call.return_value = LLMResult(text=long_text, input_tokens=10, output_tokens=500)

        result = call_llm([{"role": "user", "content": "hi"}])
        assert "*[Response truncated — output limit reached]*" in result.text

    @patch("tamubot.rag.tools.llm._call_with_timeout")
    @patch("tamubot.rag.tools.llm.config")
    def test_short_output_not_truncated(self, mock_config, mock_call):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 0
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        mock_call.return_value = LLMResult(text="Short answer.", input_tokens=5, output_tokens=3)

        result = call_llm([{"role": "user", "content": "hi"}])
        assert result.text == "Short answer."
        assert "*truncated*" not in result.text


# ---------------------------------------------------------------------------
# call_llm — retry
# ---------------------------------------------------------------------------


class TestCallLlmRetry:
    """Test retry behaviour on transient errors."""

    @patch("tamubot.rag.tools.llm.time.sleep")
    @patch("tamubot.rag.tools.llm._call_with_timeout")
    @patch("tamubot.rag.tools.llm.config")
    def test_retries_on_timeout(self, mock_config, mock_call, mock_sleep):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 2
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999

        # Fail twice with TimeoutError, succeed on third
        mock_call.side_effect = [
            TimeoutError("timeout 1"),
            TimeoutError("timeout 2"),
            LLMResult(text="Success.", input_tokens=5, output_tokens=2),
        ]

        result = call_llm([{"role": "user", "content": "hi"}])
        assert result.text == "Success."
        assert mock_call.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("tamubot.rag.tools.llm.time.sleep")
    @patch("tamubot.rag.tools.llm._call_with_timeout")
    @patch("tamubot.rag.tools.llm.config")
    def test_gives_up_after_max_retries(self, mock_config, mock_call, mock_sleep):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 2
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999

        mock_call.side_effect = TimeoutError("always fails")

        with pytest.raises(TimeoutError, match="always fails"):
            call_llm([{"role": "user", "content": "hi"}])
        assert mock_call.call_count == 3  # 1 initial + 2 retries

    @patch("tamubot.rag.tools.llm.time.sleep")
    @patch("tamubot.rag.tools.llm._call_with_timeout")
    @patch("tamubot.rag.tools.llm.config")
    def test_no_retry_on_non_transient_error(self, mock_config, mock_call, mock_sleep):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 3
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999

        mock_call.side_effect = ValueError("bad schema")

        with pytest.raises(ValueError, match="bad schema"):
            call_llm([{"role": "user", "content": "hi"}])
        assert mock_call.call_count == 1  # no retry
        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# call_llm — input guard
# ---------------------------------------------------------------------------


class TestCallLlmInputGuard:
    """Test that call_llm rejects oversized input."""

    @patch("tamubot.rag.tools.llm.config")
    def test_input_too_large_raises(self, mock_config):
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 10
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 5
        big_input = " ".join(f"word{i}" for i in range(500))

        with pytest.raises(InputTooLargeError):
            call_llm([{"role": "user", "content": big_input}])


# ---------------------------------------------------------------------------
# call_llm — timeout
# ---------------------------------------------------------------------------


class TestCallLlmTimeout:
    """Test timeout guard on call_llm."""

    @patch("tamubot.rag.tools.llm._call_raw")
    @patch("tamubot.rag.tools.llm.config")
    def test_timeout_raises(self, mock_config, mock_call_raw):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 0
        mock_config.LLM_TIMEOUT_SECONDS = 1
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999

        import time as _time

        def slow_call(*args, **kwargs):
            _time.sleep(5)
            return LLMResult(text="late", input_tokens=1, output_tokens=1)

        mock_call_raw.side_effect = slow_call

        with pytest.raises(LLMTimeoutError, match="exceeded timeout"):
            call_llm([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# stream_llm — output cap
# ---------------------------------------------------------------------------


class TestStreamLlmOutputCap:
    """Test that stream_llm stops yielding at the output cap."""

    @patch("tamubot.rag.tools.llm.config")
    def test_stream_truncated_at_cap(self, mock_config):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 10
        mock_config.LLM_MAX_RETRIES = 0
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        mock_config.USE_TAMU_API = False

        # Mock the Gemini stream to yield many tokens
        tokens = [f"word{i} " for i in range(200)]

        with patch("tamubot.rag.tools.llm._stream_gemini", return_value=iter(tokens)):
            collected = list(stream_llm([{"role": "user", "content": "hi"}]))

        # Should have been cut short with a truncation message
        full = "".join(collected)
        assert "*[Response truncated — output limit reached]*" in full
        # Should NOT contain all 200 words
        assert "word199" not in full

    @patch("tamubot.rag.tools.llm.config")
    def test_stream_short_not_truncated(self, mock_config):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 0
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        mock_config.USE_TAMU_API = False

        tokens = ["Hello ", "world."]

        with patch("tamubot.rag.tools.llm._stream_gemini", return_value=iter(tokens)):
            collected = list(stream_llm([{"role": "user", "content": "hi"}]))

        assert "".join(collected) == "Hello world."


# ---------------------------------------------------------------------------
# stream_llm — retry
# ---------------------------------------------------------------------------


class TestStreamLlmRetry:
    """Test retry behaviour for stream_llm."""

    @patch("tamubot.rag.tools.llm.time.sleep")
    @patch("tamubot.rag.tools.llm.config")
    def test_stream_retries_on_transient(self, mock_config, mock_sleep):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 2
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        mock_config.USE_TAMU_API = False

        call_count = 0

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("stream timeout")
            return iter(["OK"])

        with patch("tamubot.rag.tools.llm._stream_gemini", side_effect=mock_stream):
            collected = list(stream_llm([{"role": "user", "content": "hi"}]))

        assert "".join(collected) == "OK"
        assert call_count == 3

    @patch("tamubot.rag.tools.llm.time.sleep")
    @patch("tamubot.rag.tools.llm.config")
    def test_stream_gives_up_after_max_retries(self, mock_config, mock_sleep):
        mock_config.LLM_MAX_OUTPUT_TOKENS = 2000
        mock_config.LLM_MAX_RETRIES = 1
        mock_config.LLM_INPUT_TOKEN_SOFT_LIMIT = 999999
        mock_config.LLM_INPUT_TOKEN_HARD_LIMIT = 999999
        mock_config.USE_TAMU_API = False

        with patch(
            "tamubot.rag.tools.llm._stream_gemini",
            side_effect=TimeoutError("always"),
        ):
            with pytest.raises(TimeoutError, match="always"):
                list(stream_llm([{"role": "user", "content": "hi"}]))
