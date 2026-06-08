"""
tests/test_query_transformer.py
────────────────────────────────
Tests for the QueryTransformer module.

Test strategy:
  Most tests mock the Groq API — they test our code's logic
  without making real network calls. This makes tests:
    - Fast: no network latency
    - Reliable: no rate limits, no API downtime
    - Cheap: no API tokens consumed

  One integration test at the end uses the real API to verify
  the full flow works end-to-end.

  This split — unit tests + one integration test — is the
  professional standard for testing external API integrations.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.query_transformer import (
    QueryTransformer,
    QueryTransformationResult,
    ConversationTurn,
    QueryTransformationError,
)


# ── Mock Helper ──────────────────────────────────────────────────────
def make_mock_groq_response(content: str) -> MagicMock:
    """
    Build a fake Groq API response object.

    The real Groq response has this structure:
      response.choices[0].message.content = "the text"

    We replicate just that structure — nothing else matters.
    """
    mock_message  = MagicMock()
    mock_message.content = content

    mock_choice   = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def transformer() -> QueryTransformer:
    """Fresh QueryTransformer for each test."""
    return QueryTransformer()


@pytest.fixture
def mock_groq_call(transformer):
    """
    Patch the Groq API call so tests never hit the network.

    'patch' replaces transformer._client.chat.completions.create
    with a MagicMock for the duration of each test.
    The original is automatically restored after the test.

    Usage in tests:
        mock_groq_call.return_value = make_mock_groq_response("query 1\nquery 2")
    """
    with patch.object(
        transformer._client.chat.completions,
        "create"
    ) as mock:
        yield mock


# ── Tests: Query Parsing ─────────────────────────────────────────────
class TestQueryParsing:
    """
    Tests for _parse_queries() — our most important internal method.
    Good parsing is critical because LLMs don't always format perfectly.
    """

    def test_parses_clean_output(self, transformer):
        """Clean newline-separated output parses correctly."""
        raw    = "how neural networks learn\ngradient descent optimization\nbackpropagation weight updates"
        result = transformer._parse_queries(raw, "original question")
        assert len(result) == 3
        assert "how neural networks learn" in result

    def test_strips_numbered_bullets(self, transformer):
        """Numbered list format is cleaned: '1. query' → 'query'."""
        raw    = "1. first query\n2. second query\n3. third query"
        result = transformer._parse_queries(raw, "original")
        for q in result[:3]:
            assert not q[0].isdigit()

    def test_strips_dash_bullets(self, transformer):
        """Dash bullet format is cleaned: '- query' → 'query'."""
        raw    = "- first query\n- second query\n- third query"
        result = transformer._parse_queries(raw, "original")
        for q in result[:3]:
            assert not q.startswith("-")

    def test_original_question_appended_as_fallback(self, transformer):
        """Original question always included if not already in output."""
        raw    = "query one\nquery two"   # only 2 lines, fewer than n_queries
        result = transformer._parse_queries(raw, "my original question")
        assert "my original question" in result

    def test_deduplicates_queries(self, transformer):
        """Duplicate queries are removed — case insensitive."""
        raw    = "how transformers work\nHow Transformers Work\ndifferent query"
        result = transformer._parse_queries(raw, "original")
        lower  = [q.lower() for q in result]
        assert len(lower) == len(set(lower)), "Duplicates found after parsing"

    def test_empty_lines_ignored(self, transformer):
        """Blank lines in LLM output are not included as queries."""
        raw    = "query one\n\n\nquery two\n\nquery three"
        result = transformer._parse_queries(raw, "original")
        assert all(q.strip() != "" for q in result)


# ── Tests: Context Block Building ───────────────────────────────────
class TestContextBlock:

    def test_no_history_returns_empty_string(self, transformer):
        """No history → empty context block → cleaner prompt."""
        result = transformer._build_context_block(None)
        assert result == ""

    def test_history_included_in_block(self, transformer):
        """History questions and answers appear in context block."""
        history = [
            ConversationTurn(
                question="what is backpropagation?",
                answer  ="Backpropagation is the algorithm that..."
            )
        ]
        block = transformer._build_context_block(history)
        assert "backpropagation" in block.lower()

    def test_long_answers_truncated(self, transformer):
        """Answers over 200 chars are truncated to keep prompt lean."""
        history = [
            ConversationTurn(
                question="short question",
                answer  ="x" * 500   # very long answer
            )
        ]
        block = transformer._build_context_block(history)
        # 500 chars of 'x' should not all appear in context
        assert "x" * 500 not in block
        assert "..." in block    # truncation marker present

    def test_only_last_n_turns_included(self, transformer):
        """Only the most recent history_turns are included."""
        history = [
            ConversationTurn(f"question {i}", f"answer {i}")
            for i in range(10)    # 10 turns of history
        ]
        block = transformer._build_context_block(history)
        # First question (oldest) should NOT be in context
        assert "question 0" not in block
        # Last question (newest) SHOULD be in context
        assert "question 9" in block


# ── Tests: Transform Method ──────────────────────────────────────────
class TestTransform:

    def test_returns_correct_type(self, transformer, mock_groq_call):
        """transform() must return a QueryTransformationResult."""
        mock_groq_call.return_value = make_mock_groq_response(
            "query one\nquery two\nquery three"
        )
        result = transformer.transform("how do transformers work?")
        assert isinstance(result, QueryTransformationResult)

    def test_result_has_queries(self, transformer, mock_groq_call):
        """Result must contain a non-empty list of queries."""
        mock_groq_call.return_value = make_mock_groq_response(
            "semantic search mechanisms\nvector similarity retrieval\nembedding space nearest neighbour"
        )
        result = transformer.transform("how does search work?")
        assert isinstance(result.queries, list)
        assert len(result.queries) > 0

    def test_original_question_preserved(self, transformer, mock_groq_call):
        """original_question field must match input exactly."""
        mock_groq_call.return_value = make_mock_groq_response(
            "query one\nquery two\nquery three"
        )
        question = "what is attention mechanism?"
        result   = transformer.transform(question)
        assert result.original_question == question

    def test_used_fallback_false_on_success(self, transformer, mock_groq_call):
        """used_fallback must be False when LLM call succeeds."""
        mock_groq_call.return_value = make_mock_groq_response(
            "query one\nquery two\nquery three"
        )
        result = transformer.transform("test question")
        assert result.used_fallback is False

    def test_latency_recorded(self, transformer, mock_groq_call):
        """latency_ms must be a positive number."""
        mock_groq_call.return_value = make_mock_groq_response(
            "query one\nquery two\nquery three"
        )
        result = transformer.transform("test question")
        assert result.latency_ms > 0

    def test_primary_query_property(self, transformer, mock_groq_call):
        """primary_query returns the first query in the list."""
        mock_groq_call.return_value = make_mock_groq_response(
            "first query here\nsecond query\nthird query"
        )
        result = transformer.transform("test question")
        assert result.primary_query == result.queries[0]


# ── Tests: Graceful Degradation ──────────────────────────────────────
class TestGracefulDegradation:
    """
    The most important behaviour to test: what happens when
    the LLM API fails. The system must NEVER crash — it must
    degrade gracefully to the original question.
    """

    def test_api_failure_returns_original_question(
        self, transformer, mock_groq_call
    ):
        """When API raises any exception, original question is returned."""
        mock_groq_call.side_effect = Exception("Connection timeout")

        result = transformer.transform("test question")

        assert result.used_fallback is True
        assert "test question" in result.queries

    def test_api_failure_does_not_raise(
        self, transformer, mock_groq_call
    ):
        """API failure must NOT propagate as exception to the caller."""
        mock_groq_call.side_effect = RuntimeError("Rate limit exceeded")

        # This must not raise — graceful degradation
        result = transformer.transform("test question")
        assert result is not None

    def test_fallback_result_is_usable(
        self, transformer, mock_groq_call
    ):
        """Even fallback result must have valid structure for the pipeline."""
        mock_groq_call.side_effect = ConnectionError("No internet")

        result = transformer.transform("what is RAG?")

        assert isinstance(result.queries, list)
        assert len(result.queries) >= 1
        assert result.original_question == "what is RAG?"


# ── Tests: Input Validation ──────────────────────────────────────────
class TestInputValidation:

    def test_empty_question_raises_error(self, transformer):
        """Empty string input raises QueryTransformationError."""
        with pytest.raises(QueryTransformationError, match="empty"):
            transformer.transform("")

    def test_whitespace_question_raises_error(self, transformer):
        """Whitespace-only input raises QueryTransformationError."""
        with pytest.raises(QueryTransformationError, match="empty"):
            transformer.transform("   ")


# ── Integration Test: Real API ────────────────────────────────────────
class TestIntegration:
    """
    One real API call to verify end-to-end behaviour.
    Marked with @pytest.mark.integration so it can be
    skipped in fast unit test runs:
      pytest tests/ -m "not integration"   ← skip API calls
      pytest tests/ -m integration          ← only API calls
    """

    @pytest.mark.integration
    def test_real_transform_produces_queries(self):
        """Real Groq call produces 3 distinct, non-empty queries."""
        transformer = QueryTransformer()
        result = transformer.transform(
            "how does gradient descent work in neural networks?"
        )

        assert len(result.queries) >= 1
        assert result.used_fallback is False
        assert all(len(q) > 5 for q in result.queries)
        # All queries must be strings
        assert all(isinstance(q, str) for q in result.queries)

    @pytest.mark.integration
    def test_followup_with_history_resolves_pronoun(self):
        """Vague follow-up with history resolves correctly."""
        transformer = QueryTransformer()
        history     = [
            ConversationTurn(
                question = "what is backpropagation?",
                answer   = "Backpropagation is the algorithm used to train "
                           "neural networks by computing gradients..."
            )
        ]
        result = transformer.transform(
            "can you explain that more simply?",
            history=history,
        )
        # Queries should mention backpropagation or neural networks
        # — not just repeat "explain that more simply"
        combined = " ".join(result.queries).lower()
        assert any(
            word in combined
            for word in ["backpropagation", "neural", "gradient", "training"]
        ), f"Pronoun not resolved. Queries: {result.queries}"