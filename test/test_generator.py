"""
tests/test_generator.py
────────────────────────
Tests for LLMGenerator.

Same mocking strategy as query_transformer tests:
  - Unit tests mock the Groq API — no network, no cost
  - One integration test uses real API to verify end-to-end
  - Retry logic tested by making the first mock call fail
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.generation.generator import (
    LLMGenerator,
    GenerationResult,
    GenerationError,
)
from src.generation.prompt_builder import PromptBuilder, PromptPackage
from src.retrieval.reranker import RankedResult
from src.retrieval.retriever import HybridSearchResult
from src.storage.vector_store import SearchResult


# ── Helpers ────────────────────────────────────────────────────────────
def make_test_prompt_package() -> PromptPackage:
    """Build a minimal PromptPackage for generator tests."""
    sr = SearchResult(
        chunk_id="c0", text="Neural networks learn through gradient descent.",
        score=0.9, start_time="1:00", end_time="2:00",
        start_sec=60.0, video_id="vid", youtube_link="", chunk_index=0,
    )
    hs = HybridSearchResult(
        search_result=sr, rrf_score=0.05, sources=["semantic_q1"]
    )
    rr = RankedResult(hybrid_result=hs, rerank_score=7.5, rank=1)

    builder = PromptBuilder()
    return builder.build(
        question = "how do neural networks learn?",
        results  = [rr],
    )


def make_mock_response(content: str) -> MagicMock:
    """Build a fake Groq response with the given content string."""
    msg      = MagicMock(); msg.content = content
    choice   = MagicMock(); choice.message = msg
    response = MagicMock(); response.choices = [choice]
    return response


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def generator() -> LLMGenerator:
    return LLMGenerator()


@pytest.fixture
def mock_api(generator):
    """Patch the Groq API call for unit tests."""
    with patch.object(
        generator._client.chat.completions, "create"
    ) as mock:
        yield mock


@pytest.fixture
def prompt_package() -> PromptPackage:
    return make_test_prompt_package()


# ── Tests: GenerationResult Structure ────────────────────────────────
class TestGenerationResult:

    def test_returns_generation_result(self, generator, mock_api, prompt_package):
        """generate() must return a GenerationResult."""
        mock_api.return_value = make_mock_response(
            "As explained at [1:00], neural networks learn through..."
        )
        result = generator.generate(prompt_package)
        assert isinstance(result, GenerationResult)

    def test_answer_is_string(self, generator, mock_api, prompt_package):
        """answer field must be a non-empty string."""
        mock_api.return_value = make_mock_response("The answer is X.")
        result = generator.generate(prompt_package)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_latency_recorded(self, generator, mock_api, prompt_package):
        """latency_ms must be a positive number."""
        mock_api.return_value = make_mock_response("Some answer.")
        result = generator.generate(prompt_package)
        assert result.latency_ms > 0

    def test_model_recorded(self, generator, mock_api, prompt_package):
        """model field must be set to the configured model name."""
        from src.config import settings
        mock_api.return_value = make_mock_response("Answer.")
        result = generator.generate(prompt_package)
        assert result.model == settings.llm.model_name

    def test_word_count_property(self, generator, mock_api, prompt_package):
        """word_count property counts words in answer."""
        mock_api.return_value = make_mock_response("one two three four five")
        result = generator.generate(prompt_package)
        assert result.word_count == 5

    def test_is_empty_false_for_real_answer(
        self, generator, mock_api, prompt_package
    ):
        """is_empty must be False when answer has content."""
        mock_api.return_value = make_mock_response("A real answer.")
        result = generator.generate(prompt_package)
        assert result.is_empty is False

    def test_was_retried_false_on_first_success(
        self, generator, mock_api, prompt_package
    ):
        """was_retried must be False when first call succeeds."""
        mock_api.return_value = make_mock_response("Success on first try.")
        result = generator.generate(prompt_package)
        assert result.was_retried is False


# ── Tests: Retry Logic ────────────────────────────────────────────────
class TestRetryLogic:
    """
    Retry logic is the most important resilience feature.
    We test it by making the first mock call raise an exception
    and the second succeed.
    """

    def test_retries_on_first_failure(
        self, generator, mock_api, prompt_package
    ):
        """
        First call fails → retry → second call succeeds.
        Result must have was_retried=True and correct answer.
        """
        mock_api.side_effect = [
            Exception("Transient network error"),   # first call fails
            make_mock_response("Retry succeeded!"), # second call works
        ]
        result = generator.generate(prompt_package)
        assert result.was_retried is True
        assert result.answer == "Retry succeeded!"

    def test_raises_after_two_failures(
        self, generator, mock_api, prompt_package, monkeypatch
    ):
        """
        Both calls fail → GenerationError raised.
        monkeypatch removes the sleep() so test doesn't wait 1 second.
        """
        monkeypatch.setattr(
            "src.generation.generator.time.sleep",
            lambda x: None
        )
        mock_api.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
        ]
        with pytest.raises(GenerationError, match="after retry"):
            generator.generate(prompt_package)

    def test_none_package_raises_error(self, generator):
        """None PromptPackage raises GenerationError immediately."""
        with pytest.raises(GenerationError, match="None"):
            generator.generate(None)


# ── Integration Test ──────────────────────────────────────────────────
class TestIntegration:

    @pytest.mark.integration
    def test_real_generation_returns_answer(self):
        """Real Groq API call returns a non-empty answer string."""
        generator = LLMGenerator()
        package   = make_test_prompt_package()
        result    = generator.generate(package)

        assert not result.is_empty
        assert result.word_count > 5
        assert result.latency_ms > 0
        assert result.was_retried is False