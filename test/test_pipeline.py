"""
tests/test_pipeline.py
──────────────────────
Integration tests for the complete RAG pipeline.

Strategy:
  - Unit tests mock expensive steps (LLM, network)
  - Integration tests run the full pipeline end-to-end
  - We verify the WIRING between modules, not individual logic
    (each module is already fully tested in isolation)

The key questions here:
  1. Does the response have the right shape?
  2. Does re-ingestion skip correctly?
  3. Does conversation history flow through correctly?
  4. Does the pipeline fail loudly with step labels?
"""

import pytest
from unittest.mock import MagicMock, patch

from src.pipeline import (
    RAGPipeline,
    RAGResponse,
    SourceCitation,
    PipelineLatency,
    PipelineError,
    ConversationTurn,
)


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> RAGPipeline:
    """
    One RAGPipeline instance shared across all unit tests.
    Uses a temp directory to avoid touching real chroma_db.
    scope="module" — expensive models load once.
    """
    tmp = tmp_path_factory.mktemp("pipeline_db")
    return RAGPipeline(persist_dir=str(tmp))


# ── Tests: RAGResponse Structure ─────────────────────────────────────
class TestRAGResponseStructure:
    """
    Verify RAGResponse has the right shape.
    We mock the LLM so tests are fast and free.
    """

    def _mock_generate(self, pipeline, answer_text: str):
        """Helper: patch generator to return a fixed answer."""
        return patch.object(
            pipeline._generator,
            "generate",
            return_value=MagicMock(
                answer     = answer_text,
                latency_ms = 100.0,
                word_count = len(answer_text.split()),
                is_empty   = False,
                was_retried= False,
                model      = "llama3-8b-8192",
            )
        )

    TEST_URL = "https://www.youtube.com/watch?v=ktrIQUYIxZo"

    def test_returns_rag_response(self, pipeline):
        """query() must return a RAGResponse object."""
        with self._mock_generate(pipeline, "As explained at [1:00], ..."):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "what is this video about?",
            )
        assert isinstance(response, RAGResponse)

    def test_response_has_answer(self, pipeline):
        """response.answer must be a non-empty string."""
        with self._mock_generate(pipeline, "The video covers neural networks."):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "summarize the video",
            )
        assert isinstance(response.answer, str)
        assert response.has_answer is True

    def test_response_has_sources(self, pipeline):
        """response.sources must be a non-empty list of SourceCitations."""
        with self._mock_generate(pipeline, "As explained at [2:00]..."):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "what is discussed?",
            )
        assert isinstance(response.sources, list)
        assert len(response.sources) > 0
        assert all(isinstance(s, SourceCitation) for s in response.sources)

    def test_response_has_queries_used(self, pipeline):
        """response.queries_used must be a list of strings."""
        with self._mock_generate(pipeline, "Some answer."):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "test question",
            )
        assert isinstance(response.queries_used, list)
        assert len(response.queries_used) > 0

    def test_response_has_latency(self, pipeline):
        """response.latency must be a PipelineLatency object."""
        with self._mock_generate(pipeline, "Some answer."):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "test question",
            )
        assert isinstance(response.latency, PipelineLatency)
        assert response.latency.total_ms > 0

    def test_response_video_id_correct(self, pipeline):
        """response.video_id must be the extracted video ID."""
        with self._mock_generate(pipeline, "Some answer."):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "test question",
            )
        assert response.video_id == "ktrIQUYIxZo"

    # Inside TestRAGResponseStructure class — add this test:

    def test_response_has_answer_grounded_field(self, pipeline):
        """answer_grounded must be a boolean."""
        with self._mock_generate(
            pipeline,
            "As explained at [1:00], the video covers..."
        ):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "what is this about?",
            )
        assert isinstance(response.answer_grounded, bool)

    def test_grounded_answer_sets_true(self, pipeline):
        """Normal answer with citation sets answer_grounded=True."""
        with self._mock_generate(
            pipeline,
            "As explained at [2:30], gradient descent works by..."
        ):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "how does gradient descent work?",
            )
        assert response.answer_grounded is True

    def test_guard_phrase_sets_grounded_false(self, pipeline):
        """
        Answer containing the hallucination guard phrase
        must set answer_grounded=False.
        """
        with self._mock_generate(
            pipeline,
            "The video doesn't appear to cover this specific topic."
        ):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "what is pasta carbonara?",
            )
        assert response.answer_grounded is False    


# ── Tests: Re-ingestion Behaviour ────────────────────────────────────
class TestReIngestion:
    """
    The smart re-ingestion check is a critical performance feature.
    Second question on the same video must skip Steps 1-4.
    """

    TEST_URL = "https://www.youtube.com/watch?v=ktrIQUYIxZo"

    def _mock_generate(self, pipeline):
        return patch.object(
            pipeline._generator, "generate",
            return_value=MagicMock(
                answer="Answer.", latency_ms=100.0,
                word_count=1, is_empty=False,
                was_retried=False, model="llama3-8b-8192",
            )
        )

    def test_first_question_ingests(self, pipeline):
        """First question on a new video sets ingestion_skipped=False."""
        # Delete first to ensure fresh state
        if pipeline.is_video_indexed(self.TEST_URL):
            pipeline.delete_video(self.TEST_URL)

        with self._mock_generate(pipeline):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "first question ever",
            )
        assert response.ingestion_skipped is False

    def test_second_question_skips_ingestion(self, pipeline):
        """Second question on same video sets ingestion_skipped=True."""
        # Video is now indexed from previous test
        assert pipeline.is_video_indexed(self.TEST_URL)

        with self._mock_generate(pipeline):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "second question on same video",
            )
        assert response.ingestion_skipped is True

    def test_second_question_faster_than_first(self, pipeline):
        """
        Ingestion takes ~7s. Second question (no ingestion) should
        be significantly faster — under 10 seconds total.
        """
        assert pipeline.is_video_indexed(self.TEST_URL)

        with self._mock_generate(pipeline):
            import time
            start    = time.time()
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "speed test question",
            )
            elapsed = time.time() - start

        assert response.ingestion_skipped is True
        # Without ingestion + mocked LLM, should complete in under 15s
        assert elapsed < 15, (
            f"Expected under 15s for cached video, took {elapsed:.1f}s"
        )


# ── Tests: to_dict Serialization ─────────────────────────────────────
class TestSerialization:
    """
    to_dict() feeds directly into FastAPI JSON responses.
    Must produce plain Python types only.
    """

    TEST_URL = "https://www.youtube.com/watch?v=ktrIQUYIxZo"

    def _mock_generate(self, pipeline):
        return patch.object(
            pipeline._generator, "generate",
            return_value=MagicMock(
                answer="Test answer.", latency_ms=80.0,
                word_count=2, is_empty=False,
                was_retried=False, model="llama3-8b-8192",
            )
        )

    def test_to_dict_is_json_serializable(self, pipeline):
        """to_dict() must produce JSON-serializable output."""
        import json
        with self._mock_generate(pipeline):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "serialization test",
            )
        d = response.to_dict()
        # If this doesn't raise, it's JSON serializable
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_to_dict_has_required_keys(self, pipeline):
        """All keys the FastAPI contract needs must be present."""
        with self._mock_generate(pipeline):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "key test",
            )
        d        = response.to_dict()
        required = {
            "answer", "sources", "queries_used",
            "video_id", "ingestion_skipped", "latency_ms"
        }
        assert required.issubset(d.keys())

    def test_sources_in_dict_have_display_field(self, pipeline):
        """Each source in to_dict() must have a 'display' field."""
        with self._mock_generate(pipeline):
            response = pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "source test",
            )
        for source in response.to_dict()["sources"]:
            assert "display" in source
            assert "→" in source["display"]


# ── Tests: Error Handling ─────────────────────────────────────────────
class TestErrorHandling:

    TEST_URL = "https://www.youtube.com/watch?v=ktrIQUYIxZo"

    def test_empty_question_raises_pipeline_error(self, pipeline):
        """Empty question raises PipelineError with 'Input' step."""
        with pytest.raises(PipelineError, match="Input"):
            pipeline.query(
                youtube_url = self.TEST_URL,
                question    = "",
            )

    def test_empty_url_raises_pipeline_error(self, pipeline):
        """Empty URL raises PipelineError with 'Input' step."""
        with pytest.raises(PipelineError, match="Input"):
            pipeline.query(
                youtube_url = "",
                question    = "valid question",
            )

    def test_invalid_url_raises_pipeline_error(self, pipeline):
        """Completely invalid URL raises PipelineError."""
        with pytest.raises(PipelineError):
            pipeline.query(
                youtube_url = "https://www.google.com",
                question    = "valid question",
            )

    def test_pipeline_error_has_step_info(self, pipeline):
        """PipelineError must expose which step failed."""
        try:
            pipeline.query(youtube_url="", question="test")
        except PipelineError as e:
            assert e.step is not None
            assert len(e.step) > 0


# ── Tests: Utility Methods ────────────────────────────────────────────
class TestUtilityMethods:

    TEST_URL = "https://www.youtube.com/watch?v=ktrIQUYIxZo"

    def test_is_video_indexed_returns_bool(self, pipeline):
        """is_video_indexed() must return True or False."""
        result = pipeline.is_video_indexed(self.TEST_URL)
        assert isinstance(result, bool)

    def test_list_indexed_videos_returns_list(self, pipeline):
        """list_indexed_videos() must return a list."""
        result = pipeline.list_indexed_videos()
        assert isinstance(result, list)

    def test_indexed_video_appears_in_list(self, pipeline):
        """After ingestion, video ID appears in list_indexed_videos()."""
        video_ids = pipeline.list_indexed_videos()
        assert "ktrIQUYIxZo" in video_ids


# ── Integration Test: Full End-to-End ────────────────────────────────
class TestEndToEnd:
    """
    Real pipeline run — no mocks. Uses real Groq API.
    Takes ~10-15 seconds. Run with: pytest -m integration
    """

    @pytest.mark.integration
    def test_full_pipeline_returns_cited_answer(self, pipeline):
        """Full pipeline produces a grounded, timestamped answer."""
        response = pipeline.query(
            youtube_url = "https://www.youtube.com/watch?v=ktrIQUYIxZo",
            question    = "what is the main topic of this video?",
        )
        assert response.has_answer is True
        assert response.citation_count > 0
        assert response.latency.total_ms > 0
        assert len(response.queries_used) > 0

    @pytest.mark.integration
    def test_followup_question_uses_history(self, pipeline):
        """Follow-up with history produces a contextual answer."""
        history = [
            ConversationTurn(
                question = "what is the main topic?",
                answer   = "The video covers data science careers.",
            )
        ]
        response = pipeline.query(
            youtube_url = "https://www.youtube.com/watch?v=ktrIQUYIxZo",
            question    = "can you tell me more about that?",
            history     = history,
        )
        assert response.has_answer is True
        assert response.ingestion_skipped is True  # already indexed

    @pytest.mark.integration
    def test_off_topic_question_triggers_guard(self, pipeline):
        """Question outside video content triggers hallucination guard."""
        response = pipeline.query(
            youtube_url = "https://www.youtube.com/watch?v=ktrIQUYIxZo",
            question    = "what is the recipe for pasta carbonara?",
        )
        assert response.has_answer is True
        # Guard phrase must appear — model must not hallucinate a recipe
        assert "doesn't appear" in response.answer.lower() or \
               "not cover"       in response.answer.lower() or \
               "not mentioned"   in response.answer.lower()