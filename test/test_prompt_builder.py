"""
tests/test_prompt_builder.py
────────────────────────────
Tests for PromptBuilder.

Key insight: PromptBuilder produces strings — strings are easy
to test without any LLM calls. We verify:
  - Structure: correct sections present in correct order
  - Timestamps: every chunk's timestamp appears in context
  - History: included when provided, absent when not
  - Contract: to_messages() returns correct format for API
  - Guards: empty inputs raise errors with clear messages
"""

import pytest
from src.generation.prompt_builder import (
    PromptBuilder,
    PromptPackage,
    PromptBuildError,
    SYSTEM_PROMPT,
)
from src.retrieval.reranker import RankedResult
from src.retrieval.retriever import HybridSearchResult
from src.retrieval.query_transformer import ConversationTurn
from src.storage.vector_store import SearchResult


# ── Helpers ────────────────────────────────────────────────────────────
def make_ranked_result(
    chunk_id  : str,
    text      : str,
    start_time: str,
    end_time  : str,
    rank      : int,
    score     : float = 5.0,
) -> RankedResult:
    """Build a minimal RankedResult for testing."""
    sr = SearchResult(
        chunk_id    = chunk_id,
        text        = text,
        score       = 0.8,
        start_time  = start_time,
        end_time    = end_time,
        start_sec   = float(rank * 60),
        video_id    = "testvid",
        youtube_link= f"https://youtube.com/watch?v=testvid&t={rank*60}s",
        chunk_index = rank - 1,
    )
    hs = HybridSearchResult(
        search_result = sr,
        rrf_score     = 0.05,
        sources       = ["semantic_q1"],
    )
    return RankedResult(
        hybrid_result = hs,
        rerank_score  = score,
        rank          = rank,
    )


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def three_results() -> list[RankedResult]:
    """Three realistic RankedResults covering different timestamps."""
    return [
        make_ranked_result(
            "chunk_7", "Gradient descent minimizes the loss function "
            "by iteratively adjusting model weights in small steps.",
            "5:15", "6:17", rank=1, score=8.2,
        ),
        make_ranked_result(
            "chunk_9", "The learning rate controls how large each "
            "gradient descent step is during weight updates.",
            "8:42", "9:30", rank=2, score=7.1,
        ),
        make_ranked_result(
            "chunk_4", "Backpropagation computes the gradient of the "
            "loss with respect to every weight in the network.",
            "3:05", "4:12", rank=3, score=6.3,
        ),
    ]


@pytest.fixture
def conversation_history() -> list[ConversationTurn]:
    return [
        ConversationTurn(
            question = "what is the main topic of this video?",
            answer   = "The video covers neural networks and how "
                       "they learn through gradient descent.",
        ),
        ConversationTurn(
            question = "how long is the video?",
            answer   = "The video is approximately 18 minutes long.",
        ),
    ]


# ── Tests: Basic Build ────────────────────────────────────────────────
class TestBasicBuild:

    def test_returns_prompt_package(self, builder, three_results):
        """build() must return a PromptPackage object."""
        package = builder.build(
            question = "how does gradient descent work?",
            results  = three_results,
        )
        assert isinstance(package, PromptPackage)

    def test_system_prompt_is_correct(self, builder, three_results):
        """System prompt must be the canonical SYSTEM_PROMPT constant."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        assert package.system_prompt == SYSTEM_PROMPT

    def test_question_in_user_prompt(self, builder, three_results):
        """The user's question must appear in the user prompt."""
        question = "how does gradient descent work?"
        package  = builder.build(question=question, results=three_results)
        assert question in package.user_prompt

    def test_answer_marker_present(self, builder, three_results):
        """'ANSWER:' marker must appear — tells LLM where to write."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        assert "ANSWER:" in package.user_prompt

    def test_context_section_present(self, builder, three_results):
        """'CONTEXT FROM VIDEO' section header must be present."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        assert "CONTEXT FROM VIDEO" in package.user_prompt

    def test_question_section_present(self, builder, three_results):
        """'QUESTION:' label must appear before the question text."""
        package = builder.build(
            question = "how does backpropagation work?",
            results  = three_results,
        )
        assert "QUESTION:" in package.user_prompt

    def test_context_before_question(self, builder, three_results):
        """
        Context block must appear BEFORE the question in user prompt.
        LLM reads top-to-bottom — context loaded before question
        produces more grounded answers.
        """
        package = builder.build(
            question = "what is gradient descent?",
            results  = three_results,
        )
        context_pos  = package.user_prompt.find("CONTEXT FROM VIDEO")
        question_pos = package.user_prompt.find("QUESTION:")
        assert context_pos < question_pos, (
            "Context must appear before the question in the prompt"
        )


# ── Tests: Timestamp Citations ────────────────────────────────────────
class TestTimestampCitations:
    """
    Timestamps are our killer feature.
    Every chunk's start and end time must appear in the prompt
    so the LLM can write natural citations like 'at [5:15]'.
    """

    def test_all_start_times_in_prompt(self, builder, three_results):
        """Every chunk's start_time must appear in the user prompt."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        for result in three_results:
            assert result.start_time in package.user_prompt, (
                f"Start time {result.start_time} not found in prompt"
            )

    def test_all_end_times_in_prompt(self, builder, three_results):
        """Every chunk's end_time must appear in the user prompt."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        for result in three_results:
            assert result.end_time in package.user_prompt, (
                f"End time {result.end_time} not found in prompt"
            )

    def test_segment_headers_present(self, builder, three_results):
        """Segment headers like '[Segment 1 |' must appear."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        for i in range(1, len(three_results) + 1):
            assert f"[Segment {i} |" in package.user_prompt, (
                f"Segment {i} header not found in prompt"
            )

    def test_all_chunk_texts_in_prompt(self, builder, three_results):
        """Every chunk's actual text must appear in the prompt."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        for result in three_results:
            # Check first 30 chars — enough to confirm presence
            assert result.text[:30] in package.user_prompt, (
                f"Chunk text not found: '{result.text[:30]}'"
            )

    def test_segments_separated_clearly(self, builder, three_results):
        """Segments must be separated — not merged into one block."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        # Our separator contains "---"
        assert "---" in package.user_prompt


# ── Tests: Conversation History ───────────────────────────────────────
class TestConversationHistory:

    def test_no_history_produces_clean_prompt(
        self, builder, three_results
    ):
        """Without history, no 'CONVERSATION HISTORY' section."""
        package = builder.build(
            question = "test question",
            results  = three_results,
            history  = None,
        )
        assert "CONVERSATION HISTORY" not in package.user_prompt
        assert package.has_history is False

    def test_history_included_in_prompt(
        self, builder, three_results, conversation_history
    ):
        """With history, 'CONVERSATION HISTORY' section appears."""
        package = builder.build(
            question = "can you explain that more simply?",
            results  = three_results,
            history  = conversation_history,
        )
        assert "CONVERSATION HISTORY" in package.user_prompt
        assert package.has_history is True

    def test_history_questions_in_prompt(
        self, builder, three_results, conversation_history
    ):
        """Previous user questions appear in the history block."""
        package = builder.build(
            question = "follow up question",
            results  = three_results,
            history  = conversation_history,
        )
        for turn in conversation_history:
            assert turn.question in package.user_prompt

    def test_empty_history_list_treated_as_no_history(
        self, builder, three_results
    ):
        """Empty list [] behaves the same as None for history."""
        package = builder.build(
            question = "test question",
            results  = three_results,
            history  = [],
        )
        assert "CONVERSATION HISTORY" not in package.user_prompt
        assert package.has_history is False


# ── Tests: PromptPackage Properties ──────────────────────────────────
class TestPromptPackageProperties:

    def test_context_chunks_count_correct(
        self, builder, three_results
    ):
        """context_chunks must equal the number of results passed in."""
        package = builder.build(
            question = "test",
            results  = three_results,
        )
        assert package.context_chunks == len(three_results)

    def test_estimated_tokens_is_positive(
        self, builder, three_results
    ):
        """Token estimate must be a positive integer."""
        package = builder.build(
            question = "test question",
            results  = three_results,
        )
        assert isinstance(package.estimated_tokens, int)
        assert package.estimated_tokens > 0

    def test_normal_prompt_within_token_limit(
        self, builder, three_results
    ):
        """A typical 3-chunk prompt must be within the 6000 token limit."""
        package = builder.build(
            question = "how does gradient descent work?",
            results  = three_results,
        )
        assert package.is_within_token_limit is True


# ── Tests: to_messages() Contract ────────────────────────────────────
class TestToMessages:
    """
    to_messages() is the direct input to the Groq API.
    The format must be exact — wrong format = API error.
    """

    def test_returns_list_of_two_dicts(self, builder, three_results):
        """to_messages() must return exactly two message dicts."""
        package  = builder.build(question="test", results=three_results)
        messages = package.to_messages()
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert all(isinstance(m, dict) for m in messages)

    def test_first_message_is_system(self, builder, three_results):
        """First message must have role='system'."""
        package  = builder.build(question="test", results=three_results)
        messages = package.to_messages()
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self, builder, three_results):
        """Second message must have role='user'."""
        package  = builder.build(question="test", results=three_results)
        messages = package.to_messages()
        assert messages[1]["role"] == "user"

    def test_both_messages_have_content(self, builder, three_results):
        """Both messages must have non-empty 'content' field."""
        package  = builder.build(question="test", results=three_results)
        messages = package.to_messages()
        for msg in messages:
            assert "content" in msg
            assert len(msg["content"]) > 0

    def test_system_content_is_system_prompt(
        self, builder, three_results
    ):
        """System message content must be the SYSTEM_PROMPT constant."""
        package  = builder.build(question="test", results=three_results)
        messages = package.to_messages()
        assert messages[0]["content"] == SYSTEM_PROMPT


# ── Tests: Hallucination Guard ────────────────────────────────────────
class TestHallucinationGuard:
    """
    The hallucination guard is the most critical safety instruction.
    Verify it exists and is clearly worded.
    """

    def test_hallucination_guard_in_system_prompt(self):
        """System prompt must contain the refusal instruction."""
        assert "doesn't appear to cover" in SYSTEM_PROMPT

    def test_answer_only_from_context_instruction(self):
        """System prompt must explicitly restrict to context only."""
        assert "ONLY" in SYSTEM_PROMPT

    def test_citation_instruction_present(self):
        """System prompt must instruct the model to cite timestamps."""
        assert "timestamp" in SYSTEM_PROMPT.lower() or \
               "cite" in SYSTEM_PROMPT.lower()


# ── Tests: Error Handling ─────────────────────────────────────────────
class TestErrorHandling:

    def test_empty_question_raises_error(self, builder, three_results):
        """Empty question raises PromptBuildError."""
        with pytest.raises(PromptBuildError, match="empty"):
            builder.build(question="", results=three_results)

    def test_whitespace_question_raises_error(
        self, builder, three_results
    ):
        """Whitespace question raises PromptBuildError."""
        with pytest.raises(PromptBuildError, match="empty"):
            builder.build(question="   ", results=three_results)

    def test_empty_results_raises_error(self, builder):
        """Empty results list raises PromptBuildError."""
        with pytest.raises(PromptBuildError, match="empty"):
            builder.build(question="valid question", results=[])