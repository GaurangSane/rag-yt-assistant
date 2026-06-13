"""
tests/test_reranker.py
──────────────────────
Tests for CrossEncoderReranker.

The key challenge: reranker scores are raw logits with no fixed
range. We cannot assert "score > 0.5" because that's meaningless.
Instead we test what actually matters: RELATIVE ORDER.

Test strategy:
  - Singleton tests: model loads exactly once
  - Order tests:     more relevant chunk must rank higher
  - Structure tests: RankedResult has correct fields and types
  - to_prompt_dict:  PromptBuilder contract satisfied
  - Error tests:     empty input fails loudly
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.retrieval.reranker import (
    CrossEncoderReranker,
    RankedResult,
    RerankerError,
)
from src.retrieval.retriever import HybridSearchResult
from src.storage.vector_store import SearchResult


# ── Test Helpers ──────────────────────────────────────────────────────
def make_hybrid_result(
    chunk_id  : str,
    text      : str,
    start_time: str = "0:00",
    rrf_score : float = 0.05,
) -> HybridSearchResult:
    """
    Build a minimal HybridSearchResult for testing.
    Only sets the fields the reranker actually uses.
    """
    sr = SearchResult(
        chunk_id    = chunk_id,
        text        = text,
        score       = 0.8,
        start_time  = start_time,
        end_time    = "1:00",
        start_sec   = 0.0,
        video_id    = "testvid",
        youtube_link= f"https://youtube.com/watch?v=testvid&t=0s",
        chunk_index = 0,
    )
    return HybridSearchResult(
        search_result = sr,
        rrf_score     = rrf_score,
        sources       = ["semantic_q1"],
    )


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    """
    Shared reranker — model loaded once for the entire test module.
    Cross-encoder is ~80MB — loading once keeps tests fast.
    """
    return CrossEncoderReranker()


@pytest.fixture(scope="module")
def ml_candidates() -> list[HybridSearchResult]:
    """
    Five candidates on different topics.
    Two are about ML training, three are unrelated.
    Used to verify the reranker surfaces the right ones.
    """
    return [
        make_hybrid_result(
            "chunk_0",
            "Gradient descent iteratively adjusts model weights "
            "to minimize the loss function during training",
            "0:00",
        ),
        make_hybrid_result(
            "chunk_1",
            "The recipe requires mixing flour butter and sugar "
            "then baking at 180 degrees for 30 minutes",
            "1:00",
        ),
        make_hybrid_result(
            "chunk_2",
            "Backpropagation computes gradients by applying the "
            "chain rule backwards through the network layers",
            "2:00",
        ),
        make_hybrid_result(
            "chunk_3",
            "Renaissance painters used oil paint on canvas and "
            "developed perspective techniques in the 15th century",
            "3:00",
        ),
        make_hybrid_result(
            "chunk_4",
            "The learning rate controls how large each gradient "
            "descent step is when updating neural network weights",
            "4:00",
        ),
    ]


# ── Tests: Singleton ──────────────────────────────────────────────────
class TestSingleton:

    def test_same_instance_returned(self):
        """Two CrossEncoderReranker() calls return the same object."""
        a = CrossEncoderReranker()
        b = CrossEncoderReranker()
        assert a is b

    def test_model_is_loaded(self, reranker):
        """The underlying CrossEncoder must not be None."""
        assert reranker.model is not None

    def test_singleton_consistent_with_fixture(self, reranker):
        """A fresh instantiation returns the same object as the fixture."""
        fresh = CrossEncoderReranker()
        assert fresh is reranker


# ── Tests: Output Structure ───────────────────────────────────────────
class TestOutputStructure:
    """Verify the shape and types of RankedResult objects."""

    def test_returns_list_of_ranked_results(
        self, reranker, ml_candidates
    ):
        """rerank() must return a list of RankedResult objects."""
        results = reranker.rerank(
            question   = "how does gradient descent work?",
            candidates = ml_candidates,
            top_k      = 3,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, RankedResult) for r in results)

    def test_returns_exactly_top_k(self, reranker, ml_candidates):
        """Exactly top_k results are returned — no more, no less."""
        for k in [1, 2, 3]:
            results = reranker.rerank(
                question   = "neural network training",
                candidates = ml_candidates,
                top_k      = k,
            )
            assert len(results) == k, (
                f"Expected {k} results, got {len(results)}"
            )

    def test_ranks_are_one_indexed(self, reranker, ml_candidates):
        """Ranks must start at 1, not 0. Best chunk has rank=1."""
        results = reranker.rerank(
            question   = "gradient descent optimization",
            candidates = ml_candidates,
            top_k      = 3,
        )
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_scores_are_floats(self, reranker, ml_candidates):
        """rerank_score must be a Python float."""
        results = reranker.rerank(
            question   = "backpropagation chain rule",
            candidates = ml_candidates,
            top_k      = 3,
        )
        for r in results:
            assert isinstance(r.rerank_score, float)

    def test_pass_through_properties_accessible(
        self, reranker, ml_candidates
    ):
        """
        All pass-through properties must work without chaining.
        result.text must work — not result.hybrid_result.text
        """
        results = reranker.rerank(
            question   = "learning rate weight updates",
            candidates = ml_candidates,
            top_k      = 1,
        )
        r = results[0]

        assert isinstance(r.chunk_id,    str) and r.chunk_id    != ""
        assert isinstance(r.text,        str) and r.text        != ""
        assert isinstance(r.start_time,  str)
        assert isinstance(r.end_time,    str)
        assert isinstance(r.start_sec,   float)
        assert isinstance(r.video_id,    str)
        assert isinstance(r.youtube_link,str)
        assert isinstance(r.rrf_score,   float)


# ── Tests: Ordering — The Most Important Tests ────────────────────────
class TestOrdering:
    """
    Verify the reranker correctly orders by relevance.
    We cannot test absolute score values (logits are unbounded).
    We CAN test that more relevant chunks rank higher.
    """

    def test_scores_sorted_descending(self, reranker, ml_candidates):
        """
        Result list must be sorted highest score first.
        Rank 1 must always have the highest rerank_score.
        """
        results = reranker.rerank(
            question   = "how do neural networks learn?",
            candidates = ml_candidates,
            top_k      = 3,
        )
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results are not sorted by score descending"
        )

    def test_rank_1_score_highest(self, reranker, ml_candidates):
        """Rank 1 result has the highest score among all results."""
        results = reranker.rerank(
            question   = "gradient descent weight update mechanism",
            candidates = ml_candidates,
            top_k      = 3,
        )
        best_score = results[0].rerank_score
        for r in results[1:]:
            assert best_score >= r.rerank_score

    def test_ml_chunks_rank_above_food_chunk(
        self, reranker, ml_candidates
    ):
        """
        For an ML question, ML chunks must all score higher
        than the cooking recipe chunk.

        This is the core semantic test — proves the cross-encoder
        is doing real relevance scoring, not just returning random order.
        """
        results = reranker.rerank(
            question   = "how does backpropagation compute gradients?",
            candidates = ml_candidates,
            top_k      = 5,   # keep all to check food chunk position
        )

        # Find positions of ML chunks vs food chunk
        food_chunk_rank = next(
            r.rank for r in results if "recipe" in r.text.lower()
        )
        ml_ranks = [
            r.rank for r in results
            if any(w in r.text.lower()
                   for w in ["gradient", "backpropagation", "learning rate"])
        ]

        # Every ML chunk must rank better (lower number) than food chunk
        for ml_rank in ml_ranks:
            assert ml_rank < food_chunk_rank, (
                f"ML chunk at rank {ml_rank} should beat "
                f"food chunk at rank {food_chunk_rank}"
            )

    def test_order_can_differ_from_rrf_order(
        self, reranker, ml_candidates
    ):
        """
        Reranker may produce different order than RRF.
        This proves it adds genuine value beyond retrieval.

        We give candidates in a fixed RRF order and check
        whether the reranker ever changes it — it should
        at least occasionally.
        """
        question = "how does backpropagation use the chain rule?"

        # rrf_scores are fixed in ml_candidates (all 0.05)
        # reranker assigns its own scores — will order differently
        rrf_order    = [c.chunk_id for c in ml_candidates]
        results      = reranker.rerank(
            question   = question,
            candidates = ml_candidates,
            top_k      = 5,
        )
        rerank_order = [r.chunk_id for r in results]

        # The two orderings should differ for a specific question
        assert rrf_order != rerank_order, (
            "Reranker produced identical order to RRF — "
            "it may not be adding value"
        )


# ── Tests: to_prompt_dict ─────────────────────────────────────────────
class TestToPromptDict:
    """
    to_prompt_dict() is the contract between Reranker and PromptBuilder.
    PromptBuilder depends on exact keys being present.
    """

    def test_has_all_required_keys(self, reranker, ml_candidates):
        """All keys PromptBuilder expects must be present."""
        results  = reranker.rerank(
            question   = "neural network training",
            candidates = ml_candidates,
            top_k      = 1,
        )
        d        = results[0].to_prompt_dict()
        required = {
            "rank", "text", "start_time",
            "end_time", "youtube_link",
            "chunk_id", "rerank_score",
        }
        assert required.issubset(d.keys()), (
            f"Missing keys: {required - d.keys()}"
        )

    def test_rank_in_dict_matches_result_rank(
        self, reranker, ml_candidates
    ):
        """Dict rank must match the RankedResult.rank attribute."""
        results = reranker.rerank(
            question   = "gradient descent",
            candidates = ml_candidates,
            top_k      = 3,
        )
        for r in results:
            assert r.to_prompt_dict()["rank"] == r.rank

    def test_text_in_dict_is_non_empty(self, reranker, ml_candidates):
        """Text field in prompt dict must never be empty."""
        results = reranker.rerank(
            question   = "learning rate",
            candidates = ml_candidates,
            top_k      = 3,
        )
        for r in results:
            assert r.to_prompt_dict()["text"].strip() != ""


# ── Tests: Edge Cases ─────────────────────────────────────────────────
class TestEdgeCases:

    def test_single_candidate_returns_one_result(
        self, reranker
    ):
        """One candidate in → one result out with rank=1."""
        single = [make_hybrid_result(
            "only_chunk",
            "The only chunk available for reranking",
        )]
        results = reranker.rerank(
            question   = "some question",
            candidates = single,
            top_k      = 3,   # top_k > candidates — should return 1
        )
        assert len(results) == 1
        assert results[0].rank == 1

    def test_top_k_larger_than_candidates(self, reranker):
        """
        top_k=10 with 3 candidates returns 3, not 10.
        Never crashes, never pads with empty results.
        """
        three = [
            make_hybrid_result(f"chunk_{i}", f"some text {i}")
            for i in range(3)
        ]
        results = reranker.rerank(
            question   = "test question",
            candidates = three,
            top_k      = 10,
        )
        assert len(results) == 3


# ── Tests: Error Handling ─────────────────────────────────────────────
class TestErrorHandling:

    def test_empty_candidates_raises_error(self, reranker):
        """Empty candidates list raises RerankerError."""
        with pytest.raises(RerankerError, match="empty"):
            reranker.rerank(
                question   = "test question",
                candidates = [],
            )

    def test_empty_question_raises_error(self, reranker, ml_candidates):
        """Empty question raises RerankerError."""
        with pytest.raises(RerankerError, match="empty"):
            reranker.rerank(
                question   = "",
                candidates = ml_candidates,
            )

    def test_whitespace_question_raises_error(
        self, reranker, ml_candidates
    ):
        """Whitespace-only question raises RerankerError."""
        with pytest.raises(RerankerError, match="empty"):
            reranker.rerank(
                question   = "   ",
                candidates = ml_candidates,
            )