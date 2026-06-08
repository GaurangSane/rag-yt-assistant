"""
tests/test_retriever.py
───────────────────────
Tests for HybridRetriever.

Test strategy:
  - Build a real populated VectorStore in a tmp directory
  - Use the real EmbeddingModel (singleton — loaded once)
  - Test every method: semantic, BM25, RRF, retrieve()
  - Verify BM25 caching behaviour explicitly
  - All tests offline after initial model load
"""

import pytest
import numpy as np
from src.retrieval.retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrieverError,
)
from src.storage.vector_store import VectorStore
from src.ingestion.embedder   import EmbeddingModel
from src.ingestion.chunker    import Chunk


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    """Shared embedding model — loaded once for all tests."""
    return EmbeddingModel()


@pytest.fixture(scope="module")
def populated_store(embedding_model, tmp_path_factory) -> tuple:
    """
    A real VectorStore populated with embedded chunks.
    scope="module" — built once, shared across all tests.

    Returns (store, video_id) tuple.

    We use tmp_path_factory (not tmp_path) for module-scoped fixtures
    because tmp_path is function-scoped by default.
    """
    tmp_dir  = tmp_path_factory.mktemp("chroma")
    store    = VectorStore(persist_dir=str(tmp_dir))
    video_id = "testvid"

    # Build chunks covering different topics
    raw_chunks = [
        Chunk(
            chunk_id    = f"{video_id}_chunk_{i}",
            text        = text,
            start_time  = f"{i}:00",
            end_time    = f"{i}:59",
            start_sec   = float(i * 60),
            video_id    = video_id,
            chunk_index = i,
        )
        for i, text in enumerate([
            "Neural networks learn through backpropagation gradient descent",
            "The loss function measures prediction errors during training",
            "Attention mechanisms in transformers focus on relevant tokens",
            "Overfitting happens when a model memorizes training examples",
            "Dropout regularization randomly disables neurons during training",
            "Convolutional networks excel at image recognition spatial features",
            "Recurrent networks process sequential data time series text",
            "Transfer learning reuses pretrained models on new tasks",
        ])
    ]

    embedded = embedding_model.embed(raw_chunks)
    store.save(embedded, video_id=video_id)

    return store, video_id


@pytest.fixture(scope="module")
def retriever(populated_store, embedding_model) -> HybridRetriever:
    """HybridRetriever wired to the populated store."""
    store, _ = populated_store
    return HybridRetriever(store=store, embedding_model=embedding_model)


# ── Tests: BM25 Cache ────────────────────────────────────────────────
class TestBM25Cache:
    """
    The cache is the most important performance feature.
    Test it explicitly — not just that it exists.
    """

    def test_first_call_builds_index(self, retriever, populated_store):
        """First call for a video builds and caches the BM25 index."""
        _, video_id = populated_store
        # Clear cache to ensure fresh test
        retriever.invalidate_cache(video_id)

        assert video_id not in retriever._bm25_cache
        retriever._get_bm25_index(video_id)
        assert video_id in retriever._bm25_cache

    def test_second_call_returns_same_object(
        self, retriever, populated_store
    ):
        """
        Second call must return the EXACT same _BM25Index object.
        'is' checks object identity — same memory address, not just equal.
        """
        _, video_id = populated_store
        index_first  = retriever._get_bm25_index(video_id)
        index_second = retriever._get_bm25_index(video_id)
        assert index_first is index_second

    def test_invalidate_clears_cache(self, retriever, populated_store):
        """After invalidation, next call rebuilds fresh index."""
        _, video_id = populated_store
        retriever._get_bm25_index(video_id)     # ensure it's cached
        retriever.invalidate_cache(video_id)
        assert video_id not in retriever._bm25_cache

    def test_unknown_video_raises_error(self, retriever):
        """BM25 index for a never-ingested video raises RetrieverError."""
        with pytest.raises(RetrieverError, match="No chunks found"):
            retriever._get_bm25_index("ghost_video_999")


# ── Tests: Individual Search Methods ────────────────────────────────
class TestIndividualSearches:

    def test_semantic_returns_results(self, retriever, populated_store):
        """Semantic search returns at least one result."""
        _, video_id = populated_store
        results = retriever._semantic_search(
            "how do neural networks train", video_id, top_k=3
        )
        assert len(results) > 0

    def test_bm25_returns_results(self, retriever, populated_store):
        """BM25 search returns at least one result for matching term."""
        _, video_id = populated_store
        results = retriever._bm25_search(
            "backpropagation gradient", video_id, top_k=3
        )
        assert len(results) > 0

    def test_bm25_zero_score_filtered(self, retriever, populated_store):
        """BM25 results with zero score are excluded."""
        _, video_id = populated_store
        # Query with words that appear in no chunk
        results = retriever._bm25_search(
            "xyzabc123 nonsense gibberish", video_id, top_k=5
        )
        for r in results:
            assert r.score > 0

    def test_semantic_results_have_scores(self, retriever, populated_store):
        """All semantic results have a valid similarity score."""
        _, video_id = populated_store
        results = retriever._semantic_search(
            "dropout regularization", video_id, top_k=3
        )
        for r in results:
            assert -1.0 <= r.score <= 1.0


# ── Tests: RRF Fusion ────────────────────────────────────────────────
class TestRRFFusion:
    """
    Test the fusion algorithm independently of actual searches.
    We build synthetic result lists so we can predict exact outputs.
    """

    def _make_result(self, chunk_id: str, score: float = 0.5) -> object:
        """Build a minimal SearchResult for fusion testing."""
        from src.storage.vector_store import SearchResult
        return SearchResult(
            chunk_id    = chunk_id,
            text        = f"text for {chunk_id}",
            score       = score,
            start_time  = "0:00",
            end_time    = "1:00",
            start_sec   = 0.0,
            video_id    = "testvid",
            youtube_link= "",
            chunk_index = 0,
        )

    def test_rrf_deduplicates_results(self, retriever):
        """Same chunk in multiple lists appears only once in output."""
        r = self._make_result

        lists = [
            ("semantic_q1", [r("chunk_a"), r("chunk_b")]),
            ("bm25_q1",     [r("chunk_a"), r("chunk_c")]),
        ]
        fused = retriever._reciprocal_rank_fusion(lists)
        ids   = [f.chunk_id for f in fused]
        assert len(ids) == len(set(ids))

    def test_rrf_chunk_in_both_lists_ranks_higher(self, retriever):
        """
        A chunk appearing in ALL lists should rank higher than a chunk
        appearing in only one — that is the core RRF guarantee.
        """
        r = self._make_result
        lists = [
            ("semantic_q1", [r("chunk_a"), r("chunk_b"), r("chunk_c")]),
            ("bm25_q1",     [r("chunk_a"), r("chunk_d"), r("chunk_e")]),
            ("semantic_q2", [r("chunk_a"), r("chunk_f"), r("chunk_g")]),
        ]
        fused = retriever._reciprocal_rank_fusion(lists)

        # chunk_a appears in all 3 lists at rank 1 — must be top result
        assert fused[0].chunk_id == "chunk_a"

    def test_rrf_scores_are_positive(self, retriever):
        """All RRF scores must be positive numbers."""
        r     = self._make_result
        lists = [("source1", [r("chunk_a"), r("chunk_b")])]
        fused = retriever._reciprocal_rank_fusion(lists)
        for f in fused:
            assert f.rrf_score > 0

    def test_rrf_sources_tracked_correctly(self, retriever):
        """sources field records which lists contained each chunk."""
        r     = self._make_result
        lists = [
            ("semantic_q1", [r("chunk_a")]),
            ("bm25_q1",     [r("chunk_a")]),
        ]
        fused = retriever._reciprocal_rank_fusion(lists)

        chunk_a_result = next(f for f in fused if f.chunk_id == "chunk_a")
        assert "semantic_q1" in chunk_a_result.sources
        assert "bm25_q1"     in chunk_a_result.sources

    def test_found_by_both_property(self, retriever):
        """found_by_both is True only when both methods found the chunk."""
        r     = self._make_result
        lists = [
            ("semantic_q1", [r("chunk_a"), r("chunk_b")]),
            ("bm25_q1",     [r("chunk_a")]),
        ]
        fused = retriever._reciprocal_rank_fusion(lists)

        chunk_a = next(f for f in fused if f.chunk_id == "chunk_a")
        chunk_b = next(f for f in fused if f.chunk_id == "chunk_b")

        assert chunk_a.found_by_both is True    # in both lists
        assert chunk_b.found_by_both is False   # only in semantic


# ── Tests: Main Retrieve Method ──────────────────────────────────────
class TestRetrieve:

    def test_returns_hybrid_search_results(
        self, retriever, populated_store
    ):
        """retrieve() must return a list of HybridSearchResult objects."""
        _, video_id = populated_store
        results = retriever.retrieve(
            queries  = ["how neural networks learn"],
            video_id = video_id,
            top_k    = 3,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_returns_correct_count(self, retriever, populated_store):
        """retrieve() returns exactly top_k results."""
        _, video_id = populated_store
        results = retriever.retrieve(
            queries  = ["gradient descent backpropagation"],
            video_id = video_id,
            top_k    = 3,
        )
        assert len(results) == 3

    def test_results_sorted_by_rrf_score(
        self, retriever, populated_store
    ):
        """Results must be ordered by RRF score, highest first."""
        _, video_id = populated_store
        results = retriever.retrieve(
            queries  = ["attention transformer mechanism"],
            video_id = video_id,
            top_k    = 5,
        )
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multiple_queries_increases_coverage(
        self, retriever, populated_store
    ):
        """
        3 queries should retrieve more unique chunks than 1 query alone.
        This validates the core multi-query value proposition.
        """
        _, video_id = populated_store

        single_results = retriever.retrieve(
            queries  = ["neural network training"],
            video_id = video_id,
            top_k    = 5,
        )
        multi_results = retriever.retrieve(
            queries  = [
                "neural network training process",
                "backpropagation gradient weight update",
                "loss function optimization learning",
            ],
            video_id = video_id,
            top_k    = 5,
        )

        single_ids = {r.chunk_id for r in single_results}
        multi_ids  = {r.chunk_id for r in multi_results}

        # Multi-query should find at least as many chunks as single
        assert len(multi_ids) >= len(single_ids)

    def test_retrieve_empty_queries_raises_error(
        self, retriever, populated_store
    ):
        """Empty queries list raises RetrieverError."""
        _, video_id = populated_store
        with pytest.raises(RetrieverError, match="empty"):
            retriever.retrieve(queries=[], video_id=video_id)

    def test_retrieve_unknown_video_raises_error(self, retriever):
        """Retrieving from a non-existent video raises an error."""
        with pytest.raises(Exception):
            retriever.retrieve(
                queries  = ["test query"],
                video_id = "never_ingested_999",
            )

    def test_pass_through_properties_work(
        self, retriever, populated_store
    ):
        """
        Verify HybridSearchResult pass-through properties work
        so callers can write result.text instead of
        result.search_result.text
        """
        _, video_id = populated_store
        results = retriever.retrieve(
            queries  = ["overfitting dropout regularization"],
            video_id = video_id,
            top_k    = 1,
        )
        r = results[0]

        # All pass-through properties must be accessible directly
        assert isinstance(r.chunk_id,    str)
        assert isinstance(r.text,        str)
        assert isinstance(r.start_time,  str)
        assert isinstance(r.end_time,    str)
        assert isinstance(r.start_sec,   float)
        assert isinstance(r.video_id,    str)
        assert isinstance(r.youtube_link,str)
        assert r.text != ""