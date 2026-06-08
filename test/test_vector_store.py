"""
tests/test_vector_store.py
──────────────────────────
Tests for the VectorStore repository.

Key challenge: these tests write to a real ChromaDB database.
We solve this with a temporary directory fixture — each test
run gets a fresh, isolated database that is deleted afterwards.
This means tests never pollute your real chroma_db/ folder.

Test strategy:
  - Fixtures build real EmbeddedChunks using the real EmbeddingModel
  - A tmp_path database isolates tests from production data
  - Every public method tested: save, search, exists, delete, count
  - Idempotency verified explicitly
"""

import pytest
import numpy as np
from src.storage.vector_store import (
    VectorStore,
    SearchResult,
    VectorStoreError,
    CollectionNotFoundError,
)
from src.ingestion.chunker  import Chunk
from src.ingestion.embedder import EmbeddingModel, EmbeddedChunk


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    """Shared embedding model — loaded once for all tests."""
    return EmbeddingModel()


@pytest.fixture(scope="module")
def sample_embedded_chunks(model) -> list[EmbeddedChunk]:
    """
    Real EmbeddedChunks built from real embeddings.
    scope="module" — built once, shared across all tests.
    """
    chunks = [
        Chunk(
            chunk_id    = f"testvid_chunk_{i}",
            text        = text,
            start_time  = f"{i}:00",
            end_time    = f"{i}:59",
            start_sec   = float(i * 60),
            video_id    = "testvid",
            chunk_index = i,
        )
        for i, text in enumerate([
            "Neural networks learn through backpropagation and gradient descent",
            "The loss function measures how wrong the model predictions are",
            "Attention mechanisms allow transformers to focus on relevant tokens",
            "Overfitting occurs when a model memorizes training data too closely",
            "Dropout randomly disables neurons during training to prevent overfitting",
        ])
    ]
    return model.embed(chunks)


@pytest.fixture
def store(tmp_path) -> VectorStore:
    """
    Fresh VectorStore pointing to a temporary directory.

    tmp_path is a built-in pytest fixture that provides a
    unique temporary directory for each test function.
    The directory is automatically deleted after the test.

    This is the correct way to test database code:
      - Real database operations (not mocked)
      - Isolated from production data
      - Clean slate for every test
    """
    return VectorStore(persist_dir=str(tmp_path))


# ── Tests: Save Operation ─────────────────────────────────────────────
class TestSave:
    """Tests for VectorStore.save() — the write operation."""

    def test_save_returns_correct_count(self, store, sample_embedded_chunks):
        """save() must return how many chunks were stored."""
        count = store.save(sample_embedded_chunks, video_id="testvid")
        assert count == len(sample_embedded_chunks)

    def test_save_makes_video_exist(self, store, sample_embedded_chunks):
        """After save(), exists() must return True."""
        store.save(sample_embedded_chunks, video_id="testvid")
        assert store.exists("testvid") is True

    def test_save_is_idempotent(self, store, sample_embedded_chunks):
        """
        Saving the same video twice must not raise an error.
        Second save replaces first — count remains the same.
        This is the idempotency guarantee.
        """
        store.save(sample_embedded_chunks, video_id="testvid")
        store.save(sample_embedded_chunks, video_id="testvid")  # must not raise
        assert store.count("testvid") == len(sample_embedded_chunks)

    def test_save_no_overwrite_raises_on_duplicate(
        self, store, sample_embedded_chunks
    ):
        """overwrite=False must raise if video already exists."""
        store.save(sample_embedded_chunks, video_id="testvid")
        with pytest.raises(VectorStoreError, match="already exists"):
            store.save(
                sample_embedded_chunks,
                video_id  = "testvid",
                overwrite = False,
            )

    def test_save_empty_chunks_raises_error(self, store):
        """Empty chunks list must raise VectorStoreError."""
        with pytest.raises(VectorStoreError, match="empty"):
            store.save([], video_id="testvid")

    def test_save_multiple_videos_isolated(
        self, store, sample_embedded_chunks
    ):
        """Two videos stored separately must not interfere."""
        store.save(sample_embedded_chunks, video_id="video_a")
        store.save(sample_embedded_chunks, video_id="video_b")

        assert store.count("video_a") == len(sample_embedded_chunks)
        assert store.count("video_b") == len(sample_embedded_chunks)
        assert store.exists("video_a") is True
        assert store.exists("video_b") is True


# ── Tests: Search Operation ───────────────────────────────────────────
class TestSearch:
    """Tests for VectorStore.search() — the read operation."""

    @pytest.fixture(autouse=True)
    def populate(self, store, sample_embedded_chunks):
        """Automatically save chunks before every test in this class."""
        store.save(sample_embedded_chunks, video_id="testvid")

    def test_search_returns_search_results(self, store, model):
        """search() must return a list of SearchResult objects."""
        query_vec = model.embed_query("how do neural networks learn?")
        results   = store.search(query_vec, video_id="testvid", top_k=3)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_correct_count(self, store, model):
        """search() must return exactly top_k results."""
        query_vec = model.embed_query("gradient descent optimization")
        results   = store.search(query_vec, video_id="testvid", top_k=3)
        assert len(results) == 3

    def test_search_results_sorted_by_score(self, store, model):
        """Results must be sorted best-first (highest score first)."""
        query_vec = model.embed_query("neural network training")
        results   = store.search(query_vec, video_id="testvid", top_k=5)
        scores    = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_in_valid_range(self, store, model):
        """All similarity scores must be between -1.0 and 1.0."""
        query_vec = model.embed_query("loss function")
        results   = store.search(query_vec, video_id="testvid", top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0, (
                f"Score {r.score} out of valid range [-1, 1]"
            )

    def test_search_result_has_all_fields(self, store, model):
        """Every SearchResult must have all required fields populated."""
        query_vec = model.embed_query("transformers attention")
        results   = store.search(query_vec, video_id="testvid", top_k=1)
        r         = results[0]

        assert r.chunk_id    != ""
        assert r.text        != ""
        assert r.start_time  != ""
        assert r.end_time    != ""
        assert r.video_id    == "testvid"
        assert r.start_sec   >= 0.0
        assert r.chunk_index >= 0

    def test_search_returns_semantically_relevant_result(
        self, store, model
    ):
        """
        The top result for a specific query must be semantically
        related — not a random chunk.

        Query about "overfitting" should rank the overfitting chunk
        higher than the neural network chunk.
        """
        query_vec = model.embed_query(
            "overfitting memorizing training data problem"
        )
        results = store.search(query_vec, video_id="testvid", top_k=5)

        # The top result should mention overfitting
        top_text = results[0].text.lower()
        assert "overfitting" in top_text or "memorize" in top_text, (
            f"Expected overfitting-related chunk at top, "
            f"got: '{results[0].text[:80]}'"
        )

    def test_search_unknown_video_raises_error(self, store, model):
        """Searching a video that was never ingested raises correct error."""
        query_vec = model.embed_query("some query")
        with pytest.raises(CollectionNotFoundError):
            store.search(query_vec, video_id="never_ingested_video")


# ── Tests: Utility Operations ─────────────────────────────────────────
class TestUtilityOperations:

    def test_exists_false_before_save(self, store):
        """exists() returns False for a video that was never saved."""
        assert store.exists("nonexistent") is False

    def test_exists_true_after_save(self, store, sample_embedded_chunks):
        """exists() returns True immediately after save()."""
        store.save(sample_embedded_chunks, video_id="testvid")
        assert store.exists("testvid") is True

    def test_count_matches_saved_chunks(self, store, sample_embedded_chunks):
        """count() must equal the number of chunks saved."""
        store.save(sample_embedded_chunks, video_id="testvid")
        assert store.count("testvid") == len(sample_embedded_chunks)

    def test_count_unknown_video_raises_error(self, store):
        """count() on unsaved video raises CollectionNotFoundError."""
        with pytest.raises(CollectionNotFoundError):
            store.count("no_such_video")

    def test_delete_removes_video(self, store, sample_embedded_chunks):
        """After delete(), exists() must return False."""
        store.save(sample_embedded_chunks, video_id="testvid")
        store.delete("testvid")
        assert store.exists("testvid") is False

    def test_delete_unknown_video_raises_error(self, store):
        """Deleting a video that doesn't exist raises correct error."""
        with pytest.raises(CollectionNotFoundError):
            store.delete("ghost_video")

    def test_list_videos_empty_initially(self, store):
        """Fresh store has no videos."""
        assert store.list_videos() == []

    def test_list_videos_after_save(self, store, sample_embedded_chunks):
        """list_videos() shows video IDs without the 'yt_' prefix."""
        store.save(sample_embedded_chunks, video_id="video_one")
        store.save(sample_embedded_chunks, video_id="video_two")
        videos = store.list_videos()
        assert "video_one" in videos
        assert "video_two" in videos
        # Verify prefix was stripped
        assert "yt_video_one" not in videos

    def test_get_all_chunks_returns_all(self, store, sample_embedded_chunks):
        """get_all_chunks() must return all stored chunks."""
        store.save(sample_embedded_chunks, video_id="testvid")
        all_chunks = store.get_all_chunks("testvid")
        assert len(all_chunks) == len(sample_embedded_chunks)

    def test_get_all_chunks_have_required_keys(
        self, store, sample_embedded_chunks
    ):
        """Every chunk dict from get_all_chunks() must have these keys."""
        store.save(sample_embedded_chunks, video_id="testvid")
        all_chunks = store.get_all_chunks("testvid")
        required   = {
            "chunk_id", "text", "start_time",
            "end_time", "start_sec", "video_id"
        }
        for chunk in all_chunks:
            assert required.issubset(chunk.keys())