"""
tests/test_embedder.py
──────────────────────
Unit and integration tests for the EmbeddingModel module.

Test strategy:
  - Singleton tests: verify model loads exactly once
  - Structure tests: verify EmbeddedChunk has correct shape
  - Semantic tests:  verify similar text → similar vectors
  - Error tests:     verify empty input fails clearly
  - No network:      model loads from local cache after first run
"""

import pytest
import numpy as np
from src.ingestion.embedder import EmbeddingModel, EmbeddedChunk, EmbeddingError
from src.ingestion.chunker  import Chunk


# ── Fixtures ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    """
    Single EmbeddingModel instance shared across ALL tests in this file.

    scope="module" means pytest creates this fixture once per file,
    not once per test. This is critical for performance — we load
    the 420MB model once, not 15 times.

    Without scope="module": 15 tests × 4s load = 60s test suite
    With    scope="module": 1 load × 4s + 14 × 0s = 4s test suite
    """
    return EmbeddingModel()


@pytest.fixture(scope="module")
def sample_chunks() -> list[Chunk]:
    """Three chunks with meaningfully different content for semantic tests."""
    return [
        Chunk(
            chunk_id    = "vid_chunk_0",
            text        = "Neural networks learn by adjusting weights through backpropagation",
            start_time  = "0:00",
            end_time    = "1:00",
            start_sec   = 0.0,
            video_id    = "testvid",
            chunk_index = 0,
        ),
        Chunk(
            chunk_id    = "vid_chunk_1",
            text        = "Gradient descent minimizes the loss function iteratively",
            start_time  = "0:45",
            end_time    = "1:45",
            start_sec   = 45.0,
            video_id    = "testvid",
            chunk_index = 1,
        ),
        Chunk(
            chunk_id    = "vid_chunk_2",
            text        = "The recipe calls for flour sugar butter and eggs",
            start_time  = "1:30",
            end_time    = "2:30",
            start_sec   = 90.0,
            video_id    = "testvid",
            chunk_index = 2,
        ),
    ]


# ── Tests: Singleton Behaviour ───────────────────────────────────────
class TestSingleton:
    """
    The singleton is the core design decision of this module.
    These tests verify it works correctly.
    """

    def test_same_instance_returned(self):
        """Two EmbeddingModel() calls must return the identical object."""
        model_a = EmbeddingModel()
        model_b = EmbeddingModel()
        assert model_a is model_b, (
            "Singleton broken: two EmbeddingModel() calls returned "
            "different objects — model would be loaded twice"
        )

    def test_model_is_loaded(self, model):
        """The underlying SentenceTransformer must not be None."""
        assert model.model is not None

    def test_third_instance_same_object(self, model):
        """Singleton holds even with 3+ instantiations."""
        model_c = EmbeddingModel()
        assert model_c is model


# ── Tests: Embedding Shape and Type ─────────────────────────────────
class TestEmbeddingStructure:
    """Verify the output has exactly the right shape and types."""

    def test_returns_list_of_embedded_chunks(self, model, sample_chunks):
        """embed() returns a list of EmbeddedChunk objects."""
        result = model.embed(sample_chunks)
        assert isinstance(result, list)
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)

    def test_output_length_matches_input(self, model, sample_chunks):
        """One EmbeddedChunk out for every Chunk in."""
        result = model.embed(sample_chunks)
        assert len(result) == len(sample_chunks)

    def test_embedding_is_correct_dimension(self, model, sample_chunks):
        """all-mpnet-base-v2 produces 768-dimensional vectors."""
        result = model.embed(sample_chunks)
        for ec in result:
            assert ec.embedding_dimensions == 768, (
                f"Expected 768 dimensions, got {ec.embedding_dimensions}"
            )

    def test_embedding_is_list_of_floats(self, model, sample_chunks):
        """Embedding must be a plain Python list of floats — not numpy array."""
        result = model.embed(sample_chunks)
        for ec in result:
            assert isinstance(ec.embedding, list)
            assert all(isinstance(v, float) for v in ec.embedding)

    def test_order_preserved(self, model, sample_chunks):
        """Output chunk order must match input order exactly."""
        result = model.embed(sample_chunks)
        for original, embedded in zip(sample_chunks, result):
            assert embedded.chunk_id == original.chunk_id

    def test_original_chunk_unchanged(self, model, sample_chunks):
        """
        Immutability check: embedding must not modify the input Chunk.
        The original chunk's text and metadata must be identical after.
        """
        original_texts = [c.text for c in sample_chunks]
        model.embed(sample_chunks)
        for chunk, original_text in zip(sample_chunks, original_texts):
            assert chunk.text == original_text


# ── Tests: Normalization ─────────────────────────────────────────────
class TestNormalization:
    """
    Normalized vectors have magnitude = 1.0.
    This is critical for cosine similarity to work correctly.
    """

    def test_embeddings_are_normalized(self, model, sample_chunks):
        """
        Each embedding vector must have L2 norm ≈ 1.0.

        We use ≈ (within 0.001) because floating point arithmetic
        introduces tiny rounding errors — exact equality would be fragile.
        """
        result = model.embed(sample_chunks)
        for ec in result:
            vector = np.array(ec.embedding)
            norm   = np.linalg.norm(vector)
            assert abs(norm - 1.0) < 0.001, (
                f"Vector for chunk '{ec.chunk_id}' is not normalized. "
                f"L2 norm = {norm:.4f}, expected 1.0"
            )


# ── Tests: Semantic Correctness ──────────────────────────────────────
class TestSemanticCorrectness:
    """
    The most important tests: do similar texts get similar vectors?
    This validates that the embedding model is actually working,
    not just producing random numbers.
    """

    def test_similar_chunks_have_higher_similarity(self, model, sample_chunks):
        """
        chunk_0 (neural networks) and chunk_1 (gradient descent)
        are both about ML training — they should be more similar
        to each other than either is to chunk_2 (recipe/cooking).
        """
        result = model.embed(sample_chunks)

        vec_ml_1  = np.array(result[0].embedding)   # neural networks
        vec_ml_2  = np.array(result[1].embedding)   # gradient descent
        vec_food  = np.array(result[2].embedding)   # recipe

        # Cosine similarity (dot product of normalized vectors)
        sim_both_ml   = np.dot(vec_ml_1, vec_ml_2)
        sim_ml_food   = np.dot(vec_ml_1, vec_food)

        assert sim_both_ml > sim_ml_food, (
            f"ML chunks should be more similar to each other "
            f"({sim_both_ml:.4f}) than to food chunk ({sim_ml_food:.4f})"
        )

    def test_similarity_scores_in_valid_range(self, model, sample_chunks):
        """
        Cosine similarity of normalized vectors must be in [-1, 1].
        Scores outside this range indicate a normalization bug.
        """
        result = model.embed(sample_chunks)
        for i in range(len(result)):
            for j in range(len(result)):
                sim = np.dot(
                    np.array(result[i].embedding),
                    np.array(result[j].embedding)
                )
                assert -1.01 <= sim <= 1.01, (
                    f"Similarity between chunk {i} and {j} is {sim:.4f} "
                    f"— outside valid range [-1, 1]"
                )

    def test_self_similarity_is_one(self, model, sample_chunks):
        """A vector's similarity with itself must be exactly 1.0."""
        result = model.embed(sample_chunks)
        for ec in result:
            vec  = np.array(ec.embedding)
            self_sim = np.dot(vec, vec)
            assert abs(self_sim - 1.0) < 0.001, (
                f"Self-similarity of '{ec.chunk_id}' is {self_sim:.6f}, "
                f"expected 1.0"
            )


# ── Tests: Query Embedding ───────────────────────────────────────────
class TestQueryEmbedding:
    """embed_query() is used at retrieval time — test it separately."""

    def test_query_returns_correct_dimension(self, model):
        """Single query embedding must be 768-dimensional."""
        vector = model.embed_query("how does backpropagation work?")
        assert len(vector) == 768

    def test_query_is_normalized(self, model):
        """Query vector must be normalized for valid cosine search."""
        vector = model.embed_query("what is gradient descent?")
        norm   = np.linalg.norm(np.array(vector))
        assert abs(norm - 1.0) < 0.001

    def test_query_is_list_of_floats(self, model):
        """Query embedding must be a plain list, not numpy array."""
        vector = model.embed_query("explain neural networks")
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_query_similar_to_relevant_chunk(self, model, sample_chunks):
        """
        A query about ML should score higher against ML chunks
        than against the food chunk.

        This is the end-to-end proof that query embedding and
        chunk embedding are compatible and produce meaningful scores.
        """
        embedded = model.embed(sample_chunks)

        query_vec = np.array(
            model.embed_query("how do neural networks learn?")
        )
        ml_chunk_vec   = np.array(embedded[0].embedding)   # neural networks
        food_chunk_vec = np.array(embedded[2].embedding)   # recipe

        sim_ml   = np.dot(query_vec, ml_chunk_vec)
        sim_food = np.dot(query_vec, food_chunk_vec)

        assert sim_ml > sim_food, (
            f"ML query should be closer to ML chunk ({sim_ml:.4f}) "
            f"than to food chunk ({sim_food:.4f})"
        )


# ── Tests: ChromaDB Serialization ───────────────────────────────────
class TestChromaSerialization:
    """to_chroma_dict() feeds directly into ChromaDB — test it precisely."""

    def test_chroma_dict_has_four_keys(self, model, sample_chunks):
        """ChromaDB expects exactly: id, embedding, document, metadata."""
        result = model.embed(sample_chunks)
        chroma = result[0].to_chroma_dict()
        assert set(chroma.keys()) == {"id", "embedding", "document", "metadata"}

    def test_chroma_id_matches_chunk_id(self, model, sample_chunks):
        """The id field must match the original chunk_id."""
        result = model.embed(sample_chunks)
        for ec in result:
            chroma = ec.to_chroma_dict()
            assert chroma["id"] == ec.chunk_id

    def test_chroma_document_matches_text(self, model, sample_chunks):
        """The document field must be the chunk's text."""
        result = model.embed(sample_chunks)
        for ec in result:
            chroma = ec.to_chroma_dict()
            assert chroma["document"] == ec.text

    def test_chroma_metadata_types_are_valid(self, model, sample_chunks):
        """All metadata values must be str, int, or float for ChromaDB."""
        result = model.embed(sample_chunks)
        for ec in result:
            metadata = ec.to_chroma_dict()["metadata"]
            for key, value in metadata.items():
                assert isinstance(value, (str, int, float)), (
                    f"Metadata '{key}' has invalid type {type(value).__name__}"
                )


# ── Tests: Error Handling ────────────────────────────────────────────
class TestErrorHandling:
    """Fail loudly with clear messages — never silently."""

    def test_empty_chunks_raises_error(self, model):
        """Empty list input must raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty chunks"):
            model.embed([])

    def test_empty_query_raises_error(self, model):
        """Empty query string must raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty query"):
            model.embed_query("")

    def test_whitespace_query_raises_error(self, model):
        """Whitespace-only query is effectively empty — must raise error."""
        with pytest.raises(EmbeddingError, match="empty query"):
            model.embed_query("   ")