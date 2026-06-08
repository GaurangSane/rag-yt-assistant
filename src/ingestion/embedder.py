"""
embedder.py
───────────
Converts Chunk objects into EmbeddedChunks by generating
semantic vector representations using a local sentence transformer.

Design decisions:
  - Singleton pattern: model loads once, shared across the application
  - Immutable input: Chunk objects are never modified
  - EmbeddedChunk wraps Chunk + adds embedding — clean separation
  - Batch encoding: all chunks embedded in one efficient forward pass
  - normalize_embeddings=True: vectors on unit sphere for cosine search

Pipeline position:  Chunker → [Embedder] → VectorStore
"""

import logging
import time
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger("rag_app.ingestion.embedder")


# ── Data Structure ──────────────────────────────────────────────────
@dataclass
class EmbeddedChunk:
    """
    A Chunk paired with its semantic embedding vector.

    Keeps the original Chunk intact (immutable input principle)
    and adds the embedding alongside it. This is what gets
    stored in ChromaDB — the chunk provides metadata and text,
    the embedding provides the search key.

    Attributes:
        chunk    : The original Chunk object — all metadata intact
        embedding: 768-dimensional vector representing chunk meaning
                   Normalized to unit length for cosine similarity
    """
    chunk    : Chunk
    embedding: list[float]

    # ── Convenience pass-through properties ──────────────────
    # These let callers write embedded.chunk_id instead of
    # embedded.chunk.chunk_id — cleaner code downstream

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def video_id(self) -> str:
        return self.chunk.video_id

    @property
    def start_time(self) -> str:
        return self.chunk.start_time

    @property
    def end_time(self) -> str:
        return self.chunk.end_time

    @property
    def start_sec(self) -> float:
        return self.chunk.start_sec

    @property
    def embedding_dimensions(self) -> int:
        """How many dimensions this embedding has (should be 768)."""
        return len(self.embedding)

    def to_chroma_dict(self) -> dict:
        """
        Serialize this EmbeddedChunk into the four components
        ChromaDB expects when storing a record.

        Returns dict with keys: id, embedding, document, metadata
        These map directly to ChromaDB's .add() parameters.
        """
        return {
            "id"       : self.chunk.chunk_id,
            "embedding": self.embedding,
            "document" : self.chunk.text,
            "metadata" : self.chunk.to_metadata_dict(),
        }


# ── Custom Exception ────────────────────────────────────────────────
class EmbeddingError(Exception):
    """Raised when embedding fails for any reason."""
    pass


# ── Main Class — Singleton ──────────────────────────────────────────
class EmbeddingModel:
    """
    Wraps the sentence-transformer model with singleton behaviour.

    The ML model is loaded from disk exactly once regardless of
    how many times EmbeddingModel() is called. This prevents
    4-second repeated load times in a running application.

    Usage:
        model          = EmbeddingModel()
        embedded_chunks = model.embed(chunks)

    Singleton behaviour:
        model_a = EmbeddingModel()
        model_b = EmbeddingModel()
        assert model_a is model_b   # True — same object
    """

    # Class-level variable shared across ALL instances
    # None until first EmbeddingModel() is called
    _instance = None
    _model    = None      # the actual SentenceTransformer

    def __new__(cls):
        """
        Control object creation — the heart of the singleton pattern.

        __new__ is called before __init__.
        If an instance already exists, return it directly.
        If not, create one and store it at the class level.
        """
        if cls._instance is None:
            logger.info(
                f"Creating EmbeddingModel singleton | "
                f"model={settings.embedding.model_name}"
            )
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Load the model if not already loaded.

        Because of the singleton pattern, this runs every time
        EmbeddingModel() is called — but the if-guard ensures
        the expensive model.load() only happens once.
        """
        # Guard: if model already loaded, do nothing
        # Without this guard __init__ would reload the model
        # every time EmbeddingModel() is called
        if EmbeddingModel._model is not None:
            return

        logger.info(
            f"Loading embedding model: {settings.embedding.model_name}"
        )
        load_start = time.time()

        try:
            EmbeddingModel._model = SentenceTransformer(
                settings.embedding.model_name
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model "
                f"'{settings.embedding.model_name}': {e}"
            ) from e

        load_time = time.time() - load_start
        logger.info(
            f"Embedding model loaded | "
            f"time={load_time:.2f}s | "
            f"dimensions={settings.embedding.dimensions}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """Access the underlying SentenceTransformer model."""
        return EmbeddingModel._model

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string at retrieval time.

        This is called in Step 6 (retriever) when we need to
        embed the user's question for vector search.

        CRITICAL: Uses the exact same model and normalization
        settings as embed() — vectors must be comparable.
        Mixing models or normalization settings produces
        completely meaningless similarity scores.

        Args:
            query: The user's question or transformed query string

        Returns:
            Normalized 768-dimensional vector as Python list
        """
        if not query or not query.strip():
            raise EmbeddingError(
                "Cannot embed an empty query string."
            )

        vector = EmbeddingModel._model.encode(
            query,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )

        logger.debug(f"Query embedded | query='{query[:60]}...'")
        return vector.tolist()

    def embed(
        self,
        chunks    : list[Chunk],
        batch_size: int | None = None,
    ) -> list[EmbeddedChunk]:
        """
        Embed a list of Chunk objects into EmbeddedChunks.

        Processes all chunks in a single efficient batch call.
        Batch processing is significantly faster than embedding
        chunks one by one because the model processes them in
        parallel on the CPU/GPU.

        Args:
            chunks    : List of Chunk objects from TranscriptChunker
            batch_size: How many chunks to process per forward pass
                        Defaults to config value (32)
                        Reduce if you run out of memory

        Returns:
            List of EmbeddedChunk objects in the same order as input.
            Order preservation is guaranteed — chunk[i] pairs with
            embedded_chunks[i].

        Raises:
            EmbeddingError: If chunks list is empty or encoding fails
        """
        if not chunks:
            raise EmbeddingError(
                "Cannot embed an empty chunks list."
            )

        batch_size = batch_size or settings.embedding.batch_size

        logger.info(
            f"Embedding {len(chunks)} chunks | "
            f"model={settings.embedding.model_name} | "
            f"batch_size={batch_size}"
        )

        embed_start = time.time()

        # Extract text from each chunk for batch encoding
        # We embed text only — metadata is not part of the vector
        texts = [chunk.text for chunk in chunks]

        try:
            # encode() returns a numpy array of shape:
            # (num_chunks, embedding_dimensions) = (25, 768)
            vectors = EmbeddingModel._model.encode(
                texts,
                batch_size           = batch_size,
                show_progress_bar    = True,
                normalize_embeddings = settings.embedding.normalize_embeddings,
                convert_to_numpy     = True,   # ensures consistent output type
            )
        except Exception as e:
            raise EmbeddingError(
                f"Encoding failed for {len(chunks)} chunks: {e}"
            ) from e

        # Pair each chunk with its corresponding vector
        # zip() guarantees order: chunks[i] pairs with vectors[i]
        embedded_chunks = [
            EmbeddedChunk(
                chunk     = chunk,
                embedding = vector.tolist(),    # numpy array → Python list
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        embed_time = time.time() - embed_start
        logger.info(
            f"Embedding complete | "
            f"chunks={len(embedded_chunks)} | "
            f"dimensions={settings.embedding.dimensions} | "
            f"time={embed_time:.2f}s"
        )

        return embedded_chunks


# ── Module Exports ──────────────────────────────────────────────────
__all__ = [
    "EmbeddingModel",
    "EmbeddedChunk",
    "EmbeddingError",
]