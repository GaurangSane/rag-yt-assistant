"""
vector_store.py
───────────────
Persistent vector database operations using ChromaDB.

Implements the Repository pattern — all ChromaDB details are
encapsulated here. No other module imports chromadb directly.

Design decisions:
  - Repository pattern: one class owns all DB operations
  - Idempotent writes: saving same video twice is always safe
  - Lazy collection loading: collection opened only when needed
  - SearchResult dataclass: clean return type for retrieval
  - video_id scoping: every operation is scoped to one video

Pipeline position:  Embedder → [VectorStore] → Retriever
"""

import logging
import time
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src.config import settings
from src.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger("rag_app.storage.vector_store")


# ── Data Structure ──────────────────────────────────────────────────
@dataclass
class SearchResult:
    """
    One result returned from a vector similarity search.

    Wraps the raw ChromaDB response into a clean typed object.
    Every field the rest of the system needs is directly accessible
    without navigating nested dicts like results["ids"][0][i].

    Attributes:
        chunk_id   : Unique chunk identifier (e.g. "abc_chunk_3")
        text       : The actual spoken words — sent to LLM as context
        score      : Similarity score 0.0 → 1.0 (higher = more relevant)
                     Computed as: 1 - cosine_distance
        start_time : Human-readable timestamp (e.g. "4:32")
        end_time   : Human-readable end timestamp (e.g. "5:18")
        start_sec  : Raw seconds for sorting and YouTube deep links
        video_id   : Which video this result came from
        youtube_link: Direct URL to this timestamp in the video
        chunk_index: Position in video (for chronological ordering)
    """
    chunk_id    : str
    text        : str
    score       : float
    start_time  : str
    end_time    : str
    start_sec   : float
    video_id    : str
    youtube_link: str
    chunk_index : int


# ── Custom Exceptions ────────────────────────────────────────────────
class VectorStoreError(Exception):
    """Base exception for all vector store operations."""
    pass

class CollectionNotFoundError(VectorStoreError):
    """Raised when querying a video that has not been ingested yet."""
    pass


# ── Main Class ──────────────────────────────────────────────────────
class VectorStore:
    """
    Manages all ChromaDB operations for the RAG system.

    Implements the Repository pattern — the single point of
    contact between the application and the vector database.
    No other module imports chromadb directly.

    Each YouTube video gets its own ChromaDB collection named
    "yt_{video_id}". This makes per-video operations
    (save, search, delete, exists) clean and isolated.

    Usage:
        store  = VectorStore()
        store.save(embedded_chunks, video_id="abc123")
        results = store.search(query_vector, video_id="abc123", top_k=5)
        exists  = store.exists(video_id="abc123")
    """

    def __init__(self, persist_dir: str | None = None):
        """
        Initialise the VectorStore with a persistent ChromaDB client.

        Args:
            persist_dir: Directory for ChromaDB data files.
                         Defaults to config value (./chroma_db).
                         Pass a different path in tests to avoid
                         polluting the real database.
        """
        self._persist_dir = persist_dir or str(settings.vector_db.persist_dir)

        # Initialize ChromaDB persistent client
        # PersistentClient saves data to disk immediately on every write
        # Data survives process restarts — this is what makes it a DB
        self._client = chromadb.PersistentClient(
            path     = self._persist_dir,
            settings = Settings(anonymized_telemetry=False),
        )

        logger.info(
            f"VectorStore initialized | "
            f"persist_dir={self._persist_dir}"
        )

    def _collection_name(self, video_id: str) -> str:
        """
        Build the ChromaDB collection name for a video.

        ChromaDB collection names must be:
          - 3-63 characters long
          - Start and end with alphanumeric character
          - Contain only alphanumeric characters, underscores, hyphens

        We prefix with "yt_" to namespace our collections and
        avoid collisions with any other ChromaDB users of the same dir.
        """
        prefix = settings.vector_db.collection_prefix   # "yt_"
        return f"{prefix}{video_id}"

    def _get_collection(self, video_id: str) -> chromadb.Collection:
        """
        Retrieve an existing collection for a video.

        Private method — callers use the public methods (search, delete).
        Raises CollectionNotFoundError if video has not been ingested.
        """
        name = self._collection_name(video_id)
        try:
            return self._client.get_collection(name)
        except Exception:
            raise CollectionNotFoundError(
                f"No data found for video '{video_id}'. "
                f"Run ingestion first before querying."
            )

    # ── Write Operations ─────────────────────────────────────────────

    def save(
        self,
        embedded_chunks : list[EmbeddedChunk],
        video_id        : str,
        overwrite       : bool = True,
    ) -> int:
        """
        Persist embedded chunks to ChromaDB.

        This is the WRITE operation of the ingestion pipeline.
        Called once per video after embedding.

        Idempotent by default (overwrite=True):
          - If collection exists: delete it and recreate fresh
          - If collection is new: create it
          Running save() twice on the same video is always safe.

        Args:
            embedded_chunks : Output from EmbeddingModel.embed()
            video_id        : YouTube video ID — used as collection key
            overwrite       : If True (default), replace existing data.
                              If False, raise error if video exists.

        Returns:
            Number of chunks successfully stored

        Raises:
            VectorStoreError: If video exists and overwrite=False
            VectorStoreError: If chunks list is empty
        """
        if not embedded_chunks:
            raise VectorStoreError(
                "Cannot save empty embedded_chunks list. "
                "Ensure embedding completed successfully."
            )

        name = self._collection_name(video_id)

        # Check if collection already exists
        existing_names = [c.name for c in self._client.list_collections()]
        if name in existing_names:
            if not overwrite:
                raise VectorStoreError(
                    f"Video '{video_id}' already exists in vector store. "
                    f"Pass overwrite=True to replace it."
                )
            # Delete existing collection for clean slate
            self._client.delete_collection(name)
            logger.info(
                f"Deleted existing collection '{name}' for fresh save"
            )

        # Create collection with cosine distance metric
        # Must match normalize_embeddings=True used during embedding
        collection = self._client.create_collection(
            name     = name,
            metadata = {
                "hnsw:space": settings.vector_db.distance_metric
            },
        )

        # Serialize all embedded chunks using their own method
        # to_chroma_dict() returns: {id, embedding, document, metadata}
        chroma_data = [ec.to_chroma_dict() for ec in embedded_chunks]

        logger.info(
            f"Saving {len(chroma_data)} chunks | video_id={video_id}"
        )
        save_start = time.time()

        # ChromaDB batch add — all chunks in one call
        # Much faster than adding one chunk at a time
        collection.add(
            ids        = [d["id"]        for d in chroma_data],
            embeddings = [d["embedding"] for d in chroma_data],
            documents  = [d["document"]  for d in chroma_data],
            metadatas  = [d["metadata"]  for d in chroma_data],
        )

        save_time = time.time() - save_start
        count     = collection.count()

        logger.info(
            f"Save complete | "
            f"video_id={video_id} | "
            f"chunks={count} | "
            f"time={save_time:.2f}s"
        )

        return count

    # ── Read Operations ──────────────────────────────────────────────

    def search(
        self,
        query_vector : list[float],
        video_id     : str,
        top_k        : int | None = None,
    ) -> list[SearchResult]:
        """
        Find the most semantically similar chunks to a query vector.

        This is called by the Retriever (Step 6) for every
        transformed query. The query_vector comes from
        EmbeddingModel.embed_query() — same model, same normalization.

        Args:
            query_vector : Normalized 768-dim vector from embed_query()
            video_id     : Which video to search within
            top_k        : Number of results to return.
                           Defaults to config value (5).

        Returns:
            List of SearchResult objects sorted by score descending.
            Best match first. Empty list if no results found.

        Raises:
            CollectionNotFoundError: If video has not been ingested
        """
        top_k      = top_k or settings.retrieval.retrieve_top_k
        collection = self._get_collection(video_id)

        logger.debug(
            f"Searching | video_id={video_id} | top_k={top_k}"
        )

        raw = collection.query(
            query_embeddings = [query_vector],
            n_results        = min(top_k, collection.count()),
            include          = ["documents", "metadatas", "distances"],
        )

        # ChromaDB returns results nested in lists because it supports
        # batch queries. We sent one query so we always take index [0].
        #
        # raw = {
        #   "ids"       : [["chunk_0", "chunk_3", ...]],   ← index [0]
        #   "documents" : [["text0",   "text3",   ...]],   ← index [0]
        #   "metadatas" : [[{...},     {...},      ...]],   ← index [0]
        #   "distances" : [[0.12,      0.31,       ...]],   ← index [0]
        # }

        results = []
        for i in range(len(raw["ids"][0])):
            meta     = raw["metadatas"][0][i]
            distance = raw["distances"][0][i]

            # ChromaDB cosine distance → similarity score
            # distance=0.0 means identical, distance=2.0 means opposite
            # similarity = 1 - distance gives us 1.0=identical, -1.0=opposite
            score = round(1.0 - distance, 6)

            results.append(SearchResult(
                chunk_id    = raw["ids"][0][i],
                text        = raw["documents"][0][i],
                score       = score,
                start_time  = meta.get("start_time", ""),
                end_time    = meta.get("end_time", ""),
                start_sec   = float(meta.get("start_sec", 0.0)),
                video_id    = meta.get("video_id", video_id),
                youtube_link= meta.get("youtube_link", ""),
                chunk_index = int(meta.get("chunk_index", 0)),
            ))

        logger.debug(
            f"Search complete | "
            f"results={len(results)} | "
            f"top_score={(results[0].score if results else 0):.4f}"
        )

        return results

    def get_all_chunks(self, video_id: str) -> list[dict]:
        """
        Retrieve all stored chunks for a video without similarity search.

        Used by the Retriever to build the BM25 index — BM25 needs
        access to all chunk texts, not just the top-k similar ones.
        Called once per pipeline run then cached in the retriever.

        Args:
            video_id: YouTube video ID

        Returns:
            List of dicts with keys: chunk_id, text, metadata fields
            All chunks for the video, in storage order.

        Raises:
            CollectionNotFoundError: If video has not been ingested
        """
        collection = self._get_collection(video_id)

        raw = collection.get(
            include=["documents", "metadatas"]
        )

        chunks = []
        for i in range(len(raw["ids"])):
            meta = raw["metadatas"][i]
            chunks.append({
                "chunk_id"   : raw["ids"][i],
                "text"       : raw["documents"][i],
                "start_time" : meta.get("start_time", ""),
                "end_time"   : meta.get("end_time", ""),
                "start_sec"  : float(meta.get("start_sec", 0.0)),
                "video_id"   : meta.get("video_id", video_id),
                "youtube_link": meta.get("youtube_link", ""),
                "chunk_index": int(meta.get("chunk_index", 0)),
            })

        logger.info(
            f"Retrieved {len(chunks)} chunks | video_id={video_id}"
        )

        return chunks

    # ── Utility Operations ───────────────────────────────────────────

    def exists(self, video_id: str) -> bool:
        """
        Check whether a video has already been ingested.

        Used by the pipeline to skip Steps 1-4 when video
        is already indexed. This is what makes repeated questions
        on the same video near-instant.

        Args:
            video_id: YouTube video ID to check

        Returns:
            True if the video has been ingested, False otherwise
        """
        name            = self._collection_name(video_id)
        existing_names  = [c.name for c in self._client.list_collections()]
        found           = name in existing_names

        logger.debug(
            f"Existence check | video_id={video_id} | exists={found}"
        )

        return found

    def count(self, video_id: str) -> int:
        """
        Return the number of chunks stored for a video.

        Useful for debugging and health checks.

        Raises:
            CollectionNotFoundError: If video has not been ingested
        """
        return self._get_collection(video_id).count()

    def delete(self, video_id: str) -> None:
        """
        Remove all stored data for a video.

        Irreversible. Used in tests and when re-ingesting a video
        with updated parameters.

        Args:
            video_id: YouTube video ID to delete

        Raises:
            CollectionNotFoundError: If video does not exist
        """
        name = self._collection_name(video_id)

        # Verify it exists before trying to delete
        if not self.exists(video_id):
            raise CollectionNotFoundError(
                f"Cannot delete video '{video_id}' — not found in store."
            )

        self._client.delete_collection(name)
        logger.info(f"Deleted collection | video_id={video_id}")

    def list_videos(self) -> list[str]:
        """
        Return all video IDs currently stored in the database.

        Strips the "yt_" prefix to return clean video IDs.
        Useful for debugging and building a video library UI.
        """
        prefix      = settings.vector_db.collection_prefix
        collections = self._client.list_collections()
        video_ids   = [
            c.name.removeprefix(prefix)
            for c in collections
            if c.name.startswith(prefix)
        ]

        logger.info(f"Listed {len(video_ids)} videos in store")
        return video_ids


# ── Module Exports ───────────────────────────────────────────────────
__all__ = [
    "VectorStore",
    "SearchResult",
    "VectorStoreError",
    "CollectionNotFoundError",
]