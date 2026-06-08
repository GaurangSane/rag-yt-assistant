"""
retriever.py
────────────
Hybrid retrieval combining semantic search and BM25 keyword
search, fused via Reciprocal Rank Fusion (RRF).

Design decisions:
  - BM25 index cached per video: built once, reused every question
  - Semantic + keyword search run for EACH transformed query
  - RRF merges all result lists into one ranked output
  - HybridSearchResult wraps SearchResult cleanly
  - sources field records which methods found each chunk (debugging)

Pipeline position:  QueryTransformer → [Retriever] → Reranker
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.embedder import EmbeddingModel
from src.storage.vector_store import VectorStore, SearchResult
from src.storage.vector_store import CollectionNotFoundError

logger = logging.getLogger("rag_app.retrieval.retriever")


# ── Data Structures ──────────────────────────────────────────────────
@dataclass
class HybridSearchResult:
    """
    A retrieved chunk with its RRF fusion score and source metadata.

    Wraps SearchResult (from storage layer) and adds retrieval-
    specific fields: rrf_score and sources.

    Attributes:
        search_result : Original SearchResult from ChromaDB or BM25
        rrf_score     : Reciprocal Rank Fusion score (higher = better)
                        Formula: sum(1 / (rank + k)) across all lists
        sources       : Which search methods found this chunk
                        e.g. ["semantic_q1", "bm25_q2"]
                        Useful for debugging retrieval quality
    """
    search_result : SearchResult
    rrf_score     : float
    sources       : list[str] = field(default_factory=list)

    # ── Pass-through properties ──────────────────────────────
    # Let callers write result.text instead of result.search_result.text

    @property
    def chunk_id(self) -> str:
        return self.search_result.chunk_id

    @property
    def text(self) -> str:
        return self.search_result.text

    @property
    def start_time(self) -> str:
        return self.search_result.start_time

    @property
    def end_time(self) -> str:
        return self.search_result.end_time

    @property
    def start_sec(self) -> float:
        return self.search_result.start_sec

    @property
    def video_id(self) -> str:
        return self.search_result.video_id

    @property
    def youtube_link(self) -> str:
        return self.search_result.youtube_link

    @property
    def found_by_both(self) -> bool:
        """
        True if both semantic AND keyword search found this chunk.
        Chunks found by both methods are almost always truly relevant.
        """
        has_semantic = any("semantic" in s for s in self.sources)
        has_bm25     = any("bm25" in s for s in self.sources)
        return has_semantic and has_bm25


# ── Custom Exception ─────────────────────────────────────────────────
class RetrieverError(Exception):
    """Raised when retrieval fails for a structural reason."""
    pass


# ── BM25 Index Cache ─────────────────────────────────────────────────
@dataclass
class _BM25Index:
    """
    Internal cache entry for one video's BM25 index.

    Stores both the index and the source chunks so we can
    map BM25 result indices back to chunk metadata.
    """
    bm25        : BM25Okapi
    chunks      : list[dict]   # parallel to BM25 tokenized corpus
    video_id    : str


# ── Main Class ───────────────────────────────────────────────────────
class HybridRetriever:
    """
    Retrieves relevant chunks using semantic + BM25 hybrid search.

    For each transformed query:
      1. Semantic search  → embed query → ChromaDB similarity search
      2. BM25 search      → tokenize query → keyword frequency search
    All result lists are merged via Reciprocal Rank Fusion (RRF).

    BM25 indexes are cached per video — built on first question,
    reused for all subsequent questions on the same video.

    Usage:
        retriever = HybridRetriever(store, embedding_model)
        results   = retriever.retrieve(
            queries  = ["gradient descent optimization", ...],
            video_id = "abc123",
            top_k    = 5,
        )
    """

    def __init__(
        self,
        store           : VectorStore,
        embedding_model : EmbeddingModel,
    ):
        """
        Args:
            store           : VectorStore instance for semantic search
            embedding_model : EmbeddingModel instance for query embedding
                              Must be the SAME instance used during ingestion
        """
        self._store           = store
        self._embedding_model = embedding_model
        self._bm25_cache      : dict[str, _BM25Index] = {}
        self._rrf_k           = settings.retrieval.rrf_k

        logger.info("HybridRetriever initialized")

    # ── BM25 Index Management ─────────────────────────────────────────
    def _get_bm25_index(self, video_id: str) -> _BM25Index:
        """
        Return the BM25 index for a video, building it if necessary.

        Lazy initialisation with caching:
          - First call for a video: loads chunks from DB, builds index
          - Subsequent calls: returns cached index instantly

        Args:
            video_id: YouTube video ID

        Returns:
            _BM25Index with the BM25 object and source chunks
        """
        if video_id in self._bm25_cache:
            logger.debug(f"BM25 cache hit | video_id={video_id}")
            return self._bm25_cache[video_id]

        logger.info(
            f"Building BM25 index | video_id={video_id} "
            f"(first question for this video)"
        )
        build_start = time.time()
        try:
        # Load all chunks from vector store
           chunks = self._store.get_all_chunks(video_id)
        except CollectionNotFoundError:
            raise RetrieverError(
                f"No chunks found for video '{video_id}'"
            )
        if not chunks:
            raise RetrieverError(
                f"No chunks found for video '{video_id}'. "
                f"Ensure ingestion completed successfully."
            )

        # Tokenize: lowercase split is standard for BM25
        # More sophisticated tokenization (stemming, stopwords) can
        # improve quality but adds complexity — simple split is fine here
        tokenized = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        bm25_index = _BM25Index(
            bm25     = BM25Okapi(tokenized),
            chunks   = chunks,
            video_id = video_id,
        )

        self._bm25_cache[video_id] = bm25_index

        build_time = (time.time() - build_start) * 1000
        logger.info(
            f"BM25 index built | "
            f"video_id={video_id} | "
            f"chunks={len(chunks)} | "
            f"time={build_time:.0f}ms"
        )

        return bm25_index

    def invalidate_cache(self, video_id: str) -> None:
        """
        Remove a video's BM25 index from cache.

        Call this after re-ingesting a video with new parameters
        so the next retrieval builds a fresh index from the
        updated data.
        """
        if video_id in self._bm25_cache:
            del self._bm25_cache[video_id]
            logger.info(f"BM25 cache invalidated | video_id={video_id}")

    # ── Individual Search Methods ─────────────────────────────────────
    def _semantic_search(
        self,
        query    : str,
        video_id : str,
        top_k    : int,
    ) -> list[SearchResult]:
        """
        Single semantic search for one query string.

        Embeds the query and finds nearest vectors in ChromaDB.
        Returns top_k results sorted by cosine similarity.
        """
        query_vector = self._embedding_model.embed_query(query)
        return self._store.search(
            query_vector = query_vector,
            video_id     = video_id,
            top_k        = top_k,
        )

    def _bm25_search(
        self,
        query    : str,
        video_id : str,
        top_k    : int,
    ) -> list[SearchResult]:
        """
        Single BM25 keyword search for one query string.

        Tokenizes the query and scores all chunks by term
        frequency and inverse document frequency.
        Returns top_k results as SearchResult objects.
        """
        index         = self._get_bm25_index(video_id)
        tokenized_q   = query.lower().split()
        scores        = index.bm25.get_scores(tokenized_q)

        # argsort returns indices that would sort the array ascending
        # [::-1] reverses to descending (highest score first)
        # [:top_k] takes only what we need
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = index.chunks[idx]
            score = float(scores[idx])

            # Skip chunks with zero BM25 score — they have no
            # keyword overlap with the query at all
            if score <= 0:
                continue

            results.append(SearchResult(
                chunk_id    = chunk["chunk_id"],
                text        = chunk["text"],
                score       = score,
                start_time  = chunk["start_time"],
                end_time    = chunk["end_time"],
                start_sec   = chunk["start_sec"],
                video_id    = chunk["video_id"],
                youtube_link= chunk.get("youtube_link", ""),
                chunk_index = chunk.get("chunk_index", 0),
            ))

        return results

    # ── RRF Fusion ────────────────────────────────────────────────────
    def _reciprocal_rank_fusion(
        self,
        result_lists : list[tuple[str, list[SearchResult]]],
    ) -> list[HybridSearchResult]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.

        RRF formula:  score(chunk) = Σ  1 / (rank + k)
                                    all lists
                                    where chunk appears

        The constant k=60 is the standard from the original RRF paper.
        It prevents top-ranked items from dominating excessively.
        Chunks appearing consistently across multiple lists score highest.

        Args:
            result_lists: List of (source_name, results) tuples
                          source_name identifies which search produced it
                          e.g. ("semantic_q1", [...]), ("bm25_q2", [...])

        Returns:
            Deduplicated list of HybridSearchResult sorted by RRF score
        """
        # rrf_scores  : chunk_id → cumulative RRF score
        # chunk_data  : chunk_id → SearchResult (first occurrence wins)
        # chunk_sources: chunk_id → list of source names
        rrf_scores    : dict[str, float]       = {}
        chunk_data    : dict[str, SearchResult]= {}
        chunk_sources : dict[str, list[str]]   = {}

        for source_name, results in result_lists:
            for rank, result in enumerate(results):
                cid = result.chunk_id

                # RRF formula — rank is 0-indexed so +1 for correct math
                rrf_scores[cid] = (
                    rrf_scores.get(cid, 0.0) + 1.0 / (rank + 1 + self._rrf_k)
                )

                # Store chunk data on first encounter
                if cid not in chunk_data:
                    chunk_data[cid]    = result
                    chunk_sources[cid] = []

                chunk_sources[cid].append(source_name)

        # Sort by RRF score descending
        sorted_ids = sorted(
            rrf_scores.keys(),
            key     = lambda cid: rrf_scores[cid],
            reverse = True,
        )

        return [
            HybridSearchResult(
                search_result = chunk_data[cid],
                rrf_score     = round(rrf_scores[cid], 8),
                sources       = chunk_sources[cid],
            )
            for cid in sorted_ids
        ]

    # ── Main Public Method ────────────────────────────────────────────
    def retrieve(
        self,
        queries  : list[str],
        video_id : str,
        top_k    : int | None = None,
    ) -> list[HybridSearchResult]:
        """
        Run hybrid retrieval for a list of query variants.

        For each query: semantic search + BM25 search
        All result lists merged via RRF → top_k returned

        This is the single method the pipeline calls.
        Everything else in this class is an implementation detail.

        Args:
            queries  : List of query strings from QueryTransformer
                       Typically 3 variants of the same question
            video_id : Which video to search
            top_k    : How many final results to return
                       Defaults to config value (5)

        Returns:
            List of HybridSearchResult sorted by RRF score (best first)
            Length = min(top_k, total unique chunks found)

        Raises:
            RetrieverError: If queries list is empty
            RetrieverError: If video has not been ingested
        """
        if not queries:
            raise RetrieverError("Queries list cannot be empty.")

        top_k          = top_k or settings.retrieval.retrieve_top_k
        n_per_search   = top_k + 2   # fetch slightly more per search
                                      # to give RRF more to work with

        logger.info(
            f"Hybrid retrieval | "
            f"video_id={video_id} | "
            f"queries={len(queries)} | "
            f"top_k={top_k}"
        )
        retrieve_start = time.time()

        # ── Run all searches ──────────────────────────────────
        # Each query generates 2 result lists (semantic + BM25)
        # Total lists = len(queries) × 2
        all_result_lists : list[tuple[str, list[SearchResult]]] = []

        for i, query in enumerate(queries):
            logger.debug(f"  Searching query {i+1}: '{query[:60]}'")

            # Semantic search for this query
            try:
                sem_results = self._semantic_search(
                    query, video_id, n_per_search
                )
                all_result_lists.append((f"semantic_q{i+1}", sem_results))
                logger.debug(
                    f"    Semantic: {len(sem_results)} results"
                )
            except Exception as e:
                logger.warning(f"Semantic search failed for query {i+1}: {e}")

            # BM25 search for this query
            try:
                bm25_results = self._bm25_search(
                    query, video_id, n_per_search
                )
                all_result_lists.append((f"bm25_q{i+1}", bm25_results))
                logger.debug(
                    f"    BM25: {len(bm25_results)} results"
                )
            except Exception as e:
                logger.warning(f"BM25 search failed for query {i+1}: {e}")

        if not all_result_lists:
            raise RetrieverError(
                f"All search methods failed for video '{video_id}'. "
                f"Check logs for individual search errors."
            )

        # ── Fuse via RRF ──────────────────────────────────────
        fused   = self._reciprocal_rank_fusion(all_result_lists)
        results = fused[:top_k]

        retrieve_time = (time.time() - retrieve_start) * 1000

        # Log fusion statistics — useful for debugging retrieval quality
        both_count = sum(1 for r in results if r.found_by_both)
        logger.info(
            f"Retrieval complete | "
            f"unique_chunks={len(fused)} | "
            f"returned={len(results)} | "
            f"found_by_both={both_count}/{len(results)} | "
            f"time={retrieve_time:.0f}ms"
        )

        return results


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "HybridRetriever",
    "HybridSearchResult",
    "RetrieverError",
]