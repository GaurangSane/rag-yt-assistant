"""
reranker.py
───────────
Re-scores retrieved chunks using a cross-encoder model that
reads each [question, chunk] pair jointly for deep relevance scoring.

Design decisions:
  - Singleton pattern: cross-encoder loads once (same as embedder)
  - Immutable input: HybridSearchResult objects never modified
  - RankedResult wraps input and adds rerank_score cleanly
  - Scores are raw logits — only relative order matters, not magnitude
  - top_k strictly enforced: exactly k results returned

Pipeline position:  Retriever → [Reranker] → PromptBuilder
"""

import logging
import time
from dataclasses import dataclass

from sentence_transformers.cross_encoder import CrossEncoder

from src.config import settings
from src.retrieval.retriever import HybridSearchResult

logger = logging.getLogger("rag_app.retrieval.reranker")


# ── Data Structure ───────────────────────────────────────────────────
@dataclass
class RankedResult:
    """
    A retrieved chunk with its cross-encoder relevance score.

    Wraps HybridSearchResult (from retriever) and adds rerank_score.
    This is the final typed object that flows into PromptBuilder.

    Attributes:
        hybrid_result : The HybridSearchResult from the retriever
                        Contains chunk text, metadata, and RRF score
        rerank_score  : Raw logit from cross-encoder (unbounded float)
                        Higher = more relevant to the question
                        Only relative order matters — not absolute value
        rank          : Final position after reranking (1-indexed)
                        Rank 1 = most relevant chunk
    """
    hybrid_result : HybridSearchResult
    rerank_score  : float
    rank          : int

    # ── Pass-through properties ──────────────────────────────────
    # Clean access without chaining: result.text not result.hybrid_result.text

    @property
    def chunk_id(self) -> str:
        return self.hybrid_result.chunk_id

    @property
    def text(self) -> str:
        return self.hybrid_result.text

    @property
    def start_time(self) -> str:
        return self.hybrid_result.start_time

    @property
    def end_time(self) -> str:
        return self.hybrid_result.end_time

    @property
    def start_sec(self) -> float:
        return self.hybrid_result.start_sec

    @property
    def video_id(self) -> str:
        return self.hybrid_result.video_id

    @property
    def youtube_link(self) -> str:
        return self.hybrid_result.youtube_link

    @property
    def rrf_score(self) -> float:
        """The RRF score from retrieval — useful for comparison in logs."""
        return self.hybrid_result.rrf_score

    def to_prompt_dict(self) -> dict:
        """
        Serialize this result into a clean dict for PromptBuilder.

        PromptBuilder only needs: text, start_time, end_time,
        youtube_link, and rank. Everything else stays internal.

        Returns:
            Dict with exactly the fields PromptBuilder expects.
        """
        return {
            "rank"        : self.rank,
            "text"        : self.text,
            "start_time"  : self.start_time,
            "end_time"    : self.end_time,
            "youtube_link": self.youtube_link,
            "chunk_id"    : self.chunk_id,
            "rerank_score": self.rerank_score,
        }


# ── Custom Exception ─────────────────────────────────────────────────
class RerankerError(Exception):
    """Raised when reranking fails for a structural reason."""
    pass


# ── Main Class — Singleton ────────────────────────────────────────────
class CrossEncoderReranker:
    """
    Reranks retrieved chunks by deep pairwise relevance scoring.

    The cross-encoder reads each [question, chunk] pair as a single
    input — unlike the bi-encoder which encodes them separately.
    This joint reading catches subtle relevance signals that
    vector similarity alone misses.

    Singleton pattern: the ~80MB model loads exactly once
    regardless of how many times CrossEncoderReranker() is called.

    Two-stage retrieval pattern:
      Stage 1 — HybridRetriever: fast, retrieves top-5 candidates
      Stage 2 — CrossEncoderReranker: slow but deep, keeps top-3

    Usage:
        reranker = CrossEncoderReranker()
        ranked   = reranker.rerank(
            question  = "how does gradient descent work?",
            candidates = hybrid_results,   # from HybridRetriever
            top_k      = 3,
        )
    """

    _instance = None
    _model    = None

    def __new__(cls):
        """Singleton: create CrossEncoderReranker at most once."""
        if cls._instance is None:
            logger.info(
                f"Creating CrossEncoderReranker singleton | "
                f"model={settings.reranker.model_name}"
            )
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Load the cross-encoder model if not already loaded."""
        if CrossEncoderReranker._model is not None:
            return

        logger.info(
            f"Loading reranker model: {settings.reranker.model_name}"
        )
        load_start = time.time()

        try:
            CrossEncoderReranker._model = CrossEncoder(
                settings.reranker.model_name
            )
        except Exception as e:
            raise RerankerError(
                f"Failed to load reranker model "
                f"'{settings.reranker.model_name}': {e}"
            ) from e

        load_time = time.time() - load_start
        logger.info(
            f"Reranker model loaded | "
            f"model={settings.reranker.model_name} | "
            f"time={load_time:.2f}s"
        )

    @property
    def model(self) -> CrossEncoder:
        """Access the underlying CrossEncoder model."""
        return CrossEncoderReranker._model

    def rerank(
        self,
        question   : str,
        candidates : list[HybridSearchResult],
        top_k      : int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank candidate chunks by deep pairwise relevance scoring.

        For each candidate, builds a [question, chunk_text] pair
        and scores them together through the cross-encoder.
        Returns top_k chunks sorted by score descending.

        Args:
            question   : The user's ORIGINAL question (not transformed)
                         We rerank against what the user actually asked,
                         not against the search-optimised query variants
            candidates : Output from HybridRetriever.retrieve()
                         Typically 5 chunks
            top_k      : How many to keep after reranking
                         Defaults to config value (3)

        Returns:
            List of RankedResult sorted by rerank_score descending.
            Length = min(top_k, len(candidates))
            rank field is 1-indexed: best chunk has rank=1

        Raises:
            RerankerError: If candidates list is empty
            RerankerError: If question is empty
        """
        if not candidates:
            raise RerankerError(
                "Cannot rerank empty candidates list. "
                "Ensure retrieval completed successfully."
            )

        if not question or not question.strip():
            raise RerankerError(
                "Question cannot be empty for reranking."
            )

        top_k = top_k or settings.retrieval.rerank_top_k

        logger.info(
            f"Reranking | "
            f"candidates={len(candidates)} | "
            f"top_k={top_k} | "
            f"question='{question[:60]}'"
        )
        rerank_start = time.time()

        # ── Build question-chunk pairs ──────────────────────────
        # Cross-encoder expects: List[List[str, str]]
        # Each inner list is [query, passage]
        # This is why it's called "cross" — it reads both together
        pairs = [
            [question, candidate.text]
            for candidate in candidates
        ]

        # ── Score all pairs in one batch ────────────────────────
        # predict() returns a numpy array of raw logit scores
        # One score per pair — shape: (len(candidates),)
        # Higher = more relevant (but absolute values are meaningless)
        scores = CrossEncoderReranker._model.predict(pairs)

        # ── Build RankedResult objects ──────────────────────────
        # Pair each candidate with its score — zip preserves order
        scored = [
            (candidate, float(score))
            for candidate, score in zip(candidates, scores)
        ]

        # Sort by score descending — highest relevance first
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep only top_k, assign 1-indexed ranks
        results = [
            RankedResult(
                hybrid_result = candidate,
                rerank_score  = score,
                rank          = rank_position,
            )
            for rank_position, (candidate, score)
            in enumerate(scored[:top_k], start=1)
        ]

        rerank_time = (time.time() - rerank_start) * 1000

        # ── Log score analysis ──────────────────────────────────
        # The score gap between kept and dropped chunks tells us
        # how confident the reranker is about its cutoff.
        # Large gap = clear separation. Small gap = borderline.
        if len(scored) > top_k:
            last_kept    = scored[top_k - 1][1]
            first_dropped= scored[top_k][1]
            gap          = last_kept - first_dropped
            logger.info(
                f"Reranking complete | "
                f"kept={len(results)} | "
                f"score_gap={gap:.3f} | "
                f"time={rerank_time:.0f}ms"
            )
        else:
            logger.info(
                f"Reranking complete | "
                f"kept={len(results)} | "
                f"time={rerank_time:.0f}ms"
            )

        for r in results:
            logger.debug(
                f"  Rank {r.rank}: score={r.rerank_score:.3f} | "
                f"[{r.start_time}] {r.text[:50]}..."
            )

        return results


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "CrossEncoderReranker",
    "RankedResult",
    "RerankerError",
]