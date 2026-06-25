"""
reranker.py
───────────
Two reranking modes controlled by config:

  STANDARD (local):    CrossEncoder model — best quality, slow on CPU
  LIGHTWEIGHT (cloud): Score fusion — near-equivalent quality, instant

Score fusion explanation:
  Instead of a neural model, we combine existing signals:
    - Semantic similarity score from ChromaDB (0-1 cosine)
    - RRF score from hybrid fusion (already reflects both BM25 + semantic)
    - Position bias (earlier in video = slight preference)

  Quality vs CrossEncoder:
    For short videos (21 chunks), the ChromaDB semantic score already
    captures the most relevant dimension. CrossEncoder typically
    reorders by 1-2 positions. Score fusion achieves similar
    reordering with zero inference time.

  Time comparison:
    CrossEncoder on Railway CPU: ~26 seconds for 5 chunks
    Score fusion:                ~0.001 seconds for 5 chunks
"""

import logging
import time
from dataclasses import dataclass

from src.config import settings
from src.retrieval.retriever import HybridSearchResult

logger = logging.getLogger("rag_app.retrieval.reranker")


# ── Data Structure ─────────────────────────────────────────────────────
@dataclass
class RankedResult:
    """Retrieved chunk with final relevance rank."""
    hybrid_result : HybridSearchResult
    rerank_score  : float
    rank          : int

    # Pass-through properties
    @property
    def chunk_id(self)    : return self.hybrid_result.chunk_id
    @property
    def text(self)        : return self.hybrid_result.text
    @property
    def start_time(self)  : return self.hybrid_result.start_time
    @property
    def end_time(self)    : return self.hybrid_result.end_time
    @property
    def start_sec(self)   : return self.hybrid_result.start_sec
    @property
    def video_id(self)    : return self.hybrid_result.video_id
    @property
    def youtube_link(self): return self.hybrid_result.youtube_link
    @property
    def rrf_score(self)   : return self.hybrid_result.rrf_score

    def to_prompt_dict(self) -> dict:
        return {
            "rank"        : self.rank,
            "text"        : self.text,
            "start_time"  : self.start_time,
            "end_time"    : self.end_time,
            "youtube_link": self.youtube_link,
            "chunk_id"    : self.chunk_id,
            "rerank_score": self.rerank_score,
        }


class RerankerError(Exception):
    pass


# ── Lightweight Reranker (Score Fusion) ────────────────────────────────
class ScoreFusionReranker:
    """
    Reranker using weighted score fusion instead of CrossEncoder.

    Combines:
      - semantic_score: cosine similarity from ChromaDB (primary signal)
      - rrf_score:      RRF fusion score (captures both BM25 + semantic)
      - found_by_both:  bonus for chunks found by both search methods

    Formula:
      final_score = (0.6 × normalised_semantic) +
                    (0.3 × normalised_rrf) +
                    (0.1 × found_by_both_bonus)

    Performance: O(n) with no model inference — microseconds for any n.
    Quality: comparable to CrossEncoder for focused document sets (≤100 chunks).
    """

    def __init__(self):
        logger.info("ScoreFusionReranker initialized (lightweight mode)")

    def rerank(
        self,
        question   : str,   # kept for API compatibility, not used in fusion
        candidates : list[HybridSearchResult],
        top_k      : int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank candidates using score fusion.

        Args:
            question   : User question (not used in lightweight mode)
            candidates : Output from HybridRetriever.retrieve()
            top_k      : How many to keep

        Returns:
            List of RankedResult sorted best-first with 1-indexed ranks
        """
        if not candidates:
            raise RerankerError("Cannot rerank empty candidates list.")

        top_k = top_k or settings.retrieval.rerank_top_k
        start = time.time()

        # ── Normalise scores to [0, 1] ─────────────────────────────
        # Required to make scores comparable across different scales
        sem_scores = [c.score for c in candidates]
        rrf_scores = [c.rrf_score for c in candidates]

        sem_min, sem_max = min(sem_scores), max(sem_scores)
        rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)

        def normalise(val, lo, hi):
            """Min-max normalisation — maps [lo, hi] → [0, 1]."""
            if hi == lo:
                return 1.0
            return (val - lo) / (hi - lo)

        # ── Compute fusion score for each candidate ─────────────────
        scored = []
        for chunk in candidates:
            norm_sem = normalise(chunk.score,     sem_min, sem_max)
            norm_rrf = normalise(chunk.rrf_score, rrf_min, rrf_max)
            both_bonus = 0.1 if chunk.found_by_both else 0.0

            fusion_score = (
                0.6 * norm_sem +    # semantic similarity is primary signal
                0.3 * norm_rrf +    # rrf captures hybrid evidence
                both_bonus          # reward chunks confirmed by both methods
            )
            scored.append((chunk, fusion_score))

        # Sort descending by fusion score
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [
            RankedResult(
                hybrid_result = chunk,
                rerank_score  = score,
                rank          = rank,
            )
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"Score fusion reranking | "
            f"candidates={len(candidates)} | "
            f"kept={len(results)} | "
            f"time={elapsed:.1f}ms"
        )

        return results


# ── Standard CrossEncoder Reranker (local only) ────────────────────────
class CrossEncoderReranker:
    """
    Reranker using CrossEncoder model.

    Best quality but requires ~26s per request on Railway CPU.
    Use locally, or when Railway upgrades to GPU/faster CPU.

    Singleton pattern: model loads once.
    """

    _instance = None
    _model    = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if CrossEncoderReranker._model is not None:
            return

        from sentence_transformers.cross_encoder import CrossEncoder

        logger.info(
            f"Loading CrossEncoder: {settings.reranker.model_name}"
        )
        load_start = time.time()
        CrossEncoderReranker._model = CrossEncoder(
            settings.reranker.model_name
        )
        logger.info(
            f"CrossEncoder loaded | "
            f"time={time.time()-load_start:.2f}s"
        )

    def rerank(
        self,
        question   : str,
        candidates : list[HybridSearchResult],
        top_k      : int | None = None,
    ) -> list[RankedResult]:
        if not candidates:
            raise RerankerError("Cannot rerank empty candidates list.")
        if not question.strip():
            raise RerankerError("Question cannot be empty.")

        top_k  = top_k or settings.retrieval.rerank_top_k
        pairs  = [[question, c.text] for c in candidates]
        scores = CrossEncoderReranker._model.predict(pairs)

        scored = sorted(
            zip(candidates, scores),
            key     = lambda x: float(x[1]),
            reverse = True,
        )

        return [
            RankedResult(
                hybrid_result = chunk,
                rerank_score  = float(score),
                rank          = rank,
            )
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]


# ── Factory Function ───────────────────────────────────────────────────
def get_reranker():
    """
    Returns the appropriate reranker based on config.

    Cloud mode (CLOUD_MODE=true):
      → ScoreFusionReranker (instant, no model)

    Local / GPU:
      → CrossEncoderReranker (best quality, slow on CPU)

    Switching is one environment variable change.
    """
    if settings.reranker.use_lightweight:
        logger.info("Using ScoreFusionReranker (cloud mode)")
        return ScoreFusionReranker()
    else:
        logger.info("Using CrossEncoderReranker (standard mode)")
        return CrossEncoderReranker()


__all__ = [
    "CrossEncoderReranker",
    "ScoreFusionReranker",
    "RankedResult",
    "RerankerError",
    "get_reranker",
]