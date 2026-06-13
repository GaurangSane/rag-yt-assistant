"""
pipeline.py
───────────
Orchestrates all 9 steps of the RAG pipeline into one
clean, typed function call.

This is the single entry point for the entire system.
The Streamlit UI, FastAPI backend, and tests all call
run_pipeline() — nothing else from this codebase.

Design decisions:
  - RAGPipeline class: holds all module instances as attributes
  - Lazy singleton modules: EmbeddingModel, CrossEncoderReranker
    load once and are reused across every pipeline call
  - Smart re-ingestion: videos already in VectorStore skip Steps 1-4
  - RAGResponse dataclass: typed public contract for all consumers
  - Full latency tracking: every stage timed independently

Pipeline steps:
  INGESTION  (once per video):
    1. Fetch transcript          → YouTubeTranscriptFetcher
    2. Chunk transcript          → TranscriptChunker
    3. Embed chunks              → EmbeddingModel
    4. Store in vector DB        → VectorStore

  RETRIEVAL + GENERATION (every question):
    5. Transform query           → QueryTransformer
    6. Hybrid retrieval          → HybridRetriever
    7. Rerank candidates         → CrossEncoderReranker
    8. Build prompt              → PromptBuilder
    9. Generate answer           → LLMGenerator
"""

import logging
import time
from dataclasses import dataclass, field

from src.config import settings, validate_environment
from src.ingestion.transcript  import YoutubeTranscriptFetcher
from src.ingestion.chunker     import TranscriptChunker
from src.ingestion.embedder    import EmbeddingModel
from src.storage.vector_store  import VectorStore
from src.retrieval.query_transformer import (
    QueryTransformer,
    ConversationTurn,
)
from src.retrieval.retriever   import HybridRetriever
from src.retrieval.reranker    import CrossEncoderReranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator  import LLMGenerator

logger = logging.getLogger("rag_app.pipeline")


# ── Public Data Structures ───────────────────────────────────────────
@dataclass
class SourceCitation:
    """
    One timestamped source used in the answer.

    Shown to the user as a clickable citation below the answer.
    The youtube_link opens the video at the exact timestamp.
    """
    rank        : int
    start_time  : str
    end_time    : str
    youtube_link: str
    chunk_id    : str
    rerank_score: float

    @property
    def display(self) -> str:
        """Human-readable citation string for UI display."""
        return f"[{self.start_time} → {self.end_time}]"


@dataclass
class PipelineLatency:
    """
    Per-stage timing breakdown for the full pipeline run.

    Useful for identifying bottlenecks and monitoring performance.
    All values in milliseconds.
    """
    ingestion_ms       : float = 0.0   # Steps 1-4 (skipped if cached)
    query_transform_ms : float = 0.0   # Step 5
    retrieval_ms       : float = 0.0   # Step 6
    reranking_ms       : float = 0.0   # Step 7
    prompt_build_ms    : float = 0.0   # Step 8
    generation_ms      : float = 0.0   # Step 9
    total_ms           : float = 0.0   # wall clock for entire call

    def log_summary(self) -> None:
        """Log a formatted timing summary at INFO level."""
        logger.info(
            f"Pipeline latency breakdown | "
            f"total={self.total_ms:.0f}ms | "
            f"ingestion={self.ingestion_ms:.0f}ms | "
            f"transform={self.query_transform_ms:.0f}ms | "
            f"retrieval={self.retrieval_ms:.0f}ms | "
            f"rerank={self.reranking_ms:.0f}ms | "
            f"prompt={self.prompt_build_ms:.0f}ms | "
            f"generation={self.generation_ms:.0f}ms"
        )


@dataclass
class RAGResponse:
    """
    The public contract returned by RAGPipeline.query().

    This is what the Streamlit UI, FastAPI backend, and
    tests receive. Every field is deliberately chosen —
    adding is safe, removing or renaming breaks consumers.

    Attributes:
        answer            : The generated answer text with citations
        sources           : Timestamped chunks used in the answer
        queries_used      : The transformed queries (for debugging)
        video_id          : Which video was queried
        latency           : Per-stage timing breakdown
        ingestion_skipped : True if video was already in VectorStore
    """
    answer            : str
    sources           : list[SourceCitation]
    queries_used      : list[str]
    video_id          : str
    latency           : PipelineLatency
    ingestion_skipped : bool

    @property
    def has_answer(self) -> bool:
        """True if the answer has actual content."""
        return bool(self.answer and self.answer.strip())

    @property
    def citation_count(self) -> int:
        """How many source chunks were used in the answer."""
        return len(self.sources)

    def to_dict(self) -> dict:
        """
        Serialize for JSON responses (FastAPI) or storage.

        Returns plain Python types only — no dataclass nesting
        so json.dumps() works without a custom encoder.
        """
        return {
            "answer"            : self.answer,
            "sources"           : [
                {
                    "rank"        : s.rank,
                    "start_time"  : s.start_time,
                    "end_time"    : s.end_time,
                    "youtube_link": s.youtube_link,
                    "display"     : s.display,
                }
                for s in self.sources
            ],
            "queries_used"      : self.queries_used,
            "video_id"          : self.video_id,
            "ingestion_skipped" : self.ingestion_skipped,
            "latency_ms"        : {
                "total"           : self.latency.total_ms,
                "ingestion"       : self.latency.ingestion_ms,
                "query_transform" : self.latency.query_transform_ms,
                "retrieval"       : self.latency.retrieval_ms,
                "reranking"       : self.latency.reranking_ms,
                "generation"      : self.latency.generation_ms,
            },
        }


# ── Custom Exception ─────────────────────────────────────────────────
class PipelineError(Exception):
    """
    Raised when the pipeline fails at any step.

    Wraps the underlying exception with step context so logs
    immediately show which step failed — not just that something did.
    """
    def __init__(self, step: str, message: str, cause: Exception = None):
        self.step    = step
        super().__init__(f"[Step {step}] {message}")
        self.__cause__ = cause


# ── Main Class ───────────────────────────────────────────────────────
class RAGPipeline:
    """
    Orchestrates all 9 steps of the YouTube RAG pipeline.

    Instantiate once and reuse for multiple queries — all
    expensive resources (models, DB client) are loaded once
    and held as instance attributes.

    Usage:
        pipeline = RAGPipeline()

        # First question on a new video — runs all 9 steps
        response = pipeline.query(
            youtube_url = "https://youtube.com/watch?v=abc",
            question    = "how does attention work?",
        )

        # Follow-up question — ingestion skipped (video cached)
        response = pipeline.query(
            youtube_url = "https://youtube.com/watch?v=abc",
            question    = "can you explain that more simply?",
            history     = [ConversationTurn(...)]
        )

        print(response.answer)
        for source in response.sources:
            print(source.display, source.youtube_link)
    """

    def __init__(self, persist_dir: str | None = None):
        """
        Initialise all pipeline modules.

        Heavy models (EmbeddingModel, CrossEncoderReranker) use
        the singleton pattern — they load once regardless of how
        many RAGPipeline instances are created.

        Args:
            persist_dir: ChromaDB storage directory.
                         Defaults to config value (./chroma_db).
                         Pass a custom path in tests.
        """
        logger.info("Initialising RAGPipeline...")
        init_start = time.time()

        # Validate environment before doing any work
        # Fail fast with clear message if API keys are missing
        validate_environment()

        # ── Ingestion modules ──────────────────────────────────
        self._fetcher  = YoutubeTranscriptFetcher()
        self._chunker  = TranscriptChunker()
        self._embedder = EmbeddingModel()         # singleton

        # ── Storage ────────────────────────────────────────────
        self._store = VectorStore(persist_dir=persist_dir)

        # ── Retrieval modules ──────────────────────────────────
        self._transformer = QueryTransformer()
        self._retriever   = HybridRetriever(
            store           = self._store,
            embedding_model = self._embedder,
        )
        self._reranker    = CrossEncoderReranker()  # singleton

        # ── Generation modules ─────────────────────────────────
        self._prompt_builder = PromptBuilder()
        self._generator      = LLMGenerator()

        init_time = (time.time() - init_start) * 1000
        logger.info(
            f"RAGPipeline ready | init_time={init_time:.0f}ms"
        )

    # ── Ingestion ────────────────────────────────────────────────────
    def _run_ingestion(
        self,
        youtube_url: str,
    ) -> tuple[str, int]:
        """
        Run Steps 1-4: fetch, chunk, embed, store.

        Called only when the video is NOT already in VectorStore.

        Args:
            youtube_url: Full YouTube URL

        Returns:
            Tuple of (video_id, chunk_count)

        Raises:
            PipelineError: With step label if any stage fails
        """
        logger.info(f"Running ingestion | url={youtube_url}")
        ingestion_start = time.time()

        # Step 1 — Fetch transcript
        try:
            video_id, segments = self._fetcher.fetch(youtube_url)
            logger.info(f"Step 1 complete | segments={len(segments)}")
        except Exception as e:
            raise PipelineError(
                "1 (Transcript)", str(e), e
            ) from e

        # Step 2 — Chunk transcript
        try:
            chunks = self._chunker.chunk(segments, video_id)
            logger.info(f"Step 2 complete | chunks={len(chunks)}")
        except Exception as e:
            raise PipelineError(
                "2 (Chunker)", str(e), e
            ) from e

        # Step 3 — Embed chunks
        try:
            embedded = self._embedder.embed(chunks)
            logger.info(f"Step 3 complete | embedded={len(embedded)}")
        except Exception as e:
            raise PipelineError(
                "3 (Embedder)", str(e), e
            ) from e

        # Step 4 — Store in VectorDB
        try:
            count = self._store.save(embedded, video_id)
            logger.info(f"Step 4 complete | stored={count}")
        except Exception as e:
            raise PipelineError(
                "4 (VectorStore)", str(e), e
            ) from e

        ingestion_ms = (time.time() - ingestion_start) * 1000
        logger.info(
            f"Ingestion complete | "
            f"video_id={video_id} | "
            f"chunks={count} | "
            f"time={ingestion_ms:.0f}ms"
        )

        return video_id, count

    # ── Main Public Method ────────────────────────────────────────────
    def query(
        self,
        youtube_url : str,
        question    : str,
        history     : list[ConversationTurn] | None = None,
    ) -> RAGResponse:
        """
        Run the complete RAG pipeline for one user question.

        Smart re-ingestion:
          - If video already in VectorStore → skip Steps 1-4
          - If video is new → run full ingestion first

        This is the ONLY public method consumers should call.
        Everything else is an implementation detail.

        Args:
            youtube_url : Any valid YouTube video URL
            question    : The user's question about the video
            history     : Previous conversation turns for follow-ups
                          Pass [] or None for the first question

        Returns:
            RAGResponse with answer, sources, and metadata

        Raises:
            PipelineError: If any step fails (includes step label)
        """
        if not question or not question.strip():
            raise PipelineError(
                "Input", "Question cannot be empty."
            )

        if not youtube_url or not youtube_url.strip():
            raise PipelineError(
                "Input", "YouTube URL cannot be empty."
            )

        logger.info(
            f"Pipeline query | "
            f"url={youtube_url} | "
            f"question='{question[:80]}'"
        )

        wall_start = time.time()
        latency    = PipelineLatency()

        # ── Determine video_id ────────────────────────────────
        try:
            video_id = self._fetcher.extract_video_id(youtube_url)
        except Exception as e:
            raise PipelineError("1 (URL Parse)", str(e), e) from e

        # ── Steps 1-4: Ingestion (skipped if video cached) ────
        ingestion_skipped = self._store.exists(video_id)
        ingestion_start   = time.time()

        if ingestion_skipped:
            logger.info(
                f"Video already indexed — skipping ingestion | "
                f"video_id={video_id}"
            )
        else:
            self._run_ingestion(youtube_url)

        latency.ingestion_ms = (time.time() - ingestion_start) * 1000

        # ── Step 5: Query Transformation ─────────────────────
        step5_start = time.time()
        try:
            transform_result = self._transformer.transform(
                question = question,
                history  = history,
            )
            queries = transform_result.queries
            logger.info(
                f"Step 5 complete | "
                f"queries={len(queries)} | "
                f"fallback={transform_result.used_fallback}"
            )
        except Exception as e:
            raise PipelineError("5 (QueryTransformer)", str(e), e) from e
        latency.query_transform_ms = (time.time() - step5_start) * 1000

        # ── Step 6: Hybrid Retrieval ──────────────────────────
        step6_start = time.time()
        try:
            candidates = self._retriever.retrieve(
                queries  = queries,
                video_id = video_id,
                top_k    = settings.retrieval.retrieve_top_k,
            )
            logger.info(
                f"Step 6 complete | candidates={len(candidates)}"
            )
        except Exception as e:
            raise PipelineError("6 (Retriever)", str(e), e) from e
        latency.retrieval_ms = (time.time() - step6_start) * 1000

        # ── Step 7: Reranking ─────────────────────────────────
        step7_start = time.time()
        try:
            ranked = self._reranker.rerank(
                question   = question,
                candidates = candidates,
                top_k      = settings.retrieval.rerank_top_k,
            )
            logger.info(
                f"Step 7 complete | ranked={len(ranked)}"
            )
        except Exception as e:
            raise PipelineError("7 (Reranker)", str(e), e) from e
        latency.reranking_ms = (time.time() - step7_start) * 1000

        # ── Step 8: Prompt Building ───────────────────────────
        step8_start = time.time()
        try:
            prompt_package = self._prompt_builder.build(
                question = question,
                results  = ranked,
                history  = history,
            )
            logger.info(
                f"Step 8 complete | "
                f"tokens~{prompt_package.estimated_tokens}"
            )
        except Exception as e:
            raise PipelineError("8 (PromptBuilder)", str(e), e) from e
        latency.prompt_build_ms = (time.time() - step8_start) * 1000

        # ── Step 9: LLM Generation ────────────────────────────
        step9_start = time.time()
        try:
            gen_result = self._generator.generate(prompt_package)
            logger.info(
                f"Step 9 complete | "
                f"words={gen_result.word_count} | "
                f"latency={gen_result.latency_ms:.0f}ms"
            )
        except Exception as e:
            raise PipelineError("9 (Generator)", str(e), e) from e
        latency.generation_ms = (time.time() - step9_start) * 1000

        # ── Assemble response ─────────────────────────────────
        latency.total_ms = (time.time() - wall_start) * 1000
        latency.log_summary()

        sources = [
            SourceCitation(
                rank         = r.rank,
                start_time   = r.start_time,
                end_time     = r.end_time,
                youtube_link = r.youtube_link,
                chunk_id     = r.chunk_id,
                rerank_score = r.rerank_score,
            )
            for r in ranked
        ]

        return RAGResponse(
            answer            = gen_result.answer,
            sources           = sources,
            queries_used      = queries,
            video_id          = video_id,
            latency           = latency,
            ingestion_skipped = ingestion_skipped,
        )

    # ── Utility Methods ───────────────────────────────────────────────
    def is_video_indexed(self, youtube_url: str) -> bool:
        """
        Check if a video is already in the VectorStore.

        Useful for UI to show 'already indexed' vs 'indexing...' state.
        """
        try:
            video_id = self._fetcher.extract_video_id(youtube_url)
            return self._store.exists(video_id)
        except Exception:
            return False

    def delete_video(self, youtube_url: str) -> None:
        """
        Remove a video from the VectorStore and BM25 cache.

        Call this to force re-ingestion with new parameters.
        """
        video_id = self._fetcher.extract_video_id(youtube_url)
        self._store.delete(video_id)
        self._retriever.invalidate_cache(video_id)
        logger.info(f"Video deleted from pipeline | video_id={video_id}")

    def list_indexed_videos(self) -> list[str]:
        """Return all video IDs currently stored in VectorStore."""
        return self._store.list_videos()


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "SourceCitation",
    "PipelineLatency",
    "PipelineError",
    "ConversationTurn",
]