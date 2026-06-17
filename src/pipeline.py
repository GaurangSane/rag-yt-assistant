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



@dataclass
class SourceCitation:

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

    ingestion_ms       : float = 0.0   
    query_transform_ms : float = 0.0   
    retrieval_ms       : float = 0.0   
    reranking_ms       : float = 0.0   
    prompt_build_ms    : float = 0.0   
    generation_ms      : float = 0.0   
    total_ms           : float = 0.0   

    def log_summary(self) -> None:

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
    answer            : str
    sources           : list[SourceCitation]
    queries_used      : list[str]
    video_id          : str
    latency           : PipelineLatency
    ingestion_skipped : bool
    answer_grounded : bool = True

    @property
    def has_answer(self) -> bool:
        return bool(self.answer and self.answer.strip())

    @property
    def citation_count(self) -> int:
        return len(self.sources)

    def to_dict(self) -> dict:
        return {
            "answer"            : self.answer,
            "answer_grounded" : self.answer_grounded,
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


class PipelineError(Exception):
    def __init__(self, step: str, message: str, cause: Exception = None):
        self.step    = step
        super().__init__(f"[Step {step}] {message}")
        self.__cause__ = cause


class RAGPipeline:
    
    def __init__(self, persist_dir: str | None = None):

        logger.info("Initialising RAGPipeline...")
        init_start = time.time()

        validate_environment()


        self._fetcher  = YoutubeTranscriptFetcher()
        self._chunker  = TranscriptChunker()
        self._embedder = EmbeddingModel()         

        self._store = VectorStore(persist_dir=persist_dir)

        self._transformer = QueryTransformer()
        self._retriever   = HybridRetriever(
            store           = self._store,
            embedding_model = self._embedder,
        )
        self._reranker    = CrossEncoderReranker()  

        self._prompt_builder = PromptBuilder()
        self._generator      = LLMGenerator()

        init_time = (time.time() - init_start) * 1000
        logger.info(
            f"RAGPipeline ready | init_time={init_time:.0f}ms"
        )

    def _run_ingestion(
        self,
        youtube_url: str,
    ) -> tuple[str, int]:

        logger.info(f"Running ingestion | url={youtube_url}")
        ingestion_start = time.time()

        try:
            video_id, segments = self._fetcher.fetch(youtube_url)
            logger.info(f"Step 1 complete | segments={len(segments)}")
        except Exception as e:
            raise PipelineError(
                "1 (Transcript)", str(e), e
            ) from e

        try:
            chunks = self._chunker.chunk(segments, video_id)
            logger.info(f"Step 2 complete | chunks={len(chunks)}")
        except Exception as e:
            raise PipelineError(
                "2 (Chunker)", str(e), e
            ) from e
        try:
            embedded = self._embedder.embed(chunks)
            logger.info(f"Step 3 complete | embedded={len(embedded)}")
        except Exception as e:
            raise PipelineError(
                "3 (Embedder)", str(e), e
            ) from e

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
    def query(
        self,
        youtube_url : str,
        question    : str,
        history     : list[ConversationTurn] | None = None,
    ) -> RAGResponse:
    
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

        try:
            video_id = self._fetcher.extract_video_id(youtube_url)
        except Exception as e:
            raise PipelineError("1 (URL Parse)", str(e), e) from e


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

        GUARD_PHRASES = [
            "doesn't appear to cover",
            "does not appear to cover",
            "not mentioned in",
            "not covered in",
            "no information about",
            "cannot find",
            "not discussed",
        ]

        answer_lower = gen_result.answer.lower()
        answer_grounded = not any(
            phrases in answer_lower
            for phrases in GUARD_PHRASES
        )


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
            answer_grounded=answer_grounded
        )

    def is_video_indexed(self, youtube_url: str) -> bool:
    
        try:
            video_id = self._fetcher.extract_video_id(youtube_url)
            return self._store.exists(video_id)
        except Exception:
            return False

    def delete_video(self, youtube_url: str) -> None:
        
        video_id = self._fetcher.extract_video_id(youtube_url)
        self._store.delete(video_id)
        self._retriever.invalidate_cache(video_id)
        logger.info(f"Video deleted from pipeline | video_id={video_id}")

    def list_indexed_videos(self) -> list[str]:
        
        return self._store.list_videos()



__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "SourceCitation",
    "PipelineLatency",
    "PipelineError",
    "ConversationTurn",
]