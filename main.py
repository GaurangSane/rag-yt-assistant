import logging
import time
from contextlib import asynccontextmanager
from src.pipeline import RAGPipeline,ConversationTurn,PipelineError
from fastapi import FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from src.config import validate_environment
from pydantic import BaseModel,field_validator
import asyncio
from functools import partial
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import threading

logger = logging.getLogger("rag_app.api")

limiter = Limiter(key_func=get_remote_address)


_pipeline : RAGPipeline|None = None
_pipeline_ready : bool                = False
_warmup_complete: bool                = False

def _run_warmup_in_background(pipeline: RAGPipeline) -> None:
    global _warmup_complete

    logger.info("Background warmup starting...")
    warmup_start = time.time()

    try:

        pipeline._embedder.embed_query("warmup initialisation query")

        warmup_time = time.time() - warmup_start
        _warmup_complete = True
        logger.info(
            f"Background warmup complete | "
            f"time={warmup_time:.2f}s"
        )

    except Exception as e:
        logger.warning(
            f"Background warmup failed (non-critical): {e}"
        )
        _warmup_complete = True

@asynccontextmanager
async def fastapi_life(app:FastAPI):
    global _pipeline,_pipeline_ready
    logger.info("="*50)
    logger.info("Server starting up...")
    logger.info("="*50)

    try:
        validate_environment()
        logger.info("Environment Validate Succesfully.")
    except EnvironmentError as e:
        logger.critical(f"Environment error: {e}")
        raise
    logger.info("Pipeline loading started")
    load_start = time.time()
    try:
        _pipeline = RAGPipeline()
    except Exception as e:
        logger.critical(f"Pipeline load failed: {e}")
        raise
    time_required = time.time() - load_start
    logger.info(f"Pipeline loaded | time : {time_required} ")

    _pipeline_ready = True

    warmup_thread = threading.Thread(
        target  = _run_warmup_in_background,
        args    = (_pipeline,),
        daemon  = True,   # thread dies when server dies
        name    = "model-warmup",
    )
    warmup_thread.start()
    logger.info(
        "Warmup started in background thread — "
        "server accepting requests immediately"
    )

    # Server runs here
    yield

    # Shutdown
    logger.info("Server shutting down...")

app = FastAPI(
    title="Youtube RAG Assistant API.",
    description=("""Production-grade RAG system for YouTube videos.\n\n"
        "Index any YouTube video and ask questions about its content. "
        "Every answer includes timestamped source citations."""
      ),
      version="1.0.0",
      lifespan=fastapi_life,
      docs_url="/docs",
      redoc_url="/redoc"    
  )

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:8501",   
        "http://localhost:3000",   
        "chrome-extension://*",
    ],
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],

)

class IngestRequest(BaseModel):
    video_url : str
    
    @field_validator("video_url")
    @classmethod
    def check_yt_url(cls,v:str)->str:
        v = v.strip()
        if not v:
            raise ValueError("Video_url can't be empty")
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError(
                "video_url must be a YouTube URL "
                "(youtube.com or youtu.be)"
            )
        return v
    
class IngestResponse(BaseModel):
    video_id : str
    chunk_count : int
    was_cached : bool
    message : str

class ConversationTurnModel(BaseModel):
    question : str
    answer : str

class ChatRequest(BaseModel):
    video_url : str
    question : str
    history : list[ConversationTurnModel] = [] 

    @field_validator("video_url")
    @classmethod
    def check_yt_url(cls,v:str)-> str:
        v = v.strip()
        if not v:
            raise ValueError("Video_url can't be empty")
        if "youtube.com" not in v and "youtube.be" not in v:
            raise ValueError(
                "video_url must be a YouTube URL "
                "(youtube.com or youtu.be)"
            )
        return v       
    
    @field_validator("question")
    @classmethod
    def question_not_empty(clas,ques:str)->str:
        ques = ques.strip()
        if not ques:
            raise ValueError("question cannot be empty")
        return ques


class SourceModel(BaseModel):
    rank        : int
    start_time  : str
    end_time    : str
    youtube_link: str
    display     : str  


class LatencyModel(BaseModel):
    total_ms          : float
    ingestion_ms      : float
    query_transform_ms: float
    retrieval_ms      : float
    reranking_ms      : float
    generation_ms     : float


class ChatResponse(BaseModel):
    answer            : str
    sources           : list[SourceModel]
    queries_used      : list[str]
    video_id          : str
    answer_grounded   : bool
    ingestion_skipped : bool
    latency           : LatencyModel


class HealthResponse(BaseModel):
    status         : str    
    pipeline_loaded: bool
    version        : str


class VideosResponse(BaseModel):
    video_ids: list[str]
    count    : int

async def run_in_executor(func,*args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(func,*args)
    )

@app.exception_handler(PipelineError)
async def pipeline_error_handler(
    request : Request,
    exc : PipelineError
):
    logger.error(
        f"PipelineError in {request.method} {request.url.path} | "
        f"step={exc.step} | error={str(exc)}"
    )
    return JSONResponse(
        status_code = 422,
        content     = {
            "error"  : "pipeline_error",
            "step"   : exc.step,
            "detail" : str(exc),
        }
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description=("Returns server status. Used by deployment platforms "
        "to verify the service is running."
)
)
async def health_check()->HealthResponse:
    return HealthResponse(
        status          = "healthy" if _pipeline_ready else "starting",
        pipeline_loaded = _pipeline_ready,
        warmup_complete = _warmup_complete,
        version         = app.version,
    )

@app.get("/ready")
async def readiness_check():
    if not _pipeline_ready:
        return JSONResponse(
            status_code = 503,
            content     = {"ready": False, "reason": "Pipeline loading"},
        )
    return {"ready": True, "warmup_complete": _warmup_complete}

@app.get(
    "/videos",
    response_model=VideosResponse,
    summary="list Indexed videos",
    description="Returns all video IDs currently in the vector store."

)
async def get_video()->VideosResponse:
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return VideosResponse(
        video_ids = _pipeline.list_indexed_videos(),
        count     = len(_pipeline.list_indexed_videos()),
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post(
    "/ingest",
    response_model = IngestResponse,
    summary="Index a YouTube video",
    description=(
       "Fetches the transcript, chunks it, embeds it, and stores "
        "it in the vector database. Idempotent — safe to call multiple "
        "times on the same video."
    ),
    status_code=200
)
@limiter.limit("5/minute")
async def ingest_video(
    request : Request,
    payload : IngestRequest)->IngestResponse:
    if not _pipeline:
        raise HTTPException(
            status_code=503,
            detail="pipeline not loaded"
        ) 
    url = payload.video_url

    if _pipeline.is_video_indexed(url):
        video_id = _pipeline._fetcher.extract_video_id(url)
        chunk_count = _pipeline._store.count(video_id)

        logger.info(
            f"POST /ingest | cached | video_id={video_id}"
        )

        return IngestResponse(
            video_id=video_id,
            chunk_count=chunk_count,
            was_cached=True,
            message=f"Video already indexed with {chunk_count} chunks."
        )
    
    logger.info(f"POST /ingest | Starting | {url} ")
    start_time = time.time()

    try:
        response = await run_in_executor(
            _pipeline.query,
            url,
            "What is this video about"
        )
    except PipelineError:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed | url:{url} | error:{e}")
        raise HTTPException(
            status_code=503,
            detail=f"ingestion failed: error:{e}"
        )     
    ingest_time = time.time() - start_time
    chunk_count = _pipeline._store.count(response.video_id)   

    logger.info(
        f"POST /ingest | Complete |",
        f"video ID = {response.video_id}",
        f"chunk_count = {chunk_count}",
        f"time = {ingest_time:.1f}s"
    )

    return IngestResponse(
        video_id=response.video_id,
        chunk_count=chunk_count,
        was_cached=False,
        message=(
             f"Successfully indexed {chunk_count} chunks "
            f"in {ingest_time:.1f}s."
        )
    )    

@app.post(
    "/chat",
    response_model = ChatResponse,
    summary        = "Ask a question about a video",
    description    = (
        "Runs the full RAG pipeline: query transformation → "
        "hybrid retrieval → reranking → generation. "
        "Returns a grounded answer with timestamp citations."
    ),
)
@limiter.limit("10/minute")
async def chat(request: Request,
    payload: ChatRequest) -> ChatResponse:
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")


    history = [
        ConversationTurn(
            question = turn.question,
            answer   = turn.answer,
        )
        for turn in payload.history
    ]

    logger.info(
        f"POST /chat | "
        f"video_id={payload.video_url[-11:]} | "
        f"question='{payload.question[:60]}' | "
        f"history_turns={len(history)}"
    )

    try:
        response = await run_in_executor(
            _pipeline.query,
            payload.video_url,
            payload.question,
            history or None,
        )
    except PipelineError:
        raise   
    except Exception as e:
        logger.error(f"Chat failed | error={e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Chat failed: {str(e)}",
        )

    return ChatResponse(
        answer            = response.answer,
        answer_grounded   = response.answer_grounded,
        ingestion_skipped = response.ingestion_skipped,
        video_id          = response.video_id,
        queries_used      = response.queries_used,
        sources           = [
            SourceModel(
                rank         = s.rank,
                start_time   = s.start_time,
                end_time     = s.end_time,
                youtube_link = s.youtube_link,
                display      = s.display,
            )
            for s in response.sources
        ],
        latency = LatencyModel(
            total_ms           = response.latency.total_ms,
            ingestion_ms       = response.latency.ingestion_ms,
            query_transform_ms = response.latency.query_transform_ms,
            retrieval_ms       = response.latency.retrieval_ms,
            reranking_ms       = response.latency.reranking_ms,
            generation_ms      = response.latency.generation_ms,
        ),
    )
