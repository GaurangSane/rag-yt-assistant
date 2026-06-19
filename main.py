import logging
import time
from contextlib import asynccontextmanager
from src.pipeline import RAGPipeline
from fastapi import FastAPI,HTTPException,requests
from fastapi.middleware.cors import CORSMiddleware
from src.config import validate_environment
from pydantic import BaseModel,field_validator

logger = logging.getLogger("rag_app.api")

_pipeline = RAGPipeline|None = None

@asynccontextmanager
async def fastapi_life(app:FastAPI):
    global _pipeline

    try:
        validate_environment()
        logger("Environment Validate Succesfully.")
    except EnvironmentError as e:
        logger(f"Environment error: {e}")
        raise
    logger("Pipeline loading started")
    load_start = time.time()
    _pipeline = RAGPipeline()
    time_required = time.time() - load_start()
    logger(f"Pipeline loaded | time : {time_required} ")

    yield

    logger("server shutting down")

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


