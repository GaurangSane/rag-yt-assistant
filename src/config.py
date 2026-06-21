import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db"))
)
LOG_DIR     = ROOT_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LOG_LEVEL  = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_FILE   = LOG_DIR / "rag_app.log"

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level   = LOG_LEVEL,
        format  = LOG_FORMAT,
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ]
    )
    return logging.getLogger("rag_app")

@dataclass
class EmbeddingConfig:
    model_name           : str = "all-mpnet-base-v2"
    dimensions           : int = 768
    batch_size           : int = 32
    normalize_embeddings : bool = True


@dataclass
class LLMConfig:
    model_name          : str   = "llama-3.1-8b-instant"
    temperature_query   : float = 0.7  
    temperature_answer  : float = 0.3   
    max_tokens_query    : int   = 200
    max_tokens_answer   : int   = 1024


@dataclass
class RerankerConfig:
    model_name : str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class ChunkConfig:
    window_sec  : int = 60    
    overlap_sec : int = 15    

@dataclass
class RetrievalConfig:
    n_queries        : int = 3   
    retrieve_top_k   : int = 5   
    rerank_top_k     : int = 3   
    rrf_k            : int = 60  
    history_turns    : int = 3   

@dataclass
class VectorDBConfig:
    persist_dir      : Path  = CHROMA_DIR
    distance_metric  : str   = "cosine"
    collection_prefix: str   = "yt_"

@dataclass
class Settings:
    embedding  : EmbeddingConfig  = field(default_factory=EmbeddingConfig)
    llm        : LLMConfig        = field(default_factory=LLMConfig)
    reranker   : RerankerConfig   = field(default_factory=RerankerConfig)
    chunking   : ChunkConfig      = field(default_factory=ChunkConfig)
    retrieval  : RetrievalConfig  = field(default_factory=RetrievalConfig)
    vector_db  : VectorDBConfig   = field(default_factory=VectorDBConfig)


settings = Settings()
logger   = setup_logging()

def validate_environment() -> None:
    required = {
        "GROQ_API_KEY": GROQ_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            f"Check your .env file in the project root."
        )
    logger.info("Environment validation passed")

__all__ = [
    "settings",
    "logger",
    "validate_environment",
    "ROOT_DIR",
    "DATA_DIR",
    "CHROMA_DIR",
]