"""
config.py
─────────
Central configuration for the YT RAG Assistant.

CLOUD_MODE=true in environment enables:
  - Batch query embedding (3 calls → 1 call)
  - Lightweight score-fusion reranking (no CrossEncoder)
  - Reduced query variants (3 → 2)
  - Longer timeouts for Railway free tier
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db")))
LOG_DIR    = ROOT_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY", "")

# ── Cloud Mode Detection ───────────────────────────────────────────────
# Set CLOUD_MODE=true in Railway environment variables.
# Automatically enables performance optimisations for CPU-only cloud.
CLOUD_MODE = os.getenv("CLOUD_MODE", "false").lower() == "true"

# ── Logging ────────────────────────────────────────────────────────────
import sys

LOG_LEVEL  = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_FILE   = LOG_DIR / "rag_app.log"

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level    = LOG_LEVEL,
        format   = LOG_FORMAT,
        handlers = [
            logging.StreamHandler(sys.stdout),  # stdout → correct severity in Railway
            logging.FileHandler(LOG_FILE),
        ]
    )
    return logging.getLogger("rag_app")


# ── Model Configuration ────────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    model_name           : str  = "all-MiniLM-L6-v2"
    dimensions           : int  = 384
    batch_size           : int  = 32
    normalize_embeddings : bool = True


@dataclass
class LLMConfig:
    model_name         : str   = "llama-3.1-8b-instant"
    temperature_query  : float = 0.7
    temperature_answer : float = 0.3
    max_tokens_query   : int   = 200
    max_tokens_answer  : int   = 1024


@dataclass
class RerankerConfig:
    model_name   : str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # When True: skip CrossEncoder, use score fusion instead
    # Automatically set based on CLOUD_MODE
    use_lightweight: bool = field(default_factory=lambda: CLOUD_MODE)


@dataclass
class ChunkConfig:
    window_sec  : int = 60
    overlap_sec : int = 15


@dataclass
class RetrievalConfig:
    # Cloud mode uses 2 queries instead of 3
    # Still gets good coverage with half the embedding time
    n_queries       : int   = 2 if CLOUD_MODE else 3
    retrieve_top_k  : int   = 5
    rerank_top_k    : int   = 3
    rrf_k           : int   = 60
    history_turns   : int   = 3


@dataclass
class VectorDBConfig:
    persist_dir      : Path = CHROMA_DIR
    distance_metric  : str  = "cosine"
    collection_prefix: str  = "yt_"


@dataclass
class Settings:
    embedding : EmbeddingConfig  = field(default_factory=EmbeddingConfig)
    llm       : LLMConfig        = field(default_factory=LLMConfig)
    reranker  : RerankerConfig   = field(default_factory=RerankerConfig)
    chunking  : ChunkConfig      = field(default_factory=ChunkConfig)
    retrieval : RetrievalConfig  = field(default_factory=RetrievalConfig)
    vector_db : VectorDBConfig   = field(default_factory=VectorDBConfig)


settings = Settings()
logger   = setup_logging()


def validate_environment() -> None:
    required = {"GROQ_API_KEY": GROQ_API_KEY}
    missing  = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}"
        )
    logger.info(
        f"Environment validation passed | "
        f"cloud_mode={CLOUD_MODE} | "
        f"n_queries={settings.retrieval.n_queries} | "
        f"lightweight_rerank={settings.reranker.use_lightweight}"
    )


__all__ = [
    "settings", "logger", "validate_environment",
    "ROOT_DIR", "DATA_DIR", "CHROMA_DIR", "CLOUD_MODE",
    "GROQ_API_KEY", "SUPADATA_API_KEY",
]