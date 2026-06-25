"""
retrieval package
─────────────────
World 2: runs on every user question.
  query transformation → hybrid retrieval → reranking
"""

from src.retrieval.query_transformer import (
    QueryTransformer,
    QueryTransformationResult,
    ConversationTurn,
    QueryTransformationError,
)
from src.retrieval.retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrieverError,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    ScoreFusionReranker,
    RankedResult,
    RerankerError,
    get_reranker,
)

__all__ = [
    "QueryTransformer",
    "QueryTransformationResult",
    "ConversationTurn",
    "QueryTransformationError",
    "HybridRetriever",
    "HybridSearchResult",
    "RetrieverError",
    "CrossEncoderReranker",
    "RankedResult",
    "RerankerError",
]