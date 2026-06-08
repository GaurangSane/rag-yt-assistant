"""
query_transformer.py
────────────────────
Rewrites a user's question into multiple search-optimised
query variants using an LLM.

Design decisions:
  - Multi-query generation: 3 variants cast a wider retrieval net
  - Conversation history: resolves vague pronouns before vector search
  - Graceful degradation: API failure falls back to original question
  - Prompt as constant: visible, testable, auditable
  - QueryTransformationResult dataclass: typed return, not raw list

Pipeline position:  [QueryTransformer] → Retriever → Reranker
"""

import logging
import time
from dataclasses import dataclass, field

from groq import Groq

from src.config import settings, GROQ_API_KEY

logger = logging.getLogger("rag_app.retrieval.query_transformer")


# ── Prompts as Constants ─────────────────────────────────────────────
# Prompts are first-class code — named, visible, version-controlled.
# When retrieval quality needs tuning, this is the first thing to edit.

MULTI_QUERY_SYSTEM_PROMPT = """\
You are an expert search query optimiser for a YouTube video Q&A system.

Your task:
  Given a user's question (and optional conversation context),
  generate {n_queries} distinct search queries that together
  cover all angles of what the user is asking.

Rules:
  1. Each query must use DIFFERENT vocabulary and framing
  2. Each query must be SPECIFIC and SELF-CONTAINED
     — no pronouns like "it", "that", "this", "they"
     — a reader with no context must understand each query alone
  3. Queries should be concise: 5-12 words each
  4. Cover different aspects: definition, mechanism, examples, purpose

Output format:
  — Output ONLY the queries, one per line
  — No numbering, no bullet points, no explanation
  — No preamble, no "Here are the queries:", nothing extra
  — Exactly {n_queries} lines, no more, no less\
"""

MULTI_QUERY_USER_TEMPLATE = """\
{context_block}Question: {question}

Generate {n_queries} search queries:\
"""


# ── Data Structures ──────────────────────────────────────────────────
@dataclass
class ConversationTurn:
    """
    One question-answer exchange in a conversation.

    Storing history as typed objects (not raw dicts) means
    the rest of the system cannot accidentally pass malformed
    history — the structure is enforced at the data level.
    """
    question: str
    answer  : str


@dataclass
class QueryTransformationResult:
    """
    The output of one query transformation call.

    Returning a dataclass instead of a plain list lets us
    attach useful metadata — was this a fallback? How long
    did it take? — without changing the caller's interface.

    Attributes:
        queries         : The generated query strings (3 by default)
        original_question: The user's raw question before transformation
        used_fallback   : True if LLM failed and we fell back to original
        latency_ms      : How long the LLM call took in milliseconds
    """
    queries           : list[str]
    original_question : str
    used_fallback     : bool  = False
    latency_ms        : float = 0.0

    @property
    def primary_query(self) -> str:
        """The first (usually best) generated query."""
        return self.queries[0] if self.queries else self.original_question


# ── Custom Exception ─────────────────────────────────────────────────
class QueryTransformationError(Exception):
    """Raised when transformation fails and fallback is also unavailable."""
    pass


# ── Main Class ───────────────────────────────────────────────────────
class QueryTransformer:
    """
    Rewrites user questions into multiple search-optimised queries.

    Uses Groq's LLaMA3 to generate n_queries variants of each
    question, each using different vocabulary to maximise the
    surface area of the vector search.

    Conversation history is included so vague follow-up questions
    like "explain that more simply" are resolved into specific,
    searchable queries before touching the vector DB.

    Usage:
        transformer = QueryTransformer()
        result = transformer.transform(
            question = "how does this work?",
            history  = [ConversationTurn("what is X?", "X is...")]
        )
        print(result.queries)
        # ["how does X function mechanically",
        #  "X operating principles explained",
        #  "step by step process of X"]
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise QueryTransformationError(
                "GROQ_API_KEY not found. "
                "Add it to your .env file and restart."
            )

        self._client    = Groq(api_key=GROQ_API_KEY)
        self._n_queries = settings.retrieval.n_queries
        self._model     = settings.llm.model_name
        self._temp      = settings.llm.temperature_query
        self._max_tokens= settings.llm.max_tokens_query

        logger.info(
            f"QueryTransformer initialized | "
            f"model={self._model} | "
            f"n_queries={self._n_queries}"
        )

    def _build_context_block(
        self,
        history: list[ConversationTurn] | None,
    ) -> str:
        """
        Format recent conversation history into a context string.

        Includes only the last N turns (config: retrieval.history_turns)
        to keep the prompt concise. Older turns are rarely relevant
        to the current question.

        Returns empty string if no history provided.
        """
        if not history:
            return ""

        recent = history[-settings.retrieval.history_turns:]
        lines  = ["Conversation context (for resolving references):"]

        for turn in recent:
            # Truncate long answers — we only need enough context
            # to resolve pronouns, not the full answer
            answer_preview = turn.answer[:200]
            if len(turn.answer) > 200:
                answer_preview += "..."
            lines.append(f"  User: {turn.question}")
            lines.append(f"  Assistant: {answer_preview}")

        lines.append("")   # blank line before the question
        return "\n".join(lines) + "\n"

    def _parse_queries(
        self,
        raw_output      : str,
        original_question: str,
    ) -> list[str]:
        """
        Parse raw LLM output into a clean list of query strings.

        The LLM is prompted to return one query per line with no
        extras — but LLMs are not always perfectly obedient.
        This parser handles common deviations defensively.

        Fallback: original question always appended if parsing
        produces fewer than expected queries.
        """
        lines = [
            line.strip()
            for line in raw_output.strip().split("\n")
            if line.strip()
        ]

        # Clean common LLM formatting artifacts:
        # "1. query text" → "query text"
        # "- query text"  → "query text"
        # "• query text"  → "query text"
        cleaned = []
        for line in lines:
            # Remove leading number+dot: "1. " "2. " etc.
            if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                line = line[2:].strip()
            # Remove leading bullet characters
            if line.startswith(("-", "•", "*", "–")):
                line = line[1:].strip()
            if line:
                cleaned.append(line)

        # Deduplicate while preserving order
        seen    = set()
        unique  = []
        for q in cleaned:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)

        # Always include original question as final fallback
        # Ensures we never have zero queries even if LLM output was garbage
        if original_question.lower() not in seen:
            unique.append(original_question)

        # Return exactly n_queries (or fewer if that's all we have)
        return unique[:self._n_queries]

    def transform(
        self,
        question : str,
        history  : list[ConversationTurn] | None = None,
    ) -> QueryTransformationResult:
        """
        Transform a user question into multiple search queries.

        Graceful degradation: if the LLM call fails for any reason,
        returns the original question as a single-item query list.
        The pipeline continues — with lower quality but without crashing.

        Args:
            question : The user's raw question string
            history  : Previous conversation turns for context resolution

        Returns:
            QueryTransformationResult with generated queries and metadata

        Never raises: all exceptions caught internally, fallback applied
        """
        if not question or not question.strip():
            raise QueryTransformationError(
                "Question cannot be empty."
            )

        question = question.strip()
        logger.info(f"Transforming query | question='{question[:80]}'")

        start_time = time.time()

        try:
            # ── Build prompts ──────────────────────────────────
            context_block = self._build_context_block(history)

            system_prompt = MULTI_QUERY_SYSTEM_PROMPT.format(
                n_queries=self._n_queries
            )
            user_prompt = MULTI_QUERY_USER_TEMPLATE.format(
                context_block = context_block,
                question      = question,
                n_queries     = self._n_queries,
            )

            # ── Call LLM ───────────────────────────────────────
            response = self._client.chat.completions.create(
                model    = self._model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = self._temp,
                max_tokens  = self._max_tokens,
            )

            raw_output = response.choices[0].message.content
            queries    = self._parse_queries(raw_output, question)

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Query transformation complete | "
                f"queries={len(queries)} | "
                f"latency={latency_ms:.0f}ms"
            )
            for i, q in enumerate(queries, 1):
                logger.debug(f"  Query {i}: {q}")

            return QueryTransformationResult(
                queries            = queries,
                original_question  = question,
                used_fallback      = False,
                latency_ms         = latency_ms,
            )

        except Exception as e:
            # ── Graceful degradation ───────────────────────────
            # ANY failure — network, rate limit, parsing — falls back
            # to the original question so the pipeline never crashes
            latency_ms = (time.time() - start_time) * 1000

            logger.warning(
                f"Query transformation failed — using original question "
                f"as fallback | error={type(e).__name__}: {e} | "
                f"latency={latency_ms:.0f}ms"
            )

            return QueryTransformationResult(
                queries           = [question],
                original_question = question,
                used_fallback     = True,
                latency_ms        = latency_ms,
            )


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "QueryTransformer",
    "QueryTransformationResult",
    "ConversationTurn",
    "QueryTransformationError",
]