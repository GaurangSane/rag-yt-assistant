"""
prompt_builder.py
─────────────────
Assembles structured LLM prompts from reranked chunks
and user questions.

Design decisions:
  - Prompts as constants: named, visible, version-controlled
  - System/user split: rules separate from dynamic content
  - PromptPackage dataclass: typed return, not raw strings
  - Timestamp headers: LLM reads and copies citations naturally
  - Hallucination guard: explicit instruction for out-of-scope questions
  - Token estimation: catch oversized prompts before LLM call

Pipeline position:  Reranker → [PromptBuilder] → Generator
"""

import logging
from dataclasses import dataclass, field

from src.config import settings
from src.retrieval.query_transformer import ConversationTurn
from src.retrieval.reranker import RankedResult

logger = logging.getLogger("rag_app.generation.prompt_builder")


# ── Prompts As Constants ─────────────────────────────────────────────
# The system prompt is the permanent ruleset for every response.
# Changing this changes how EVERY answer is generated.
# That is why it lives here as a named, visible constant.

SYSTEM_PROMPT = """\
You are an intelligent assistant that helps users understand \
YouTube videos by answering questions about their content.

RULES YOU MUST ALWAYS FOLLOW:
1. Answer ONLY using the context segments provided below.
   Never use your general training knowledge to answer.

2. ALWAYS cite the timestamp of every segment you use.
   Format your citations naturally:
     "As explained at [5:15]..."
     "According to the video at [8:42]..."
     "The speaker mentions at [2:30] that..."

3. If the answer is not in the provided context segments, respond:
   "The video doesn't appear to cover this specific topic in \
the sections I have access to."
   Never guess. Never make things up.

4. If multiple segments support the answer, reference all of them.

5. Be conversational and clear — you are explaining video content
   to a curious viewer, not writing an academic paper.

6. Keep answers focused and concise while being complete."""

# Separator between context segments in the user prompt.
# Clear visual separation helps the LLM treat each segment
# as a distinct source rather than continuous text.
SEGMENT_SEPARATOR = "\n\n---\n\n"

# Template for each context segment header.
# The LLM reads this header and naturally copies the timestamp
# format into its citations — no extra instruction needed.
SEGMENT_HEADER_TEMPLATE = "[Segment {rank} | {start} → {end}]"


# ── Data Structure ───────────────────────────────────────────────────
@dataclass
class PromptPackage:
    """
    The complete prompt ready for the LLM API call.

    Keeps system and user prompts together as a typed unit.
    The Generator receives this and unpacks it directly into
    the messages list — no string manipulation needed there.

    Attributes:
        system_prompt    : Permanent rules and identity instructions
        user_prompt      : Dynamic content: context + history + question
        estimated_tokens : Rough token count for budget checking
        context_chunks   : How many chunks are in the context
        has_history      : Whether conversation history was included
    """
    system_prompt    : str
    user_prompt      : str
    estimated_tokens : int
    context_chunks   : int
    has_history      : bool

    def to_messages(self) -> list[dict]:
        """
        Format as the messages list the Groq/OpenAI API expects.

        Usage:
            package  = builder.build(...)
            messages = package.to_messages()
            response = groq_client.chat.completions.create(
                messages=messages, ...
            )
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.user_prompt},
        ]

    @property
    def is_within_token_limit(self) -> bool:
        """
        True if estimated tokens fit within model context window.
        Leaves headroom for the generated answer.

        LLaMA3-8b has 8192 token limit.
        We use 6000 as our prompt budget — leaves ~2000 for answers.
        """
        return self.estimated_tokens < 6000


# ── Custom Exception ─────────────────────────────────────────────────
class PromptBuildError(Exception):
    """Raised when prompt assembly fails for a structural reason."""
    pass


# ── Main Class ───────────────────────────────────────────────────────
class PromptBuilder:
    """
    Assembles structured LLM prompts from reranked chunks.

    Takes the typed outputs of the retrieval pipeline
    (RankedResult objects) and assembles them into a clean,
    structured prompt the Generator can send directly to the LLM.

    Usage:
        builder = PromptBuilder()
        package = builder.build(
            question = "how does gradient descent work?",
            results  = ranked_results,
            history  = conversation_history,
        )
        response = llm.complete(package.to_messages())
    """

    def __init__(self):
        logger.info("PromptBuilder initialized")

    def _format_segment(self, result: RankedResult) -> str:
        """
        Format one RankedResult into a labelled context segment.

        The timestamp header is what enables natural citations.
        When the LLM sees:
          [Segment 1 | 5:15 → 6:17]
          gradient descent works by...

        It naturally writes:
          "As explained at [5:15], gradient descent works by..."

        No separate citation instruction needed — the header format
        teaches the model the citation style by example.

        Args:
            result: One RankedResult from CrossEncoderReranker

        Returns:
            Formatted string with header and text
        """
        header = SEGMENT_HEADER_TEMPLATE.format(
            rank  = result.rank,
            start = result.start_time,
            end   = result.end_time,
        )
        return f"{header}\n{result.text.strip()}"

    def _format_context_block(
        self,
        results: list[RankedResult],
    ) -> str:
        """
        Format all reranked results into one labelled context block.

        Segments are separated by SEGMENT_SEPARATOR so the LLM
        treats each as a distinct source rather than continuous text.

        Args:
            results: Top-k RankedResults from reranker

        Returns:
            Multi-segment context string with headers and separators
        """
        segments = [self._format_segment(r) for r in results]
        return SEGMENT_SEPARATOR.join(segments)

    def _format_history_block(
        self,
        history: list[ConversationTurn] | None,
    ) -> str:
        """
        Format recent conversation history for the prompt.

        Includes only the last N turns (config: retrieval.history_turns).
        History enables the LLM to:
          - Resolve "that" / "it" / "the previous answer" references
          - Maintain consistent tone across turns
          - Avoid repeating itself

        Returns empty string if no history — produces cleaner prompt.
        """
        if not history:
            return ""

        recent = history[-settings.retrieval.history_turns:]
        lines  = ["\nCONVERSATION HISTORY:"]

        for turn in recent:
            lines.append(f"User: {turn.question}")
            # Truncate long answers — model only needs enough
            # context to resolve references, not the full answer
            answer = turn.answer
            if len(answer) > 300:
                answer = answer[:300] + "..."
            lines.append(f"Assistant: {answer}")

        lines.append("")   # trailing newline before the question
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from character count.

        Rule of thumb: 1 token ≈ 4 characters for English text.
        This is approximate — actual tokenization varies by model.
        We use it for budget checking, not exact accounting.

        A more accurate approach would use tiktoken, but that adds
        a dependency for minimal accuracy gain at this scale.
        """
        return len(text) // 4

    def build(
        self,
        question : str,
        results  : list[RankedResult],
        history  : list[ConversationTurn] | None = None,
    ) -> PromptPackage:
        """
        Assemble the complete prompt from question, context, and history.

        Three-part structure (in this order):
          1. Context block   — retrieved segments with timestamp headers
          2. History block   — recent conversation turns (if any)
          3. Question        — the user's original question

        Context before question because the LLM reads top-to-bottom.
        Having context loaded into attention before the question arrives
        produces more grounded answers — the model has already "read"
        the sources before deciding how to answer.

        Args:
            question : User's original question (not transformed variants)
            results  : Top-k RankedResults from CrossEncoderReranker
            history  : Previous conversation turns (optional)

        Returns:
            PromptPackage ready for Generator.generate()

        Raises:
            PromptBuildError: If question is empty
            PromptBuildError: If results list is empty
        """
        if not question or not question.strip():
            raise PromptBuildError("Question cannot be empty.")

        if not results:
            raise PromptBuildError(
                "Cannot build prompt with empty results. "
                "Ensure reranking completed successfully."
            )

        question = question.strip()

        logger.info(
            f"Building prompt | "
            f"chunks={len(results)} | "
            f"has_history={history is not None and len(history) > 0}"
        )

        # ── Assemble each block ───────────────────────────────
        context_block = self._format_context_block(results)
        history_block = self._format_history_block(history)

        # ── Assemble user prompt ──────────────────────────────
        # Explicit section labels (CONTEXT FROM VIDEO, QUESTION)
        # reduce ambiguity — the model knows exactly what each
        # section contains without having to infer it.
        user_prompt = (
            f"CONTEXT FROM VIDEO:\n\n"
            f"{context_block}"
            f"{history_block}\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        # ── Token estimation ──────────────────────────────────
        total_tokens = (
            self._estimate_tokens(SYSTEM_PROMPT) +
            self._estimate_tokens(user_prompt)
        )

        package = PromptPackage(
            system_prompt    = SYSTEM_PROMPT,
            user_prompt      = user_prompt,
            estimated_tokens = total_tokens,
            context_chunks   = len(results),
            has_history      = bool(history),
        )

        # Warn if prompt is large — don't raise, let Generator decide
        if not package.is_within_token_limit:
            logger.warning(
                f"Prompt may be too large | "
                f"estimated_tokens={total_tokens} | "
                f"limit=6000"
            )

        logger.info(
            f"Prompt built | "
            f"estimated_tokens={total_tokens} | "
            f"within_limit={package.is_within_token_limit}"
        )

        return package


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "PromptBuilder",
    "PromptPackage",
    "PromptBuildError",
    "SYSTEM_PROMPT",
]