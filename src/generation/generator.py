"""
generator.py
────────────
Calls the Groq LLM with the assembled prompt and returns
the generated answer with metadata.

Design decisions:
  - GenerationResult dataclass: answer + metadata together
  - Temperature 0.3: factual consistency over creativity
  - Retry logic: transient API errors retried once automatically
  - Latency tracking: every call timed for performance monitoring
  - Clean separation: Generator knows nothing about prompts —
    it receives a PromptPackage and returns a GenerationResult

Pipeline position:  PromptBuilder → [Generator] → pipeline.py
"""

import logging
import time
from dataclasses import dataclass

from groq import Groq

from src.config import settings, GROQ_API_KEY
from src.generation.prompt_builder import PromptPackage

logger = logging.getLogger("rag_app.generation.generator")


# ── Data Structure ───────────────────────────────────────────────────
@dataclass
class GenerationResult:
    """
    The LLM's response with metadata for the pipeline.

    Attributes:
        answer        : The generated text answer with citations
        latency_ms    : How long the API call took in milliseconds
        model         : Which LLM model was used
        prompt_tokens : Estimated tokens sent to the model
        was_retried   : True if a transient error caused one retry
    """
    answer        : str
    latency_ms    : float
    model         : str
    prompt_tokens : int
    was_retried   : bool = False

    @property
    def is_empty(self) -> bool:
        """True if the model returned empty or whitespace-only text."""
        return not self.answer or not self.answer.strip()

    @property
    def word_count(self) -> int:
        """Approximate word count of the answer."""
        return len(self.answer.split()) if self.answer else 0


# ── Custom Exception ─────────────────────────────────────────────────
class GenerationError(Exception):
    """Raised when the LLM call fails after retries."""
    pass


# ── Main Class ───────────────────────────────────────────────────────
class LLMGenerator:
    """
    Sends assembled prompts to the Groq LLM and returns answers.

    Knows nothing about prompt structure — receives PromptPackage
    from PromptBuilder and returns GenerationResult. Clean separation
    between assembly (PromptBuilder) and execution (LLMGenerator).

    Retry behaviour:
      - First failure: waits 1 second, retries once
      - Second failure: raises GenerationError
      - This handles transient rate limits and network blips

    Usage:
        generator = LLMGenerator()
        result    = generator.generate(prompt_package)
        print(result.answer)
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise GenerationError(
                "GROQ_API_KEY not found. "
                "Add it to your .env file and restart."
            )

        self._client     = Groq(api_key=GROQ_API_KEY)
        self._model      = settings.llm.model_name
        self._temp       = settings.llm.temperature_answer
        self._max_tokens = settings.llm.max_tokens_answer

        logger.info(
            f"LLMGenerator initialized | "
            f"model={self._model} | "
            f"temperature={self._temp}"
        )

    def _call_api(self, messages: list[dict]) -> str:
        """
        Make one API call and return the response text.

        Private method — callers use generate() which adds
        retry logic and result packaging on top of this.
        """
        response = self._client.chat.completions.create(
            model       = self._model,
            messages    = messages,
            temperature = self._temp,
            max_tokens  = self._max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate(self, prompt_package: PromptPackage) -> GenerationResult:
        """
        Generate an answer from a PromptPackage.

        Sends the assembled prompt to Groq and returns the
        model's response as a typed GenerationResult.

        Automatically retries once on transient failures
        (network errors, rate limit blips). Raises on second failure.

        Args:
            prompt_package: Output from PromptBuilder.build()

        Returns:
            GenerationResult with the answer and call metadata

        Raises:
            GenerationError: If API call fails after one retry
        """
        if prompt_package is None:
            raise GenerationError("PromptPackage cannot be None.")

        # Warn if prompt is large — answer quality may degrade
        if not prompt_package.is_within_token_limit:
            logger.warning(
                f"Prompt exceeds recommended token limit | "
                f"estimated={prompt_package.estimated_tokens}"
            )

        messages   = prompt_package.to_messages()
        was_retried = False

        logger.info(
            f"Generating answer | "
            f"model={self._model} | "
            f"estimated_tokens={prompt_package.estimated_tokens}"
        )
        start_time = time.perf_counter()

        # ── First attempt ─────────────────────────────────────
        try:
            answer = self._call_api(messages)

        except Exception as first_error:
            logger.warning(
                f"First API attempt failed: {first_error}. "
                f"Retrying in 1 second..."
            )
            time.sleep(1)
            was_retried = True

            # ── Retry (once) ──────────────────────────────────
            try:
                answer = self._call_api(messages)
                logger.info("Retry succeeded.")

            except Exception as second_error:
                raise GenerationError(
                    f"LLM generation failed after retry. "
                    f"First error: {first_error}. "
                    f"Second error: {second_error}"
                ) from second_error

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = GenerationResult(
            answer        = answer,
            latency_ms    = latency_ms,
            model         = self._model,
            prompt_tokens = prompt_package.estimated_tokens,
            was_retried   = was_retried,
        )

        logger.info(
            f"Generation complete | "
            f"words={result.word_count} | "
            f"latency={latency_ms:.0f}ms | "
            f"retried={was_retried}"
        )

        return result


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "LLMGenerator",
    "GenerationResult",
    "GenerationError",
]