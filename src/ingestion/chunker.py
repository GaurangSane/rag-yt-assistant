"""
chunker.py
──────────
Splits a YouTube transcript into overlapping time-window chunks.

Design decisions:
  - Timestamp-based windowing preserves accurate start/end times
  - Overlap prevents ideas at window boundaries from being truncated
  - Each chunk is a self-contained Chunk dataclass with all metadata
  - The Chunker class reads all parameters from central config

Pipeline position:  Transcript Fetcher → [Chunker] → Embedder
"""

import logging
from dataclasses import dataclass

from src.config import settings
from src.ingestion.transcript import Transcript_segment

logger = logging.getLogger("rag_app.ingestion.chunker")


# ── Data Structure ─────────────────────────────────────────────────
# A Chunk is the fundamental unit of our RAG system.
# Everything downstream — embedder, vector DB, retriever, LLM —
# works with Chunk objects. Defining it here makes that contract explicit.

@dataclass
class Chunk:
    """
    One time-windowed segment of a YouTube transcript.

    This is the core data unit of the RAG pipeline. After creation
    here, a Chunk flows through: Embedder → VectorStore → Retriever
    → Reranker → PromptBuilder → LLM.

    Attributes:
        chunk_id   : Unique ID for this chunk (e.g. "abc123_chunk_4")
                     Used as the primary key in ChromaDB
        text       : The combined spoken words in this time window
                     This is what gets embedded and sent to the LLM
        start_time : Human-readable start timestamp (e.g. "1:30")
                     Shown to users in citations
        end_time   : Human-readable end timestamp (e.g. "2:28")
                     Shown to users in citations
        start_sec  : Start time as raw float seconds (e.g. 90.0)
                     Used for sorting, math, and YouTube deep links
        video_id   : YouTube video ID this chunk belongs to
                     Essential when supporting multiple videos
        chunk_index: Position of this chunk in the video (0-based)
                     Useful for ordering results chronologically
    """
    chunk_id   : str
    text       : str
    start_time : str
    end_time   : str
    start_sec  : float
    video_id   : str
    chunk_index: int

    @property
    def youtube_link(self) -> str:
        """
        Direct YouTube link that jumps to this chunk's timestamp.

        Example:
            https://youtube.com/watch?v=abc123&t=90s
            Clicking this opens YouTube at exactly 1:30

        This is the feature that makes our citations clickable.
        """
        return (
            f"https://www.youtube.com/watch?v={self.video_id}"
            f"&t={int(self.start_sec)}s"
        )

    @property
    def word_count(self) -> int:
        """Approximate word count of this chunk's text."""
        return len(self.text.split())

    def to_metadata_dict(self) -> dict:
        """
        Serialize chunk metadata for ChromaDB storage.

        ChromaDB metadata must contain only str, int, or float values.
        This method enforces that contract explicitly.

        Returns:
            Dict with all metadata fields — everything EXCEPT 'text'
            and 'embedding' which ChromaDB stores separately.
        """
        return {
            "start_time" : str(self.start_time),
            "end_time"   : str(self.end_time),
            "start_sec"  : float(self.start_sec),
            "video_id"   : str(self.video_id),
            "chunk_index": int(self.chunk_index),
            "youtube_link": str(self.youtube_link),
            "word_count" : int(self.word_count),
        }


# ── Custom Exceptions ───────────────────────────────────────────────
class ChunkingError(Exception):
    """Raised when chunking fails for a structural reason."""
    pass


# ── Helper ──────────────────────────────────────────────────────────
def _format_timestamp(seconds: float) -> str:
    """
    Convert raw seconds → human-readable MM:SS or H:MM:SS.

    Private function — only used inside this module.
    Identical to transcript.py's version intentionally:
    each module is self-contained and does not import
    utilities from sibling modules.
    """
    total = int(seconds)
    h     = total // 3600
    m     = (total % 3600) // 60
    s     = total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── Main Class ──────────────────────────────────────────────────────
class TranscriptChunker:
    """
    Splits a list of TranscriptSegments into overlapping Chunks.

    Uses a sliding time window algorithm:
      - Each chunk covers `window_sec` seconds of transcript
      - Consecutive chunks overlap by `overlap_sec` seconds
      - Window advances by (window_sec - overlap_sec) each step

    Example with window=60, overlap=15:
      Chunk 0: segments from  0s →  60s
      Chunk 1: segments from 45s → 105s  (15s overlap with chunk 0)
      Chunk 2: segments from 90s → 150s  (15s overlap with chunk 1)

    The overlap ensures ideas near window boundaries appear in
    both the preceding and following chunk — nothing is lost.

    Usage:
        chunker = TranscriptChunker()
        chunks  = chunker.chunk(segments, video_id="abc123")
    """

    def __init__(
        self,
        window_sec  : int | None = None,
        overlap_sec : int | None = None,
    ):
        """
        Initialise the chunker.

        Parameters default to values from central config.
        Pass explicit values to override for testing or experimentation.

        Args:
            window_sec  : Seconds each chunk covers. Default: config value (60)
            overlap_sec : Seconds of overlap between chunks. Default: config (15)
        """
        # If caller didn't pass values, use config defaults
        # This pattern — "accept explicit or fall back to config" — 
        # appears in every module. It makes testing easy (pass custom values)
        # while keeping production consistent (config drives everything).
        self.window_sec  = window_sec  or settings.chunking.window_sec
        self.overlap_sec = overlap_sec or settings.chunking.overlap_sec

        # Validate: overlap must be smaller than window
        # Otherwise step_sec ≤ 0 and the while loop never terminates
        if self.overlap_sec >= self.window_sec:
            raise ChunkingError(
                f"overlap_sec ({self.overlap_sec}) must be less than "
                f"window_sec ({self.window_sec}). "
                f"Otherwise chunks would not advance forward."
            )

        # step_sec is how far the window moves each iteration
        # Example: window=60, overlap=15 → step=45
        # Chunk 0 starts at 0s, Chunk 1 at 45s, Chunk 2 at 90s ...
        self.step_sec = self.window_sec - self.overlap_sec

        logger.info(
            f"TranscriptChunker initialized | "
            f"window={self.window_sec}s | "
            f"overlap={self.overlap_sec}s | "
            f"step={self.step_sec}s"
        )

    def chunk(
        self,
        segments : list[Transcript_segment],
        video_id : str,
    ) -> list[Chunk]:
        """
        Split transcript segments into overlapping time-window chunks.

        Args:
            segments : Output from YouTubeTranscriptFetcher.fetch()
                       List of TranscriptSegment objects with timestamps
            video_id : YouTube video ID — embedded in every chunk for
                       traceability across the full pipeline

        Returns:
            List of Chunk objects sorted chronologically by start_sec.
            Consecutive chunks overlap by self.overlap_sec seconds.

        Raises:
            ChunkingError : If segments list is empty or malformed
        """
        # ── Input validation ──────────────────────────────────
        # Fail fast with a clear message rather than letting a
        # cryptic IndexError appear somewhere deep in the pipeline
        if not segments:
            raise ChunkingError(
                "Cannot chunk an empty segments list. "
                "Ensure transcript was fetched successfully."
            )

        if not video_id or not video_id.strip():
            raise ChunkingError(
                "video_id cannot be empty. "
                "Every chunk needs a video_id for traceability."
            )

        logger.info(
            f"Chunking transcript | video_id={video_id} | "
            f"segments={len(segments)}"
        )

        # ── Sliding window algorithm ───────────────────────────
        chunks        = []
        chunk_index   = 0

        # The window slides between the first and last segment
        video_start   = segments[0].start
        video_end     = segments[-1].end   # .end = .start + .duration
        window_start  = video_start

        while window_start < video_end:

            window_end = window_start + self.window_sec

            # Collect all segments whose start falls inside this window.
            # Note: we use segment.start (not segment.end) for the window
            # check — a segment "belongs to" the window where it begins.
            segments_in_window = [
                seg for seg in segments
                if window_start <= seg.start < window_end
            ]

            # Empty window — can happen at the very end of some videos.
            # Skip and advance rather than creating an empty chunk.
            if not segments_in_window:
                logger.debug(
                    f"Empty window at {_format_timestamp(window_start)} "
                    f"— skipping"
                )
                window_start += self.step_sec
                continue

            # ── Build the chunk ────────────────────────────────
            # Join all segment texts with a single space.
            # .strip() on each segment removes leading/trailing whitespace
            # that youtube-transcript-api sometimes includes.
            combined_text = " ".join(
                seg.text.strip() for seg in segments_in_window
            )

            # Actual boundaries snap to real segment positions —
            # not to the theoretical window start/end.
            # This ensures timestamps are always accurate.
            actual_start = segments_in_window[0].start
            actual_end   = segments_in_window[-1].end

            chunk = Chunk(
                chunk_id    = f"{video_id}_chunk_{chunk_index}",
                text        = combined_text,
                start_time  = _format_timestamp(actual_start),
                end_time    = _format_timestamp(actual_end),
                start_sec   = actual_start,
                video_id    = video_id,
                chunk_index = chunk_index,
            )

            chunks.append(chunk)
            chunk_index  += 1
            window_start += self.step_sec   # slide the window forward

        # ── Post-processing ────────────────────────────────────
        # Log statistics useful for debugging retrieval quality
        if chunks:
            avg_words = sum(c.word_count for c in chunks) // len(chunks)
            logger.info(
                f"Chunking complete | "
                f"chunks={len(chunks)} | "
                f"avg_words={avg_words} | "
                f"video_duration={_format_timestamp(video_end)}"
            )
        else:
            # Extremely short video or all-empty segments — warn loudly
            logger.warning(
                f"Chunking produced 0 chunks for video_id={video_id}. "
                f"Video may be too short or transcript may be empty."
            )

        return chunks


# ── Module Exports ──────────────────────────────────────────────────
__all__ = [
    "TranscriptChunker",
    "Chunk",
    "ChunkingError",
]