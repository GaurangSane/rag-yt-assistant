"""
transcript.py
─────────────
Fetches YouTube video transcripts with timestamps.

Cloud deployment strategy:
  Primary (cloud):  Supadata API — bypasses YouTube IP blocking
  Primary (local):  youtube-transcript-api — direct, no API needed
  Fallback:         youtube-transcript-api without proxy

Environment variables:
  SUPADATA_API_KEY      → enables Supadata (set on Railway)
  ENVIRONMENT           → "production" triggers cloud mode

Pipeline position: [TranscriptFetcher] → Chunker → Embedder
"""

import os
import logging
import time
from dataclasses import dataclass

import requests as http_requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)

logger = logging.getLogger("rag_app.ingestion.transcript")


# ── Custom Exceptions ─────────────────────────────────────────────────
class TranscriptError(Exception):
    """Base exception for all transcript errors."""
    pass

class TranscriptNotAvailableError(TranscriptError):
    """Video exists but has no transcript."""
    pass

class InvalidYoutubeURLError(TranscriptError):
    """URL could not be parsed into a video ID."""
    pass


# ── Data Structure ────────────────────────────────────────────────────
@dataclass
class Transcript_segment:
    """
    One timestamped unit of speech from a YouTube transcript.
    Identical structure regardless of which fetch method was used.
    """
    text    : str
    start   : float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def timestamp(self) -> str:
        return _format_timestamp(self.start)


def _format_timestamp(seconds: float) -> str:
    """Convert raw seconds → MM:SS or H:MM:SS string."""
    total = int(seconds)
    h     = total // 3600
    m     = (total % 3600) // 60
    s     = total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── Supadata Fetcher ──────────────────────────────────────────────────
class _SupadataFetcher:
    """
    Fetches transcripts via Supadata API.

    Supadata runs their own infrastructure to fetch YouTube transcripts.
    They handle the cloud IP blocking problem on their end.
    We make one clean API call and get back the transcript.

    API docs: docs.supadata.ai
    Free tier: 100 transcripts/day
    """

    BASE_URL = "https://api.supadata.ai/v1/youtube/transcript"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._session = http_requests.Session()
        self._session.headers.update({
            "x-api-key": api_key,
        })
        logger.info("SupadataFetcher initialized")

    def fetch(
        self,
        video_id: str,
        lang    : str = "en",
    ) -> list[Transcript_segment]:
        """
        Fetch transcript via Supadata API.

        Args:
            video_id : YouTube video ID (not full URL)
            lang     : Preferred language code (default: "en")
                       Falls back to any available language if not found

        Returns:
            List of TranscriptSegment objects with text + timestamps

        Raises:
            TranscriptNotAvailableError: Video has no captions
            Exception: API error (caller handles and falls back)
        """
        logger.info(
            f"Fetching via Supadata | "
            f"video_id={video_id} | lang={lang}"
        )

        # First try requested language
        resp = self._session.get(
            self.BASE_URL,
            params  = {
                "videoId": video_id,
                "lang"   : lang,
                "text"   : False,   # get timestamped segments not plain text
            },
            timeout = 30,
        )

        # If requested language not found, try without language filter
        if resp.status_code == 404:
            logger.info(
                f"Language '{lang}' not found — "
                f"trying auto-detect..."
            )
            resp = self._session.get(
                self.BASE_URL,
                params  = {"videoId": video_id, "text": False},
                timeout = 30,
            )

        # No transcript exists at all
        if resp.status_code == 404:
            raise TranscriptNotAvailableError(
                f"No transcript available for video '{video_id}'. "
                f"The video may not have captions."
            )

        # Authentication error
        if resp.status_code == 401:
            raise Exception(
                "Supadata API key is invalid or expired. "
                "Check SUPADATA_API_KEY environment variable."
            )

        # Rate limit
        if resp.status_code == 429:
            raise Exception(
                "Supadata API rate limit reached. "
                "Free tier: 100 transcripts/day."
            )

        resp.raise_for_status()
        data = resp.json()

        # Parse Supadata response format
        # Response: {"content": [{"text": "...", "offset": 0, "duration": 3200}]}
        # Note: Supadata uses milliseconds for offset and duration
        raw_segments = data.get("content", [])

        if not raw_segments:
            raise TranscriptNotAvailableError(
                f"Supadata returned empty transcript for video '{video_id}'."
            )

        segments = []
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Supadata uses milliseconds — convert to seconds
            start_ms    = seg.get("offset", 0)
            duration_ms = seg.get("duration", 3000)

            segments.append(Transcript_segment(
                text     = text,
                start    = start_ms / 1000.0,
                duration = duration_ms / 1000.0,
            ))

        logger.info(
            f"Supadata fetch complete | "
            f"video_id={video_id} | "
            f"segments={len(segments)}"
        )

        return segments


# ── Direct Fetcher (youtube-transcript-api) ───────────────────────────
class _DirectFetcher:
    """
    Fetches transcripts directly via youtube-transcript-api.

    Works reliably on local machines with residential IPs.
    May fail on cloud deployments due to YouTube IP blocking.
    Used as primary method locally, fallback on cloud.
    """

    def __init__(self):
        # youtube-transcript-api v1.2.4
        # Constructor: YouTubeTranscriptApi(proxy_config=None, http_client=None)
        self._api = YouTubeTranscriptApi()
        logger.info("DirectFetcher initialized (no proxy)")

    def fetch(self, video_id: str) -> list[Transcript_segment]:
        """
        Fetch transcript directly from YouTube.

        Args:
            video_id: YouTube video ID

        Returns:
            List of TranscriptSegment objects

        Raises:
            TranscriptNotAvailableError: Video has no captions
            Exception: Network error, IP block, etc.
        """
        logger.info(f"Direct fetch | video_id={video_id}")

        try:
            raw = self._api.fetch(video_id)
        except TranscriptsDisabled:
            raise TranscriptNotAvailableError(
                f"Captions disabled for video '{video_id}'."
            )
        except NoTranscriptFound:
            raise TranscriptNotAvailableError(
                f"No transcript for video '{video_id}'."
            )

        segments = [
            Transcript_segment(
                text     = seg.text.strip(),
                start    = seg.start,
                duration = seg.duration,
            )
            for seg in raw
            if seg.text.strip()
        ]

        logger.info(
            f"Direct fetch complete | "
            f"video_id={video_id} | "
            f"segments={len(segments)}"
        )

        return segments


# ── Main Class ────────────────────────────────────────────────────────
class YoutubeTranscriptFetcher:
    """
    Fetches YouTube transcripts with automatic method selection.

    Method selection logic:
      If SUPADATA_API_KEY is set → Supadata first, direct as fallback
      If no SUPADATA_API_KEY    → direct only (local development)

    This means:
      Local dev   → direct fetch, no API keys needed
      Railway     → Supadata first (bypasses IP block)

    The TranscriptSegment output is identical regardless
    of which method fetched it — zero pipeline accuracy difference.

    Usage:
        fetcher = YouTubeTranscriptFetcher()
        video_id, segments = fetcher.fetch("https://youtube.com/watch?v=abc")
    """

    def __init__(self):
        supadata_key = os.getenv("SUPADATA_API_KEY", "").strip()

        # Build available fetchers
        self._supadata = (
            _SupadataFetcher(supadata_key)
            if supadata_key
            else None
        )
        self._direct = _DirectFetcher()

        if self._supadata:
            logger.info(
                "TranscriptFetcher ready | "
                "mode=cloud (Supadata primary + direct fallback)"
            )
        else:
            logger.info(
                "TranscriptFetcher ready | "
                "mode=local (direct only) | "
                "Set SUPADATA_API_KEY for cloud deployment"
            )

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from any YouTube URL format."""
        url = url.strip()

        if "v=" in url:
            return url.split("v=")[1].split("&")[0]

        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]

        if "/shorts/" in url:
            return url.split("/shorts/")[1].split("?")[0]

        if "/embed/" in url:
            return url.split("/embed/")[1].split("?")[0]

        raise InvalidYoutubeURLError(
            "Could not extract video ID from URL. "
            "Supported formats: youtube.com/watch?v=ID, "
            "youtu.be/ID, youtube.com/shorts/ID"
        )

    def fetch(
        self,
        url: str,
    ) -> tuple[str, list[Transcript_segment]]:
        """
        Fetch transcript using the best available method.

        Automatically selects Supadata or direct based on
        environment configuration.

        Args:
            url: Full YouTube URL (any format)

        Returns:
            Tuple of (video_id, list[TranscriptSegment])
            video_id is returned alongside segments so callers
            never need to call extract_video_id separately.

        Raises:
            InvalidYouTubeURLError      : URL cannot be parsed
            TranscriptNotAvailableError : Video has no captions
                                          (from any method)
        """
        video_id = self.extract_video_id(url)

        if self._supadata:
            return self._fetch_with_supadata_primary(video_id)
        else:
            return self._fetch_direct_only(video_id)

    def _fetch_with_supadata_primary(
        self,
        video_id: str,
    ) -> tuple[str, list[Transcript_segment]]:
        """
        Cloud mode: Supadata first, direct fallback.

        Supadata handles Railway's IP block.
        Direct method kept as fallback in case Supadata
        is temporarily unavailable.
        """
        # ── Try Supadata first ────────────────────────────────
        try:
            segments = self._supadata.fetch(video_id)
            logger.info(
                f"Transcript via Supadata | "
                f"video_id={video_id} | "
                f"segments={len(segments)}"
            )
            return video_id, segments

        except TranscriptNotAvailableError:
            # Video genuinely has no captions — don't try fallback
            raise

        except Exception as e:
            logger.warning(
                f"Supadata failed | "
                f"video_id={video_id} | "
                f"error={e} | "
                f"trying direct fallback..."
            )

        # ── Fallback: direct fetch ────────────────────────────
        try:
            segments = self._direct.fetch(video_id)
            logger.info(
                f"Transcript via direct fallback | "
                f"video_id={video_id} | "
                f"segments={len(segments)}"
            )
            return video_id, segments

        except TranscriptNotAvailableError:
            raise

        except Exception as e2:
            logger.error(
                f"All transcript methods failed | "
                f"video_id={video_id} | "
                f"direct_error={e2}"
            )
            raise TranscriptNotAvailableError(
                f"Could not fetch transcript for '{video_id}'. "
                f"Supadata and direct fetch both failed. "
                f"The video may have no captions or all services "
                f"are temporarily unavailable."
            )

    def _fetch_direct_only(
        self,
        video_id: str,
    ) -> tuple[str, list[Transcript_segment]]:
        """
        Local mode: direct fetch only.
        Used when SUPADATA_API_KEY is not set.
        """
        try:
            segments = self._direct.fetch(video_id)
            return video_id, segments

        except TranscriptNotAvailableError:
            raise

        except Exception as e:
            error_str = str(e).lower()

            # Give helpful message if it looks like an IP block
            if any(phrase in error_str for phrase in [
                "blocking", "ip", "429", "too many requests",
            ]):
                raise TranscriptNotAvailableError(
                    f"YouTube is blocking requests for video '{video_id}'. "
                    f"On cloud deployments, set SUPADATA_API_KEY "
                    f"to route through Supadata."
                ) from e
            raise TranscriptNotAvailableError(
                f"Failed to fetch transcript for '{video_id}': {e}"
            ) from e


# ── Module Exports ────────────────────────────────────────────────────
__all__ = [
    "YoutubeTranscriptFetcher",
    "Transcript_segment",
    "TranscriptNotAvailableError",
    "InvalidYoutubeURLError",
    "TranscriptError",
]