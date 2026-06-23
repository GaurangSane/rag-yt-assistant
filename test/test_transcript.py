"""
Tests for the transcript fetcher module.

We test three things:
  1. Happy path    — valid URL, transcript exists
  2. Error path    — invalid URL raises correct exception
  3. Unit logic    — video ID extraction works for both URL formats

Notice: we test the MODULE in isolation.
No LLM, no vector DB, no other modules involved.
This is what "unit test" means.
"""

import pytest
from src.ingestion.transcript import (
    YoutubeTranscriptFetcher,
    Transcript_segment,
    InvalidYoutubeURLError,
    TranscriptNotAvailableError,
)

# Create one fetcher instance reused across all tests
fetcher = YoutubeTranscriptFetcher()


# ── Test 1: Video ID Extraction ────────────────────────────
def test_extract_video_id_standard_url():
    """Standard youtube.com/watch?v= format."""
    vid = fetcher.extract_video_id(
        "https://www.youtube.com/watch?v=ktrIQUYIxZo"
    )
    assert vid == "ktrIQUYIxZo"

def test_extract_video_id_short_url():
    """Short youtu.be/ format."""
    vid = fetcher.extract_video_id(
        "https://youtu.be/ktrIQUYIxZo"
    )
    assert vid == "ktrIQUYIxZo"

def test_extract_video_id_with_extra_params():
    """URL with timestamp and playlist params."""
    vid = fetcher.extract_video_id(
        "https://www.youtube.com/watch?v=ktrIQUYIxZo&t=120s&list=PLxxx"
    )
    assert vid == "ktrIQUYIxZo"

def test_extract_video_id_invalid_url():
    """Non-YouTube URL raises correct exception."""
    with pytest.raises(InvalidYoutubeURLError):
        fetcher.extract_video_id("https://www.google.com")


# ── Test 2: TranscriptSegment Dataclass ───────────────────
def test_segment_end_property():
    """end = start + duration."""
    seg = Transcript_segment(text="hello", start=10.0, duration=3.5)
    assert seg.end == 13.5

def test_segment_timestamp_property():
    """Timestamp formats correctly."""
    seg = Transcript_segment(text="hi", start=90.0, duration=2.0)
    assert seg.timestamp == "1:30"

def test_segment_timestamp_hours():
    """Long videos format with hours."""
    seg = Transcript_segment(text="hi", start=3672.0, duration=2.0)
    assert seg.timestamp == "1:01:12"


# ── Test 3: Full Fetch (integration test) ─────────────────
def test_fetch_returns_correct_types():
    """
    Fetch a real video and check return types.
    This hits the network — mark as integration test.
    """
    url = "https://www.youtube.com/watch?v=ktrIQUYIxZo"
    video_id, segments = fetcher.fetch(url)

    assert isinstance(video_id, str)
    assert len(video_id) > 0
    assert isinstance(segments, list)
    assert len(segments) > 0
    assert isinstance(segments[0], Transcript_segment)

# Replace proxy-related tests in tests/test_transcript.py

def test_fetcher_initialises_without_supadata_key():
    """
    Without SUPADATA_API_KEY, fetcher uses direct mode.
    No environment variables needed for local development.
    """
    import os
    os.environ.pop("SUPADATA_API_KEY", None)

    from src.ingestion.transcript import YoutubeTranscriptFetcher
    fetcher = YoutubeTranscriptFetcher()
    assert fetcher._supadata is None
    assert fetcher._direct   is not None


def test_fetcher_initialises_with_supadata_key(monkeypatch):
    """
    With SUPADATA_API_KEY set, Supadata fetcher is created.
    """
    monkeypatch.setenv("SUPADATA_API_KEY", "sup_test_key_123")

    # Force reimport to pick up new env var
    import importlib
    import src.ingestion.transcript as mod
    importlib.reload(mod)

    fetcher = mod.YoutubeTranscriptFetcher()
    assert fetcher._supadata is not None
    assert fetcher._direct   is not None


def test_invalid_url_raises_error():
    """Non-YouTube URL raises InvalidYouTubeURLError."""
    from src.ingestion.transcript import (
        YoutubeTranscriptFetcher,
        InvalidYoutubeURLError,
    )
    fetcher = YoutubeTranscriptFetcher()
    with pytest.raises(InvalidYoutubeURLError):
        fetcher.extract_video_id("https://google.com")


def test_extract_video_id_all_formats():
    """All YouTube URL formats parse correctly."""
    from src.ingestion.transcript import YoutubeTranscriptFetcher
    fetcher = YoutubeTranscriptFetcher()

    cases = [
        ("https://www.youtube.com/watch?v=ktrIQUYIxZo", "ktrIQUYIxZo"),
        ("https://youtu.be/ktrIQUYIxZo",                "ktrIQUYIxZo"),
        ("https://www.youtube.com/shorts/ktrIQUYIxZo",  "ktrIQUYIxZo"),
        ("https://www.youtube.com/embed/ktrIQUYIxZo",   "ktrIQUYIxZo"),
        ("https://www.youtube.com/watch?v=abc&t=120s",  "abc"),
    ]

    for url, expected_id in cases:
        assert fetcher.extract_video_id(url) == expected_id, (
            f"Failed for URL: {url}"
        )

def test_fetch_segments_have_content():
    """All segments have non-empty text and valid timestamps."""
    url = "https://www.youtube.com/watch?v=ktrIQUYIxZo"
    _, segments = fetcher.fetch(url)

    for seg in segments:
        assert isinstance(seg.text, str)
        assert len(seg.text.strip()) > 0
        assert seg.start >= 0
        assert seg.duration > 0