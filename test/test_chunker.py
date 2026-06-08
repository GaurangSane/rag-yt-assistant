"""
tests/test_chunker.py
─────────────────────
Unit tests for the TranscriptChunker module.

Key principle: we test chunking logic in ISOLATION.
No network calls. No YouTube API. No LLM.
We build fake TranscriptSegments directly and verify
the chunker handles them correctly.

This is what makes unit tests fast and reliable —
they test ONE thing with no external dependencies.
"""

import pytest
from src.ingestion.chunker import TranscriptChunker, Chunk, ChunkingError
from src.ingestion.transcript import Transcript_segment


# ── Fixtures ────────────────────────────────────────────────────────
# A pytest fixture is a reusable piece of test data or setup.
# Any test function that lists a fixture name as a parameter
# automatically receives its return value.
# This avoids copy-pasting the same setup across 10 tests.

@pytest.fixture
def sample_segments() -> list[Transcript_segment]:
    """
    A controlled set of TranscriptSegments for testing.

    We build these manually so tests never depend on
    YouTube being available or returning specific content.

    Covers 3 minutes (180 seconds) of transcript:
      0s  → 10 segments covering  0s -  90s
      90s → 10 segments covering 90s - 180s
    """
    segments = []
    for i in range(20):
        segments.append(Transcript_segment(
            text     = f"This is segment number {i} discussing topic {i}",
            start    = float(i * 9),     # each segment starts 9s apart
            duration = 8.5,              # each segment lasts 8.5s
        ))
    return segments


@pytest.fixture
def chunker() -> TranscriptChunker:
    """
    A chunker with small, predictable settings for testing.

    window=60, overlap=15 → step=45
    With 20 segments spanning 180s, we expect ~4 chunks.
    """
    return TranscriptChunker(window_sec=60, overlap_sec=15)


# ── Tests: Basic Chunking ────────────────────────────────────────────
class TestBasicChunking:
    """Tests for normal, happy-path chunking behaviour."""

    def test_returns_list_of_chunks(self, chunker, sample_segments):
        """Output is a non-empty list of Chunk objects."""
        result = chunker.chunk(sample_segments, video_id="test123")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_ids_are_unique(self, chunker, sample_segments):
        """Every chunk has a different ID — no duplicates."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_id_contains_video_id(self, chunker, sample_segments):
        """Video ID is embedded in every chunk ID."""
        chunks = chunker.chunk(sample_segments, video_id="myVideoABC")
        for chunk in chunks:
            assert "myVideoABC" in chunk.chunk_id

    def test_chunk_text_is_not_empty(self, chunker, sample_segments):
        """No chunk should have empty or whitespace-only text."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunks_sorted_chronologically(self, chunker, sample_segments):
        """Chunks are ordered by start time, earliest first."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        starts = [c.start_sec for c in chunks]
        assert starts == sorted(starts)

    def test_chunk_indices_are_sequential(self, chunker, sample_segments):
        """chunk_index goes 0, 1, 2, 3 ... with no gaps."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ── Tests: Overlap Verification ─────────────────────────────────────
class TestOverlap:
    """
    The overlap is the most critical design feature.
    These tests verify it is actually working — not just
    that the code runs without errors.
    """

    def test_consecutive_chunks_overlap(self, chunker, sample_segments):
        """
        Chunk N+1 must start BEFORE chunk N ends.
        If this fails, there is no overlap — the feature is broken.
        """
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_   = chunks[i + 1]
            assert next_.start_sec < current.start_sec + chunker.window_sec, (
                f"Chunk {i+1} starts at {next_.start_sec}s but chunk {i} "
                f"window ends at {current.start_sec + chunker.window_sec}s "
                f"— no overlap detected"
            )

    def test_shared_text_in_overlapping_chunks(self, chunker, sample_segments):
        """
        Because chunks overlap in time, some segment text should
        appear in two consecutive chunks. This is the overlap working.
        """
        chunks = chunker.chunk(sample_segments, video_id="test123")
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")

        words_in_chunk_0 = set(chunks[0].text.lower().split())
        words_in_chunk_1 = set(chunks[1].text.lower().split())
        shared_words     = words_in_chunk_0 & words_in_chunk_1

        assert len(shared_words) > 0, (
            "No shared words between chunk 0 and chunk 1 — overlap not working"
        )


# ── Tests: Timestamps ────────────────────────────────────────────────
class TestTimestamps:
    """Timestamps are our killer feature. Test them thoroughly."""

    def test_timestamps_are_strings(self, chunker, sample_segments):
        """start_time and end_time must be strings like '1:30'."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for chunk in chunks:
            assert isinstance(chunk.start_time, str)
            assert isinstance(chunk.end_time, str)

    def test_start_sec_is_float(self, chunker, sample_segments):
        """start_sec must be a float for math operations."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for chunk in chunks:
            assert isinstance(chunk.start_sec, float)
            assert chunk.start_sec >= 0.0

    def test_end_time_after_start_time(self, chunker, sample_segments):
        """Every chunk's end must be after its start."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for chunk in chunks:
            assert chunk.start_sec < chunk.start_sec + 1, (
                "Chunk start_sec must be less than end"
            )

    def test_first_chunk_starts_near_zero(self, chunker, sample_segments):
        """First chunk should start at or very near the video beginning."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        assert chunks[0].start_sec < 10.0, (
            f"First chunk starts at {chunks[0].start_sec}s — expected near 0"
        )

    def test_youtube_link_format(self, chunker, sample_segments):
        """YouTube deep link must be a valid URL with timestamp param."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for chunk in chunks:
            assert chunk.youtube_link.startswith("https://www.youtube.com")
            assert "&t=" in chunk.youtube_link
            assert chunk.video_id in chunk.youtube_link


# ── Tests: Metadata Dict ─────────────────────────────────────────────
class TestMetadataDict:
    """to_metadata_dict() is what we pass to ChromaDB."""

    def test_metadata_dict_has_required_keys(self, chunker, sample_segments):
        """All fields ChromaDB needs must be present."""
        chunks   = chunker.chunk(sample_segments, video_id="test123")
        metadata = chunks[0].to_metadata_dict()
        required = {
            "start_time", "end_time", "start_sec",
            "video_id", "chunk_index", "youtube_link", "word_count"
        }
        assert required.issubset(metadata.keys())

    def test_metadata_values_are_correct_types(self, chunker, sample_segments):
        """ChromaDB requires str/int/float — no lists, sets, or None."""
        chunks   = chunker.chunk(sample_segments, video_id="test123")
        metadata = chunks[0].to_metadata_dict()
        for key, value in metadata.items():
            assert isinstance(value, (str, int, float)), (
                f"Metadata key '{key}' has type {type(value).__name__} "
                f"— ChromaDB only accepts str, int, float"
            )


# ── Tests: Error Handling ────────────────────────────────────────────
class TestErrorHandling:
    """Chunker should fail loudly with clear messages — never silently."""

    def test_empty_segments_raises_error(self, chunker):
        """Empty input must raise ChunkingError, not IndexError."""
        with pytest.raises(ChunkingError, match="empty segments"):
            chunker.chunk([], video_id="test123")

    def test_empty_video_id_raises_error(self, chunker, sample_segments):
        """Blank video_id must raise ChunkingError."""
        with pytest.raises(ChunkingError, match="video_id"):
            chunker.chunk(sample_segments, video_id="")

    def test_invalid_overlap_raises_error(self):
        """overlap >= window makes the algorithm infinite — must be caught."""
        with pytest.raises(ChunkingError, match="overlap_sec"):
            TranscriptChunker(window_sec=60, overlap_sec=60)

    def test_overlap_greater_than_window_raises_error(self):
        """overlap > window is even more invalid."""
        with pytest.raises(ChunkingError, match="overlap_sec"):
            TranscriptChunker(window_sec=30, overlap_sec=45)


# ── Tests: Edge Cases ────────────────────────────────────────────────
class TestEdgeCases:
    """Real-world videos have edge cases. Handle them gracefully."""

    def test_single_segment_video(self, chunker):
        """A video with only one segment still produces one chunk."""
        single = [Transcript_segment(
            text="only one segment in this video",
            start=0.0, duration=10.0
        )]
        chunks = chunker.chunk(single, video_id="short123")
        assert len(chunks) == 1
        assert chunks[0].text == "only one segment in this video"

    def test_different_video_ids_produce_different_chunk_ids(
        self, chunker, sample_segments
    ):
        """Same segments, different video IDs → different chunk IDs."""
        chunks_a = chunker.chunk(sample_segments, video_id="videoA")
        chunks_b = chunker.chunk(sample_segments, video_id="videoB")
        ids_a = {c.chunk_id for c in chunks_a}
        ids_b = {c.chunk_id for c in chunks_b}
        assert ids_a.isdisjoint(ids_b), (
            "Chunk IDs from different videos should never overlap"
        )

    def test_word_count_is_positive(self, chunker, sample_segments):
        """Every chunk must contain at least one word."""
        chunks = chunker.chunk(sample_segments, video_id="test123")
        for chunk in chunks:
            assert chunk.word_count > 0