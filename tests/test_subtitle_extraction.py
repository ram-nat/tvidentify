"""
Tests for subtitle extraction pipeline (with mocked ffmpeg/OCR).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from tvidentify.subtitle_extractor import (
    extract_subtitles,
    find_subtitle_stream,
    get_subtitle_tracks,
)
from tvidentify.subtitle_handlers import (
    SubtitleHandler,
    PGSHandler,
    SRTHandler,
    get_handler_for_codec,
)
from tvidentify.batch_identifier import get_subtitle_fingerprint


class TestFindSubtitleStream:
    """Tests for finding subtitle streams."""

    def test_extraction_uses_english_subtitle_track(
        self, mocker, mock_ffprobe_english_subtitle
    ):
        """When English subtitle exists, it is selected."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_english_subtitle)
        mocker.patch("subprocess.run", return_value=mock_result)
        
        result = find_subtitle_stream("/fake/video.mkv")
        
        assert result is not None
        stream_index, handler = result
        assert stream_index == 2  # Index of English subtitle stream
        assert isinstance(handler, PGSHandler)

    def test_extraction_uses_specified_subtitle_track(
        self, mocker, mock_ffprobe_english_subtitle
    ):
        """When a subtitle track is specified, it is used."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_english_subtitle)
        mocker.patch("subprocess.run", return_value=mock_result)

        result = find_subtitle_stream(
            "/fake/video.mkv", subtitle_track_index=3
        )

        assert result is not None
        stream_index, handler = result
        assert stream_index == 3
        assert isinstance(handler, PGSHandler)

    def test_extraction_falls_back_when_no_english(
        self, mocker, mock_ffprobe_no_english_subtitle
    ):
        """When no English subtitle exists, falls back to first available."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_no_english_subtitle)
        mocker.patch("subprocess.run", return_value=mock_result)
        
        result = find_subtitle_stream("/fake/video.mkv")
        
        # Should fall back to the first subtitle stream (index 2)
        assert result is not None
        stream_index, handler = result
        assert stream_index == 2

    def test_extraction_returns_none_for_no_subtitles(
        self, mocker, mock_ffprobe_no_subtitles
    ):
        """When no subtitles exist, returns None."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_no_subtitles)
        mocker.patch("subprocess.run", return_value=mock_result)
        
        result = find_subtitle_stream("/fake/video.mkv")
        
        assert result is None

    def test_extraction_returns_none_for_nonexistent_track(
        self, mocker, mock_ffprobe_english_subtitle
    ):
        """When a nonexistent track is specified, returns None."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_english_subtitle)
        mocker.patch("subprocess.run", return_value=mock_result)

        result = find_subtitle_stream(
            "/fake/video.mkv", subtitle_track_index=99
        )

        assert result is None


class TestExtractSubtitles:
    """Tests for the main extract_subtitles function."""

    def test_extraction_returns_empty_for_no_subtitles(
        self, mocker, mock_ffprobe_no_subtitles
    ):
        """Video with no subtitles returns empty list, no crash."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_no_subtitles)
        mocker.patch("subprocess.run", return_value=mock_result)
        
        with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
            subtitles = extract_subtitles(f.name)
        
        assert subtitles == []

    def test_extraction_returns_empty_for_missing_file(self):
        """Non-existent file returns empty list."""
        subtitles = extract_subtitles("/nonexistent/video.mkv")
        assert subtitles == []

    def test_extraction_uses_handler(self, mocker, mock_ffprobe_english_subtitle):
        """extract_subtitles delegates to the appropriate handler."""
        mock_ffprobe_result = MagicMock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = json.dumps(mock_ffprobe_english_subtitle)
        mocker.patch("subprocess.run", return_value=mock_ffprobe_result)
        
        # Mock the handler's extract_text method
        mock_handler = MagicMock(spec=PGSHandler)
        mock_handler.extract_text.return_value = ["Test subtitle"]
        mocker.patch(
            "tvidentify.subtitle_extractor.get_handler_for_codec",
            return_value=mock_handler
        )
        
        with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
            subtitles = extract_subtitles(f.name, scan_duration_minutes=10)
        
        # Verify handler was called with correct args
        mock_handler.extract_text.assert_called_once()
        assert subtitles == ["Test subtitle"]


class TestHandlerSelection:
    """Tests for codec-to-handler mapping."""

    def test_pgs_codec_returns_pgs_handler(self):
        """PGS codec returns PGSHandler."""
        handler = get_handler_for_codec("hdmv_pgs_subtitle")
        assert isinstance(handler, PGSHandler)

    def test_srt_codec_returns_srt_handler(self):
        """SRT codec returns SRTHandler."""
        handler = get_handler_for_codec("subrip")
        assert isinstance(handler, SRTHandler)

    def test_unknown_codec_returns_none(self):
        """Unknown codec returns None."""
        handler = get_handler_for_codec("unknown_codec")
        assert handler is None

    def test_vobsub_codec_returns_handler(self):
        """VobSub codec returns VobSubHandler."""
        from tvidentify.subtitle_handlers import VobSubHandler
        handler = get_handler_for_codec("dvd_subtitle")
        assert isinstance(handler, VobSubHandler)


class TestSubtitleFingerprint:
    """Tests for subtitle fingerprinting (duplicate detection)."""

    def test_fingerprint_returns_consistent_hash(self, mocker):
        """Same subtitles produce the same fingerprint."""
        subtitles = ["Line one", "Line two", "Line three"]
        
        # Mock extract_subtitles to return our fixed subtitles
        mocker.patch(
            "tvidentify.batch_identifier.extract_subtitles",
            return_value=subtitles
        )
        
        fp1, subs1 = get_subtitle_fingerprint("/video1.mkv", 0, 0, 15)
        fp2, subs2 = get_subtitle_fingerprint("/video2.mkv", 0, 0, 15)
        
        assert fp1 == fp2
        assert subs1 == subs2

    def test_fingerprint_detects_duplicates(self, mocker):
        """Two files with same subtitles have same fingerprint."""
        subtitles = ["I am the one who knocks!", "Say my name."]
        
        mocker.patch(
            "tvidentify.batch_identifier.extract_subtitles",
            return_value=subtitles
        )
        
        fp1, _ = get_subtitle_fingerprint("/episode_copy1.mkv", 0, 0, 15)
        fp2, _ = get_subtitle_fingerprint("/episode_copy2.mkv", 0, 0, 15)
        
        assert fp1 == fp2

    def test_fingerprint_different_for_different_content(self, mocker):
        """Different subtitles produce different fingerprints."""
        call_count = [0]
        
        def mock_extract(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ["Episode one dialogue"]
            else:
                return ["Episode two dialogue"]
        
        mocker.patch(
            "tvidentify.batch_identifier.extract_subtitles",
            side_effect=mock_extract
        )
        
        fp1, _ = get_subtitle_fingerprint("/episode1.mkv", 0, 0, 15)
        fp2, _ = get_subtitle_fingerprint("/episode2.mkv", 0, 0, 15)
        
        assert fp1 != fp2

    def test_fingerprint_returns_none_for_no_subtitles(self, mocker):
        """When extraction fails, returns (None, None)."""
        mocker.patch(
            "tvidentify.batch_identifier.extract_subtitles",
            return_value=[]
        )
        
        fp, subs = get_subtitle_fingerprint("/video.mkv", 0, 0, 15)
        
        assert fp is None
        assert subs is None
