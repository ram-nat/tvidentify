"""
Integration tests - require real ffmpeg, ffprobe, and tesseract installed.

These tests use synthetic fixtures to verify the real tool chain works.
Run with: pytest tests/test_integration.py -v -m integration
Skip with: pytest tests/ -v -m "not integration"
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def subtitle_test_image(fixtures_dir):
    """Load the subtitle-style test image as an OpenCV array."""
    img_path = fixtures_dir / "subtitle_test.png"
    if not img_path.exists():
        pytest.skip("Test fixture not found. Run: python tests/generate_fixtures.py")
    return cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)



@pytest.fixture
def sup_test_file(fixtures_dir):
    """Path to the test SUP file."""
    sup_path = fixtures_dir / "test_subtitle.sup"
    if not sup_path.exists():
        pytest.skip("Test fixture not found. Run: python tests/generate_fixtures.py")
    return str(sup_path)


class TestTesseractOCR:
    """Tests that require tesseract to be installed."""

    def test_ocr_reads_subtitle_image(self, subtitle_test_image):
        """Tesseract can read subtitle-style image (white text on transparent background)."""
        from tvidentify.subtitle_extractor import ocr_image
        
        result = ocr_image(subtitle_test_image)
        
        # Should extract some text from the subtitle
        assert len(result) > 0
        # Check for key words (OCR may have minor errors)
        assert "knock" in result.lower() or "one" in result.lower() or "who" in result.lower()


class TestSubtitleExtraction:
    """Tests for full subtitle extraction from SUP files."""

    def test_extract_text_from_sup_file(self, sup_test_file):
        """Full extraction pipeline: SUP file → PGS parsing → OCR → text."""
        from tvidentify.subtitle_extractor import extract_text_from_sup
        
        subtitles = extract_text_from_sup(sup_test_file)
        
        # Should extract at least one subtitle
        assert len(subtitles) >= 1
        # The SUP file contains "TEST"
        assert any("test" in s.lower() for s in subtitles)


class TestToolChain:
    """Tests that require ffmpeg/ffprobe/tesseract to be installed."""

    def test_check_required_tools_passes(self):
        """All required tools (ffmpeg, ffprobe, tesseract) are installed."""
        from tvidentify.utils import check_required_tools
        
        assert check_required_tools() is True
