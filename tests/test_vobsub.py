"""
Integration tests for VobSub (DVD) subtitle OCR.

These tests use the eng_subs.idx/sub fixtures to verify VobSub parsing and OCR.
Run with: pytest tests/test_vobsub.py -v -m integration
"""

import os
from pathlib import Path

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def vobsub_idx_file(fixtures_dir):
    """Path to the VobSub idx file."""
    idx_path = fixtures_dir / "eng_subs.idx"
    if not idx_path.exists():
        pytest.skip("VobSub test fixtures not found (eng_subs.idx)")
    return str(idx_path)


@pytest.fixture
def vobsub_ocr_test_image(fixtures_dir):
    """Path to a VobSub OCR test image."""
    img_path = fixtures_dir / "vobsub_ocr_input.png"
    if not img_path.exists():
        pytest.skip("VobSub OCR test image not found (vobsub_ocr_input.png)")
    return img_path


class TestVobSubReader:
    """Tests for VobSubReader parsing functionality."""

    def test_vobsub_reader_parses_idx_file(self, vobsub_idx_file):
        """VobSubReader correctly parses eng_subs.idx timestamps."""
        from tvidentify.vobsubreader import VobSubReader
        
        reader = VobSubReader(vobsub_idx_file)
        
        # Should have multiple events
        assert len(reader.events) > 0
        # eng_subs.idx has 438 subtitle events
        assert len(reader.events) >= 400
        
        # Check first event timestamp (00:00:02:269)
        first_timestamp = reader.events[0][0]
        assert first_timestamp == 2269  # milliseconds
        
    def test_vobsub_reader_parses_palette(self, vobsub_idx_file):
        """VobSubReader correctly parses palette from idx file."""
        from tvidentify.vobsubreader import VobSubReader
        
        reader = VobSubReader(vobsub_idx_file)
        
        # Should have 16 colors
        assert len(reader.palette.colors) == 16
        
        # First color should be bebebe (light gray)
        r, g, b, a = reader.palette.colors[0]
        assert (r, g, b) == (0xbe, 0xbe, 0xbe)
        
    def test_vobsub_reader_decodes_images(self, vobsub_idx_file):
        """VobSubReader decodes RLE bitmaps to valid PIL images."""
        from tvidentify.vobsubreader import VobSubReader
        
        reader = VobSubReader(vobsub_idx_file)
        
        # Get first 5 events
        images_decoded = 0
        for event in reader.iter_events(max_events=5):
            if event.image is not None:
                # Should be a valid PIL image
                assert event.image.width > 0
                assert event.image.height > 0
                # Should be RGBA
                assert event.image.mode == 'RGBA'
                images_decoded += 1
        
        # Should have decoded at least some images
        assert images_decoded > 0


class TestVobSubOCR:
    """Tests for VobSub OCR functionality."""

    def test_ocr_vobsub_image_from_file(self, vobsub_ocr_test_image):
        """OCR a VobSub test image from file."""
        from PIL import Image
        from tvidentify.subtitle_handlers import ocr_vobsub_image
        
        # Load the test image
        pil_img = Image.open(vobsub_ocr_test_image)
        
        # Run OCR
        text = ocr_vobsub_image(pil_img)
        
        # Should extract some text
        assert len(text) > 0
        
    def test_ocr_vobsub_from_reader(self, vobsub_idx_file):
        """OCR VobSub images extracted from reader."""
        from tvidentify.vobsubreader import VobSubReader
        from tvidentify.subtitle_handlers import ocr_vobsub_image
        
        reader = VobSubReader(vobsub_idx_file)
        
        # OCR first 3 subtitles
        results = []
        for event in reader.iter_events(max_events=3):
            if event.image:
                text = ocr_vobsub_image(event.image)
                if text:
                    results.append(text)
        
        # Should have extracted text from all 3
        assert len(results) >= 2  # At least 2 should succeed
        
        # Each result should contain readable characters
        for text in results:
            # Should have reasonable length (not gibberish)
            assert len(text) > 0
            # Should contain mostly ASCII printable characters
            printable_count = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,!?'-")
            assert printable_count / len(text) > 0.8


class TestVobSubHandler:
    """Tests for VobSubHandler full extraction pipeline."""

    def test_extract_from_idx_basic(self, vobsub_idx_file):
        """VobSubHandler.extract_from_idx extracts and OCRs subtitles."""
        from tvidentify.subtitle_handlers import VobSubHandler
        
        handler = VobSubHandler()
        
        # Extract first 5 subtitles
        subtitles = handler.extract_from_idx(vobsub_idx_file, max_subtitles=5)
        
        # Should have extracted some subtitles
        assert len(subtitles) > 0
        assert len(subtitles) <= 5
        
        # Print results for manual inspection
        print("\n--- VobSub OCR Results (first 5) ---")
        for i, text in enumerate(subtitles):
            print(f"{i+1}: {text!r}")
        print("---")
        
    def test_extract_from_idx_with_offset(self, vobsub_idx_file):
        """VobSubHandler respects time offset and duration filters."""
        from tvidentify.subtitle_handlers import VobSubHandler
        
        handler = VobSubHandler()
        
        # Extract subtitles starting at 1 minute for 30 seconds
        subtitles = handler.extract_from_idx(
            vobsub_idx_file,
            offset_seconds=60,
            duration_seconds=30,
            max_subtitles=10
        )
        
        # Should have extracted some subtitles in that range
        # (there are subtitles around 1:00-1:30 in eng_subs.idx)
        assert len(subtitles) > 0

    def test_handler_registered_in_codec_handlers(self):
        """VobSubHandler is properly registered for dvd_subtitle codec."""
        from tvidentify.subtitle_handlers import get_handler_for_codec, VobSubHandler
        
        handler = get_handler_for_codec('dvd_subtitle')
        
        assert handler is not None
        assert isinstance(handler, VobSubHandler)


class TestVobSubOCRAccuracy:
    """Tests to measure and report OCR accuracy."""
    
    def test_ocr_quality_report(self, vobsub_idx_file):
        """Generate a quality report for the first 20 subtitles."""
        from tvidentify.subtitle_handlers import VobSubHandler
        
        handler = VobSubHandler()
        
        # Extract first 20 subtitles
        subtitles = handler.extract_from_idx(vobsub_idx_file, max_subtitles=20)
        
        print("\n" + "="*60)
        print("VobSub OCR Quality Report")
        print("="*60)
        
        total_chars = 0
        total_words = 0
        
        for i, text in enumerate(subtitles):
            total_chars += len(text)
            total_words += len(text.split())
            print(f"[{i+1:2d}] {text}")
        
        print("="*60)
        print(f"Total subtitles: {len(subtitles)}")
        print(f"Total characters: {total_chars}")
        print(f"Total words: {total_words}")
        print(f"Avg chars/subtitle: {total_chars/len(subtitles):.1f}" if subtitles else "N/A")
        print("="*60)
        
        # Basic quality checks
        assert len(subtitles) >= 15, "Should extract at least 15 of 20 subtitles"
        assert total_words > 50, "Should have extracted substantial text"
