"""
Tests for utilities module - environment and configuration checks.
"""

import logging
import os
from unittest.mock import MagicMock

import pytest

from tvidentify.utils import check_api_key, check_required_tools


class TestApiKeyCheck:
    """Tests for API key validation."""

    def test_api_key_check_validates_env_var(self, mock_google_api_key):
        """When GOOGLE_API_KEY is set, check_api_key returns True."""
        assert check_api_key("google") is True

    def test_api_key_check_fails_when_missing(self, clear_api_keys):
        """When API key is not set, check_api_key returns False."""
        assert check_api_key("google") is False

    def test_api_key_check_openai(self, mock_openai_api_key):
        """OpenAI key check works correctly."""
        assert check_api_key("openai") is True

    def test_api_key_check_perplexity(self, mock_perplexity_api_key):
        """Perplexity key check works correctly."""
        assert check_api_key("perplexity") is True

    def test_api_key_check_unknown_provider(self):
        """Unknown provider returns False."""
        assert check_api_key("unknown_provider") is False


class TestRequiredToolsCheck:
    """Tests for checking required external tools."""

    def test_required_tools_check_passes_when_installed(self, mocker):
        """When all tools are available, returns True."""
        # Mock subprocess.run to succeed for all tools
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)
        
        assert check_required_tools() is True
        
        # Verify base tools (ffmpeg, ffprobe) were checked
        assert mock_run.call_count == 2

    def test_required_tools_check_fails_when_missing(self, mocker):
        """When a tool is missing, returns False."""
        # Mock subprocess.run to raise FileNotFoundError for ffmpeg
        def side_effect(args, **kwargs):
            if args[0] == "ffmpeg":
                raise FileNotFoundError("ffmpeg not found")
            return MagicMock(returncode=0)
        
        mocker.patch("subprocess.run", side_effect=side_effect)
        
        assert check_required_tools() is False

    def test_ocr_tools_check_passes(self, mocker):
        """OCR tools check verifies tesseract."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)
        
        from tvidentify.utils import check_ocr_tools
        assert check_ocr_tools() is True
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0][0] == 'tesseract'

    def test_vobsub_tools_check_passes(self, mocker):
        """VobSub tools check verifies mkvextract and mkvmerge."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)
        
        from tvidentify.utils import check_vobsub_tools
        assert check_vobsub_tools() is True
        # Checks mkvextract and mkvmerge
        assert mock_run.call_count == 2
