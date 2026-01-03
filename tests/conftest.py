"""
Shared pytest fixtures for tvidentify tests.

This module provides common fixtures for mocking external dependencies
like ffmpeg, ffprobe, tesseract, and LLM API clients.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_google_api_key(monkeypatch):
    """Set a mock Google API key in the environment."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-12345")


@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Set a mock OpenAI API key in the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-12345")


@pytest.fixture
def mock_perplexity_api_key(monkeypatch):
    """Set a mock Perplexity API key in the environment."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-perplexity-key-12345")


@pytest.fixture
def mock_all_api_keys(mock_google_api_key, mock_openai_api_key, mock_perplexity_api_key):
    """Set all API keys for testing."""
    pass


@pytest.fixture
def clear_api_keys(monkeypatch):
    """Ensure no API keys are set."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)


# ============================================================================
# FFprobe/FFmpeg Fixtures
# ============================================================================

@pytest.fixture
def mock_ffprobe_english_subtitle():
    """Returns mock ffprobe output with an English subtitle stream."""
    return {
        "streams": [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio"},
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng"}
            },
            {
                "index": 3,
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "spa"}
            }
        ]
    }


@pytest.fixture
def mock_ffprobe_no_english_subtitle():
    """Returns mock ffprobe output without English subtitles."""
    return {
        "streams": [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio"},
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "spa"}
            }
        ]
    }


@pytest.fixture
def mock_ffprobe_no_subtitles():
    """Returns mock ffprobe output with no subtitle streams."""
    return {
        "streams": [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio"}
        ]
    }


@pytest.fixture
def mock_subprocess_success(mocker):
    """Mock subprocess.run to always succeed."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b""
    mock_result.stderr = b""
    return mocker.patch("subprocess.run", return_value=mock_result)


# ============================================================================
# LLM Response Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response_success():
    """Returns a successful LLM identification response."""
    return {
        "season": 3,
        "episode": 7,
        "confidence_score": 95,
        "reasoning": "The dialogue mentions Walter's confession which occurs in S03E07."
    }


@pytest.fixture
def mock_llm_response_null():
    """Returns an LLM response with null values (couldn't identify)."""
    return {
        "season": None,
        "episode": None,
        "confidence_score": 10,
        "reasoning": "The subtitles are too generic to identify the episode."
    }


@pytest.fixture
def mock_llm_response_markdown():
    """Returns an LLM response wrapped in markdown code blocks."""
    return '''```json
{
  "season": 2,
  "episode": 5,
  "confidence_score": 88,
  "reasoning": "Character dialogue matches S02E05."
}
```'''


@pytest.fixture
def mock_google_client(mocker, mock_llm_response_success):
    """Mock the Google GenAI client."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(mock_llm_response_success)
    
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response
    
    # The Client is used as a context manager, so we need to mock __enter__
    mock_client_class = MagicMock()
    mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
    
    return mocker.patch("tvidentify.episode_identifier.Client", mock_client_class)



@pytest.fixture
def mock_openai_client(mocker, mock_llm_response_success):
    """Mock the OpenAI client."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_llm_response_success)
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    # Patch at openai.OpenAI since it's imported locally in the function
    return mocker.patch("openai.OpenAI", return_value=mock_client)



# ============================================================================
# Filesystem Fixtures
# ============================================================================

@pytest.fixture
def temp_video_dir():
    """
    Create a temporary directory with mock video files of varying sizes.
    
    Creates:
    - episode_01.mkv (1GB - simulated)
    - episode_02.mkv (1GB - simulated)
    - episode_03.mkv (1GB - simulated)
    - bonus_feature.mkv (100MB - simulated, should be excluded)
    - sample.mkv (50MB - simulated, should be excluded)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock video files with different sizes
        # We use sparse files to simulate sizes without using disk space
        files = [
            ("episode_01.mkv", 1_000_000_000),  # 1GB
            ("episode_02.mkv", 1_000_000_000),  # 1GB
            ("episode_03.mkv", 950_000_000),    # 950MB (still within threshold)
            ("bonus_feature.mkv", 100_000_000), # 100MB (below threshold)
            ("sample.mkv", 50_000_000),         # 50MB (below threshold)
        ]
        
        for filename, size in files:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'wb') as f:
                f.seek(size - 1)
                f.write(b'\0')
        
        yield tmpdir


@pytest.fixture
def temp_video_dir_empty():
    """Create an empty temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_video_dir_already_named():
    """
    Create a temporary directory with files already in the correct format.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        files = [
            ("Breaking Bad S01E01.mkv", 1_000_000_000),
            ("Breaking Bad S01E02.mkv", 1_000_000_000),
            ("random_file.mkv", 1_000_000_000),
        ]
        
        for filename, size in files:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'wb') as f:
                f.seek(size - 1)
                f.write(b'\0')
        
        yield tmpdir


# ============================================================================
# Subtitle Fixtures
# ============================================================================

@pytest.fixture
def sample_subtitles():
    """Sample subtitle text for testing."""
    return [
        "I am the one who knocks!",
        "Say my name.",
        "You're goddamn right.",
        "I did it for me. I liked it.",
        "We're done when I say we're done."
    ]


@pytest.fixture
def empty_subtitles():
    """Empty subtitle list for testing."""
    return []
