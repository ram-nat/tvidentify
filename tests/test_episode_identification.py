"""
Tests for episode identification - LLM-based episode matching.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tvidentify.episode_identifier import identify_episode, _parse_json_response


class TestParseJsonResponse:
    """Tests for JSON response parsing from LLM output."""

    def test_parses_plain_json(self):
        """Plain JSON is parsed correctly."""
        response = '{"season": 1, "episode": 5, "confidence_score": 90}'
        result = _parse_json_response(response)
        
        assert result["season"] == 1
        assert result["episode"] == 5
        assert result["confidence_score"] == 90

    def test_parses_markdown_wrapped_json(self, mock_llm_response_markdown):
        """JSON wrapped in markdown code blocks is parsed correctly."""
        result = _parse_json_response(mock_llm_response_markdown)
        
        assert result["season"] == 2
        assert result["episode"] == 5

    def test_returns_none_for_invalid_json(self):
        """Invalid JSON returns None."""
        result = _parse_json_response("this is not json at all")
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Empty string returns None."""
        result = _parse_json_response("")
        assert result is None


class TestIdentifyEpisode:
    """Tests for the main identify_episode function."""

    def test_identify_returns_season_episode_from_llm_response(
        self, mock_google_api_key, mock_google_client, sample_subtitles
    ):
        """Valid subtitles return season/episode from LLM."""
        result = identify_episode(
            "Breaking Bad",
            sample_subtitles,
            model="gemini-2.5-flash",
            provider="google"
        )
        
        assert result["season"] == 3
        assert result["episode"] == 7
        assert result["confidence_score"] == 95

    def test_identify_returns_error_for_empty_subtitles(
        self, mock_google_api_key, empty_subtitles
    ):
        """Empty subtitle list returns error without making API call."""
        result = identify_episode(
            "Breaking Bad",
            empty_subtitles,
            provider="google"
        )
        
        assert "error" in result
        assert "no subtitles" in result["error"].lower()

    def test_identify_returns_error_when_api_key_missing(
        self, clear_api_keys, sample_subtitles
    ):
        """Missing API key returns error."""
        result = identify_episode(
            "Breaking Bad",
            sample_subtitles,
            provider="google"
        )
        
        assert "error" in result
        assert "api key" in result["error"].lower() or "missing" in result["error"].lower()

    def test_identify_handles_llm_returning_null_values(
        self, mock_google_api_key, mocker, sample_subtitles
    ):
        """LLM returning null season/episode is handled."""
        # Mock client to return null values
        null_response = {
            "season": None,
            "episode": None,
            "confidence_score": 10,
            "reasoning": "Could not identify"
        }
        
        mock_response = MagicMock()
        mock_response.text = json.dumps(null_response)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mocker.patch("tvidentify.episode_identifier.Client", return_value=mock_client)
        
        result = identify_episode(
            "Breaking Bad",
            sample_subtitles,
            provider="google"
        )
        
        assert result["season"] is None
        assert result["episode"] is None

    def test_identify_handles_malformed_llm_response(
        self, mock_google_api_key, mocker, sample_subtitles
    ):
        """Malformed LLM response is handled gracefully."""
        mock_response = MagicMock()
        mock_response.text = "I don't know what episode this is, sorry!"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mocker.patch("tvidentify.episode_identifier.Client", return_value=mock_client)
        
        result = identify_episode(
            "Breaking Bad",
            sample_subtitles,
            provider="google"
        )
        
        # Should return None or error, not crash
        assert result is None or "error" in result or result.get("season") is None

    def test_identify_confidence_score_in_response(
        self, mock_google_api_key, mock_google_client, sample_subtitles
    ):
        """Confidence score from LLM is included in response."""
        result = identify_episode(
            "Breaking Bad",
            sample_subtitles,
            provider="google"
        )
        
        assert "confidence_score" in result
        assert isinstance(result["confidence_score"], int)


class TestProviderRouting:
    """Tests for routing to different LLM providers."""

    def test_google_provider_uses_google_client(
        self, mock_google_api_key, mock_google_client, sample_subtitles
    ):
        """Provider 'google' uses Google GenAI client."""
        identify_episode("Series", sample_subtitles, provider="google")
        
        # Verify Google client was instantiated
        mock_google_client.assert_called()

    def test_openai_provider_uses_openai_client(
        self, mock_openai_api_key, mock_openai_client, sample_subtitles
    ):
        """Provider 'openai' uses OpenAI client."""
        identify_episode("Series", sample_subtitles, provider="openai")
        
        # Verify OpenAI client was instantiated
        mock_openai_client.assert_called()

    def test_unknown_provider_returns_error(
        self, mock_all_api_keys, sample_subtitles
    ):
        """Unknown provider returns error."""
        result = identify_episode(
            "Series",
            sample_subtitles,
            provider="unknown_provider"
        )
        
        assert "error" in result
        assert "unknown" in result["error"].lower()
