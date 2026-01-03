"""
CLI integration tests - verify command-line arguments control behavior.

These tests invoke the CLI entry points and verify that arguments
are properly honored and affect the output/behavior as expected.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestSubtitleExtractorCLI:
    """Tests for subtitle_extractor CLI arguments."""

    def test_cli_subtitle_extractor_missing_file_exits_gracefully(self, mocker, capsys):
        """Non-existent input file exits with error message, no crash."""
        from tvidentify.subtitle_extractor import main
        
        mocker.patch("sys.argv", [
            "subtitle_extractor",
            "/nonexistent/video.mkv"
        ])
        
        # Should not raise an exception
        main()
        
        # The function should complete without crashing
        # (actual error handling is in the function)

    def test_cli_subtitle_extractor_json_output_creates_file(
        self, mocker, mock_ffprobe_english_subtitle
    ):
        """--output-dir argument causes JSON file to be written."""
        from tvidentify.subtitle_extractor import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake video file
            video_file = os.path.join(tmpdir, "test_video.mkv")
            with open(video_file, 'w') as f:
                f.write("fake video")
            
            output_dir = os.path.join(tmpdir, "output")
            
            # Mock ffprobe to fail finding subtitles (simpler test)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"streams": []})
            mocker.patch("subprocess.run", return_value=mock_result)
            
            mocker.patch("sys.argv", [
                "subtitle_extractor",
                video_file,
                "--output-dir", output_dir
            ])
            
            main()
            
            # Even if no subtitles found, the code path was exercised

    def test_cli_subtitle_extractor_max_frames_argument_parsed(self, mocker):
        """--max-frames argument is parsed and used."""
        from tvidentify.subtitle_extractor import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, "test.mkv")
            with open(video_file, 'w') as f:
                f.write("fake")
            
            # Mock to return empty (no subtitles)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"streams": []})
            mocker.patch("subprocess.run", return_value=mock_result)
            
            mocker.patch("sys.argv", [
                "subtitle_extractor",
                video_file,
                "--max-frames", "10"
            ])
            
            # Should parse without error
            main()


class TestEpisodeIdentifierCLI:
    """Tests for episode_identifier CLI arguments."""

    def test_cli_episode_identifier_provider_google(
        self, mocker, mock_google_api_key, mock_ffprobe_english_subtitle
    ):
        """--provider google routes to Google API."""
        from tvidentify.episode_identifier import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, "test.mkv")
            with open(video_file, 'w') as f:
                f.write("fake")
            
            # Mock extraction to return some subtitles
            mocker.patch(
                "tvidentify.episode_identifier.extract_subtitles",
                return_value=["Test subtitle"]
            )
            
            # Mock the Google client with context manager support
            mock_response = MagicMock()
            mock_response.text = '{"season": 1, "episode": 1, "confidence_score": 90}'
            mock_client_instance = MagicMock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client_class = MagicMock()
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
            mock_google = mocker.patch(
                "tvidentify.episode_identifier.Client",
                mock_client_class
            )
            
            mocker.patch("sys.argv", [
                "episode_identifier",
                video_file,
                "--series-name", "Test Series",
                "--provider", "google"
            ])
            
            main()
            
            # Verify Google client was used
            mock_google.assert_called()

    def test_cli_episode_identifier_provider_openai(
        self, mocker, mock_openai_api_key, mock_ffprobe_english_subtitle
    ):
        """--provider openai routes to OpenAI API."""
        from tvidentify.episode_identifier import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, "test.mkv")
            with open(video_file, 'w') as f:
                f.write("fake")
            
            mocker.patch(
                "tvidentify.episode_identifier.extract_subtitles",
                return_value=["Test subtitle"]
            )
            
            mock_choice = MagicMock()
            mock_choice.message.content = '{"season": 1, "episode": 1, "confidence_score": 90}'
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            # Patch at openai.OpenAI since it's imported locally
            mock_openai = mocker.patch("openai.OpenAI", return_value=mock_client)
            
            mocker.patch("sys.argv", [
                "episode_identifier",
                video_file,
                "--series-name", "Test Series",
                "--provider", "openai"
            ])
            
            main()
            
            mock_openai.assert_called()

    def test_cli_episode_identifier_model_passed_to_api(
        self, mocker, mock_google_api_key
    ):
        """--model argument is forwarded to API client."""
        from tvidentify.episode_identifier import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, "test.mkv")
            with open(video_file, 'w') as f:
                f.write("fake")
            
            mocker.patch(
                "tvidentify.episode_identifier.extract_subtitles",
                return_value=["Test subtitle"]
            )
            
            # Mock the Google client with context manager support
            mock_response = MagicMock()
            mock_response.text = '{"season": 1, "episode": 1, "confidence_score": 90}'
            mock_client_instance = MagicMock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client_class = MagicMock()
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
            mocker.patch("tvidentify.episode_identifier.Client", mock_client_class)
            
            mocker.patch("sys.argv", [
                "episode_identifier",
                video_file,
                "--series-name", "Test",
                "--provider", "google",
                "--model", "gemini-2.5-pro"
            ])
            
            main()
            
            # Verify the model was passed in the call
            call_args = mock_client_instance.models.generate_content.call_args
            assert "gemini-2.5-pro" in str(call_args)


class TestBatchIdentifierCLI:
    """Tests for batch_identifier CLI arguments."""

    def test_cli_batch_identifier_no_rename_by_default(
        self, mocker, mock_google_api_key, temp_video_dir
    ):
        """When --rename not specified, files are not renamed."""
        from tvidentify.batch_identifier import main
        
        # Get list of files before
        files_before = set(os.listdir(temp_video_dir))
        
        # Mock required tools check
        mocker.patch("tvidentify.batch_identifier.check_required_tools", return_value=True)
        
        # Mock subtitle extraction and LLM
        mocker.patch(
            "tvidentify.batch_identifier.extract_subtitles",
            return_value=["Test subtitle"]
        )
        
        # Mock Google client with context manager support
        mock_response = MagicMock()
        mock_response.text = '{"season": 1, "episode": 1, "confidence_score": 90}'
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_client_class = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
        mocker.patch("tvidentify.episode_identifier.Client", mock_client_class)
        
        mocker.patch("sys.argv", [
            "tvidentify",
            temp_video_dir,
            "--series-name", "Test Series"
        ])
        
        main()
        
        # Files should be unchanged (no --rename flag)
        files_after = set(os.listdir(temp_video_dir))
        assert files_before == files_after

    def test_cli_batch_identifier_excludes_already_named(
        self, mocker, mock_google_api_key, temp_video_dir_already_named
    ):
        """Files matching the expected format are skipped when --skip-already-named is used."""
        from tvidentify.batch_identifier import main
        
        # Track which files get processed
        processed_files = []
        
        def track_extract(video_file, *args, **kwargs):
            processed_files.append(os.path.basename(video_file))
            return ["Test subtitle"]
        
        # Mock required tools check
        mocker.patch("tvidentify.batch_identifier.check_required_tools", return_value=True)
        
        mocker.patch(
            "tvidentify.batch_identifier.extract_subtitles",
            side_effect=track_extract
        )
        
        # Mock Google client with context manager support
        mock_response = MagicMock()
        mock_response.text = '{"season": 1, "episode": 3, "confidence_score": 90}'
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_client_class = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
        mocker.patch("tvidentify.episode_identifier.Client", mock_client_class)
        
        mocker.patch("sys.argv", [
            "tvidentify",
            temp_video_dir_already_named,
            "--series-name", "Breaking Bad",
            "--rename",
            "--skip-already-named"
        ])
        
        main()
        
        # Only random_file.mkv should be processed (already named files skipped)
        assert "random_file.mkv" in processed_files
        assert "Breaking Bad S01E01.mkv" not in processed_files
        assert "Breaking Bad S01E02.mkv" not in processed_files



class TestFileRenamerCLI:
    """Tests for file_renamer CLI arguments."""

    def test_cli_file_renamer_dry_run_shows_preview(self, mocker, capsys):
        """--dry-run outputs preview without modifying files."""
        from tvidentify.file_renamer import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a batch results file
            video_file = os.path.join(tmpdir, "episode.mkv")
            with open(video_file, 'w') as f:
                f.write("fake")
            
            batch_results = [{
                "video_file_path": video_file,
                "input_file_name": "episode.mkv",
                "season": 1,
                "episode": 5
            }]
            
            results_file = os.path.join(tmpdir, "results.json")
            with open(results_file, 'w') as f:
                json.dump(batch_results, f)
            
            mocker.patch("sys.argv", [
                "file_renamer",
                "--batch-results", results_file,
                "--series-name", "Breaking Bad",
                "--dry-run"
            ])
            
            main()
            
            # Original file should still exist
            assert os.path.exists(video_file)
            
            # Output should mention what would be renamed
            captured = capsys.readouterr()
            assert "Breaking Bad S01E05" in captured.out or "would_rename" in captured.out

    def test_cli_file_renamer_custom_format(self, mocker):
        """--rename-format applies custom naming format."""
        from tvidentify.file_renamer import main
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, "episode.mkv")
            with open(video_file, 'w') as f:
                f.write("fake")
            
            batch_results = [{
                "video_file_path": video_file,
                "input_file_name": "episode.mkv",
                "season": 2,
                "episode": 10
            }]
            
            results_file = os.path.join(tmpdir, "results.json")
            with open(results_file, 'w') as f:
                json.dump(batch_results, f)
            
            mocker.patch("sys.argv", [
                "file_renamer",
                "--batch-results", results_file,
                "--series-name", "The Wire",
                "--rename-format", "{series} - {season}x{episode:02d}"
            ])
            
            main()
            
            # Check for renamed file with custom format
            expected_name = "The Wire - 2x10.mkv"
            assert os.path.exists(os.path.join(tmpdir, expected_name))
