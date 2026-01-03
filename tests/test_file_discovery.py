"""
Tests for file discovery - finding episode files in directories.
"""

import os
import tempfile

import pytest

from tvidentify.batch_identifier import find_episode_files, is_already_named


class TestFindEpisodeFiles:
    """Tests for finding episode files in directories."""

    def test_finds_all_mkv_files_similar_size(self, temp_video_dir):
        """All similar-sized MKV files are found."""
        files = find_episode_files(temp_video_dir)
        
        # Should find the 3 episode files (all ~1GB), not the bonus content
        assert len(files) == 3
        
        # Verify they are the episode files
        basenames = [os.path.basename(f) for f in files]
        assert "episode_01.mkv" in basenames
        assert "episode_02.mkv" in basenames
        assert "episode_03.mkv" in basenames

    def test_excludes_bonus_content_by_size(self, temp_video_dir):
        """Files significantly smaller than the largest are excluded."""
        files = find_episode_files(temp_video_dir)
        
        basenames = [os.path.basename(f) for f in files]
        assert "bonus_feature.mkv" not in basenames
        assert "sample.mkv" not in basenames

    def test_handles_empty_directory(self, temp_video_dir_empty):
        """Empty directory returns empty list without error."""
        files = find_episode_files(temp_video_dir_empty)
        assert files == []

    def test_respects_extension_filter(self, temp_video_dir):
        """Extension filter limits results to specified extension."""
        # Create an mp4 file
        mp4_path = os.path.join(temp_video_dir, "video.mp4")
        with open(mp4_path, 'wb') as f:
            f.seek(1_000_000_000 - 1)
            f.write(b'\0')
        
        # Should only find the mp4 file
        mp4_files = find_episode_files(temp_video_dir, extension=".mp4")
        assert len(mp4_files) == 1
        assert os.path.basename(mp4_files[0]) == "video.mp4"

    def test_returns_sorted_output(self, temp_video_dir):
        """Files are returned in sorted order."""
        files = find_episode_files(temp_video_dir)
        basenames = [os.path.basename(f) for f in files]
        
        assert basenames == sorted(basenames)


class TestIsAlreadyNamed:
    """Tests for checking if files match the expected naming format."""

    def test_exact_match(self):
        """Exact format match returns True."""
        assert is_already_named(
            "Breaking Bad S01E05.mkv",
            "Breaking Bad"
        ) is True

    def test_case_insensitive(self):
        """Match is case-insensitive."""
        assert is_already_named(
            "breaking bad s01e05.mkv",
            "Breaking Bad"
        ) is True

    def test_not_matching(self):
        """Non-matching filename returns False."""
        assert is_already_named(
            "random_episode_file.mkv",
            "Breaking Bad"
        ) is False

    def test_custom_format(self):
        """Custom format string works correctly."""
        assert is_already_named(
            "Breaking Bad 1x05.mkv",
            "Breaking Bad",
            rename_format="{series} {season}x{episode:02d}"
        ) is True

    def test_partial_match_returns_false(self):
        """Partial match (missing episode) returns False."""
        assert is_already_named(
            "Breaking Bad S01.mkv",
            "Breaking Bad"
        ) is False

    def test_wrong_series_name(self):
        """Wrong series name returns False."""
        assert is_already_named(
            "Breaking Bad S01E05.mkv",
            "Better Call Saul"
        ) is False


class TestSkipsAlreadyNamedFiles:
    """Integration tests for skipping already-named files."""

    def test_find_excludes_already_named_when_checking(self, temp_video_dir_already_named):
        """
        Files already matching the format can be identified.
        (The actual filtering happens in batch_identifier.main, but we verify 
        is_already_named works correctly here.)
        """
        files = find_episode_files(temp_video_dir_already_named)
        
        # All files are found by find_episode_files (it doesn't filter by name)
        assert len(files) == 3
        
        # But is_already_named correctly identifies which are formatted
        already_named = [
            f for f in files 
            if is_already_named(os.path.basename(f), "Breaking Bad")
        ]
        assert len(already_named) == 2
        
        # And which need processing
        needs_processing = [
            f for f in files 
            if not is_already_named(os.path.basename(f), "Breaking Bad")
        ]
        assert len(needs_processing) == 1
        assert "random_file.mkv" in os.path.basename(needs_processing[0])
