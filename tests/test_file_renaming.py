"""
Tests for file renaming operations.
"""

import json
import os
import tempfile

import pytest

from tvidentify.file_renamer import rename_file, rename_files_from_batch_results


class TestRenameFile:
    """Tests for the rename_file function."""

    def test_rename_applies_format_correctly(self):
        """Rename applies the format string correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            original = os.path.join(tmpdir, "original.mkv")
            with open(original, 'w') as f:
                f.write("test")
            
            result = rename_file(original, "Breaking Bad", 1, 5)
            
            assert result["success"] is True
            assert result["new_path"].endswith("Breaking Bad S01E05.mkv")
            assert os.path.exists(result["new_path"])
            assert not os.path.exists(original)

    def test_rename_preserves_original_extension(self):
        """Original file extension is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "video.mp4")
            with open(original, 'w') as f:
                f.write("test")
            
            result = rename_file(original, "The Wire", 2, 3)
            
            assert result["success"] is True
            assert result["new_path"].endswith(".mp4")

    def test_rename_fails_gracefully_for_missing_file(self):
        """Missing file returns error without crashing."""
        result = rename_file("/nonexistent/path/video.mkv", "Series", 1, 1)
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_rename_fails_when_target_exists(self):
        """When target file already exists, returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "original.mkv")
            target = os.path.join(tmpdir, "Breaking Bad S01E05.mkv")
            
            with open(original, 'w') as f:
                f.write("original")
            with open(target, 'w') as f:
                f.write("existing")
            
            result = rename_file(original, "Breaking Bad", 1, 5)
            
            assert result["success"] is False
            assert "already exists" in result["error"]
            # Original file should be untouched
            assert os.path.exists(original)

    def test_rename_fails_for_null_season(self):
        """Null season returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "video.mkv")
            with open(original, 'w') as f:
                f.write("test")
            
            result = rename_file(original, "Series", None, 5)
            
            assert result["success"] is False
            assert "error" in result

    def test_rename_fails_for_null_episode(self):
        """Null episode returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "video.mkv")
            with open(original, 'w') as f:
                f.write("test")
            
            result = rename_file(original, "Series", 1, None)
            
            assert result["success"] is False
            assert "error" in result

    def test_custom_format(self):
        """Custom format string is applied correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "video.mkv")
            with open(original, 'w') as f:
                f.write("test")
            
            result = rename_file(
                original, "The Wire", 2, 5,
                rename_format="{series} - {season}x{episode:02d}"
            )
            
            assert result["success"] is True
            assert "The Wire - 2x05.mkv" in result["new_path"]


class TestRenameFilesFromBatchResults:
    """Tests for batch rename operations."""

    def test_rename_batch_skips_nulls(self):
        """Batch rename skips entries with null season/episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "ep1.mkv")
            file2 = os.path.join(tmpdir, "ep2.mkv")
            for f in [file1, file2]:
                with open(f, 'w') as fh:
                    fh.write("test")
            
            batch_results = [
                {"video_file_path": file1, "season": 1, "episode": 1, "input_file_name": "ep1.mkv"},
                {"video_file_path": file2, "season": None, "episode": None, "input_file_name": "ep2.mkv"},
            ]
            
            results = rename_files_from_batch_results(batch_results, "Series")
            
            # First should succeed, second should be skipped
            assert results[0]["success"] is True
            assert results[1]["skipped"] is True

    def test_rename_batch_skips_duplicates(self):
        """Batch rename skips entries marked as duplicates."""
        batch_results = [
            {
                "input_file_name": "ep1.mkv",
                "duplicate_of": "ep2.mkv"
            }
        ]
        
        results = rename_files_from_batch_results(batch_results, "Series")
        
        assert len(results) == 1
        assert results[0]["skipped"] is True
        assert "duplicate" in results[0]["reason"].lower()

    def test_dry_run_returns_preview_without_changes(self):
        """Dry run mode returns preview without modifying files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "episode.mkv")
            with open(original, 'w') as f:
                f.write("test")
            
            batch_results = [
                {"video_file_path": original, "season": 1, "episode": 5, "input_file_name": "episode.mkv"}
            ]
            
            results = rename_files_from_batch_results(
                batch_results, "Breaking Bad", dry_run=True
            )
            
            # Result should indicate dry run
            assert results[0]["dry_run"] is True
            assert "Breaking Bad S01E05.mkv" in results[0]["would_rename_to"]
            
            # Original file should still exist
            assert os.path.exists(original)
