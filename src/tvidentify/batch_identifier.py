
import argparse
import json
import os
from .subtitle_extractor import extract_subtitles, check_required_tools
from .episode_identifier import identify_episode
from .file_renamer import rename_file

def get_subtitle_fingerprint(video_file, subtitle_track_index, offset_minutes, scan_duration_minutes, num_events=20):
    """
    Get a fingerprint of extracted subtitles for duplicate detection.
    
    Uses the new FFmpeg-based extractor to extract a few subtitles as a fingerprint.
    
    Args:
        video_file: Path to the video file
        subtitle_track_index: Subtitle track index (ignored, finder uses English automatically)
        offset_minutes: Offset in minutes
        scan_duration_minutes: Duration to scan in minutes
        num_events: Number of subtitles to extract for fingerprint (default: 20)
    
    Returns:
        tuple: ((fingerprint as tuple of subtitle hashes, subtitles), or (None, None) if error)
    """
    try:
        # Extract subtitles to use as fingerprint
        subtitles = extract_subtitles(
            video_file,
            subtitle_track_index=subtitle_track_index,
            offset_minutes=offset_minutes,
            max_frames=num_events,
            scan_duration_minutes=scan_duration_minutes
        )
        
        if not subtitles:
            return None, None
        
        # Create a fingerprint from the subtitle hashes
        fingerprint = tuple(hash(sub) for sub in subtitles[:num_events])
        return fingerprint, subtitles  # Return both fingerprint and extracted subtitles
    except Exception as e:
        print(f"  Error generating fingerprint: {e}")
        return None, None

def is_already_named(filename, series_name, rename_format="{series} S{season:02d}E{episode:02d}"):
    """
    Check if a filename matches the expected naming format.
    
    Args:
        filename (str): The filename to check (without path)
        series_name (str): The expected series name
        rename_format (str): The expected rename format pattern
    
    Returns:
        bool: True if the filename matches the expected format, False otherwise
    """
    import re
    
    # Get the filename without extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Build a regex pattern from the rename_format string
    # Replace placeholders with regex groups
    pattern = re.escape(rename_format)
    pattern = pattern.replace(r'\{series\}', re.escape(series_name))
    pattern = re.sub(r'\\\{season.*?\\\}', r'\\d+', pattern)
    pattern = re.sub(r'\\\{episode.*?\\\}', r'\\d+', pattern)
    pattern = f'^{pattern}$'
    
    # Check if the filename matches the expected format
    try:
        match = re.match(pattern, name_without_ext, re.IGNORECASE)
        if match:
            return True
    except re.error as e:
        # If regex fails, fall back to false
        print(f"    [DEBUG] Regex error: {e}")
    
    return False

def find_episode_files(directory, extension=".mkv", size_threshold=0.7):
    """
    Finds likely episode files in a directory based on file size.

    Args:
        directory (str): The path to the directory to scan.
        extension (str): The file extension to look for (default: .mkv).
        size_threshold (float): Files with a size less than this percentage of the
                              largest file will be excluded.

    Returns:
        list[str]: A list of full paths to the likely episode files.
    """
    files_with_sizes = []
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(extension):
            try:
                size = entry.stat().st_size
                if size > 0:
                    files_with_sizes.append((entry.path, size))
            except OSError:
                # Ignore files that can't be stat'd
                continue

    if not files_with_sizes:
        return []

    # Find the size of the largest file
    max_size = max(size for _, size in files_with_sizes)
    size_limit = max_size * size_threshold

    # Filter for files that are close in size to the largest
    episode_files = [path for path, size in files_with_sizes if size >= size_limit]
    episode_files.sort()
    return episode_files

def main():
    """
    Main function to run the batch identification process.
    """
    parser = argparse.ArgumentParser(
        description='Batch identify TV show episodes in a directory and rename them to match Plex TV episode naming.'
    )
    parser.add_argument('input_dir', help='The directory containing video files.')
    parser.add_argument('--series-name', required=True, help='The name of the TV series.')
    parser.add_argument(
        '--size-threshold', type=float, default=0.7,
        help='Size similarity threshold for filtering episodes (default: 0.7).'
    )
    parser.add_argument('--provider', type=str, default='google', choices=['google', 'openai', 'perplexity'],
                        help='LLM provider to use (default: google).')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name. If not provided, defaults based on provider.')
    # Arguments for subtitle extraction, mirroring episode_identifier.py
    parser.add_argument(
        '--max-frames', type=int, default=10,
        help='Maximum number of subtitle events to process (default: 10).'
    )
    parser.add_argument(
        '--subtitle-track', type=int, default=0,
        help='The subtitle track index to use (default: 0).'
    )
    parser.add_argument(
        '--offset', type=int, default=0,
        help='Skip the first N minutes for subtitle extraction (default: 0).'
    )
    parser.add_argument(
        '--scan-duration', type=int, default=15,
        help='How many minutes to scan for subtitles from the offset (default: 15).'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Optional directory to save JSON output files (one per video) instead of printing to console.'
    )
    parser.add_argument(
        '--rename', action='store_true',
        help='Rename files to "<series_name> S<season>E<episode>" format if identification is successful.'
    )
    parser.add_argument(
        '--rename-format', type=str, default="{series} S{season:02d}E{episode:02d}",
        help='Format for renamed files. Available placeholders: {{series}}, {{season}}, {{episode}}. '
             'Default: "{{series}} S{{season:02d}}E{{episode:02d}}"'
    )
    parser.add_argument(
        '--skip-already-named', action='store_true',
        help='Skip files that are already in the expected naming format (only when --rename is specified).'
    )
    args = parser.parse_args()

    # Check for required tools once at startup
    print("Checking for required tools...")
    if not check_required_tools():
        print("\nError: Not all required tools are available. Please install the missing tools and try again.")
        return

    # Set default model based on provider
    if args.model is None:
        if args.provider == 'google':
            args.model = 'gemini-2.5-flash'
        elif args.provider == 'openai':
            args.model = 'gpt-4'
        elif args.provider == 'perplexity':
            args.model = 'sonar-pro'

    # Find potential episode files based on size
    episode_files = find_episode_files(args.input_dir, size_threshold=args.size_threshold)

    if not episode_files:
        print(f"No likely episode files found in '{args.input_dir}'.")
        return

    print(f"Found {len(episode_files)} potential episode files. Processing...")

    all_results = []
    fingerprint_cache = {}  # Maps fingerprint -> (filename, result)
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for video_file in episode_files:
        filename = os.path.basename(video_file)
        
        # Skip already-named files if requested and rename is enabled
        if args.skip_already_named and args.rename:
            if is_already_named(filename, args.series_name, args.rename_format):
                print(f"\n--- Skipping: {filename} (already in expected format) ---")
                result = {
                    "input_file_name": filename,
                    "video_file_path": video_file,
                    "skipped": True,
                    "reason": "Already in expected naming format"
                }
                all_results.append(result)
                continue
        
        print(f"\n--- Processing: {filename} ---")
        result = {
            "input_file_name": filename,
            "video_file_path": video_file  # Store full path for renaming
        }

        # 1. Generate fingerprint for duplicate detection and extract subtitles
        print(f"  Generating subtitle fingerprint for duplicate detection...")
        fingerprint, subtitles = get_subtitle_fingerprint(
            video_file,
            args.subtitle_track,
            args.offset,
            args.scan_duration,
            num_events=args.max_frames  # Use max_frames to extract the desired number of subtitles
        )
        
        if fingerprint is None:
            result["error"] = "Could not generate subtitle fingerprint."
            all_results.append(result)
            continue
        
        # 2. Check if this is a duplicate
        if fingerprint in fingerprint_cache:
            original_filename, original_result = fingerprint_cache[fingerprint]
            result["duplicate_of"] = original_filename
            result["season"] = original_result.get("season")
            result["episode"] = original_result.get("episode")
            result["subtitles"] = original_result.get("subtitles", [])
            result["provider"] = args.provider
            result["model"] = args.model
            print(f"  Duplicate detected! Matches: {original_filename}")
            all_results.append(result)
            continue

        # 3. Identify episode if subtitles were found
        if subtitles:
            id_result = identify_episode(args.series_name, subtitles, model=args.model, provider=args.provider)
            result.update(id_result)
            result["subtitles"] = subtitles
            result["provider"] = args.provider
            result["model"] = args.model
            # Cache this result for future duplicates
            fingerprint_cache[fingerprint] = (os.path.basename(video_file), result)
            
            # 4. Rename file if requested and identification was successful
            if args.rename and result.get("season") is not None and result.get("episode") is not None:
                rename_result = rename_file(
                    video_file,
                    args.series_name,
                    result["season"],
                    result["episode"],
                    args.rename_format
                )
                result["rename"] = rename_result
                if rename_result["success"]:
                    print(f"  Renamed to: {os.path.basename(rename_result['new_path'])}")
                else:
                    print(f"  Rename failed: {rename_result['error']}")
        else:
            result["error"] = "Could not extract subtitles."

        all_results.append(result)
        
        # Save individual result immediately after processing each file
        if args.output_dir:
            if "input_file_name" in result:
                base_name = os.path.splitext(result["input_file_name"])[0]
                individual_file = os.path.join(args.output_dir, f"{base_name}_result.json")
                try:
                    with open(individual_file, 'w') as f:
                        json.dump(result, f, indent=2)
                except IOError as e:
                    print(f"  Warning: Could not save result for {result['input_file_name']}: {e}")

    # Print the final JSON output
    print("\n--- Batch Identification Complete ---")
    
    # Save/print batch results summary
    if args.output_dir:
        output_file = os.path.join(args.output_dir, "batch_results.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Batch summary saved to: {output_file}")
        except IOError as e:
            print(f"Error saving batch results summary: {e}")
    else:
        # Print to console if no output_dir specified
        print(json.dumps(all_results, indent=2))

if __name__ == '__main__':
    main()
