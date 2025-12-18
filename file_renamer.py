import argparse
import os
import json


def rename_file(video_file, series_name, season, episode, rename_format="{series} S{season:02d}E{episode:02d}"):
    """
    Renames a video file based on TV series identification.
    
    Args:
        video_file (str): Path to the video file to rename
        series_name (str): Name of the TV series
        season (int): Season number
        episode (int): Episode number
        rename_format (str): Format string for the new filename. 
                           Default: "{series} S{season:02d}E{episode:02d}"
                           Available placeholders: {series}, {season}, {episode}
    
    Returns:
        dict: Result object with keys:
            - success (bool): Whether rename was successful
            - old_path (str): Original file path
            - new_path (str): New file path (if successful)
            - error (str): Error message (if failed)
    """
    if not os.path.exists(video_file):
        return {
            "success": False,
            "old_path": video_file,
            "error": f"File not found: {video_file}"
        }
    
    if season is None or episode is None:
        return {
            "success": False,
            "old_path": video_file,
            "error": "Season and/or episode information is missing or null"
        }
    
    # Ensure season and episode are integers
    try:
        season = int(season)
        episode = int(episode)
    except (ValueError, TypeError):
        return {
            "success": False,
            "old_path": video_file,
            "error": f"Invalid season ({season}) or episode ({episode}) values"
        }
    
    # Get file extension
    _, ext = os.path.splitext(video_file)
    
    # Format the new filename
    try:
        new_filename = rename_format.format(
            series=series_name,
            season=season,
            episode=episode
        )
    except KeyError as e:
        return {
            "success": False,
            "old_path": video_file,
            "error": f"Invalid rename format - unknown placeholder: {e}"
        }
    
    new_filename = new_filename + ext
    
    # Get directory and construct new path
    directory = os.path.dirname(video_file)
    new_path = os.path.join(directory, new_filename)
    
    # Check if target file already exists
    if os.path.exists(new_path) and os.path.abspath(video_file) != os.path.abspath(new_path):
        return {
            "success": False,
            "old_path": video_file,
            "new_path": new_path,
            "error": f"Target file already exists: {new_path}"
        }
    
    # Perform the rename
    try:
        os.rename(video_file, new_path)
        return {
            "success": True,
            "old_path": video_file,
            "new_path": new_path,
            "series": series_name,
            "season": season,
            "episode": episode
        }
    except OSError as e:
        return {
            "success": False,
            "old_path": video_file,
            "new_path": new_path,
            "error": f"Failed to rename file: {e}"
        }


def rename_files_from_batch_results(batch_results, series_name, rename_format="{series} S{season:02d}E{episode:02d}", dry_run=False):
    """
    Renames multiple files based on batch identification results.
    
    Args:
        batch_results (list): List of identification results from batch_identifier
        series_name (str): Name of the TV series
        rename_format (str): Format string for the new filename
        dry_run (bool): If True, don't actually rename files, just show what would happen
    
    Returns:
        list: List of rename operation results
    """
    rename_results = []
    
    for result in batch_results:
        # Skip if this was a duplicate or had an error
        if "duplicate_of" in result or "error" in result:
            rename_results.append({
                "input_file_name": result.get("input_file_name", "unknown"),
                "skipped": True,
                "reason": result.get("duplicate_of") and f"Duplicate of {result['duplicate_of']}" or result.get("error", "Unknown reason")
            })
            continue
        
        # Check if season and episode are available
        season = result.get("season")
        episode = result.get("episode")
        
        if season is None or episode is None:
            rename_results.append({
                "input_file_name": result.get("input_file_name", "unknown"),
                "skipped": True,
                "reason": "Season and/or episode could not be determined"
            })
            continue
        
        # If we have the file path from batch processing
        video_file = result.get("video_file_path")
        if not video_file:
            rename_results.append({
                "input_file_name": result.get("input_file_name", "unknown"),
                "skipped": True,
                "reason": "File path not available in result"
            })
            continue
        
        # Perform rename
        if dry_run:
            # Just show what would happen
            _, ext = os.path.splitext(video_file)
            new_filename = rename_format.format(series=series_name, season=int(season), episode=int(episode)) + ext
            rename_results.append({
                "input_file_name": result.get("input_file_name", "unknown"),
                "dry_run": True,
                "would_rename_to": new_filename
            })
        else:
            rename_result = rename_file(video_file, series_name, season, episode, rename_format)
            rename_results.append(rename_result)
    
    return rename_results


def main():
    parser = argparse.ArgumentParser(description='Rename video files based on TV episode identification results.')
    parser.add_argument('--batch-results', type=str, required=True,
                        help='Path to batch_results.json from batch_identifier')
    parser.add_argument('--series-name', type=str, required=True,
                        help='Name of the TV series')
    parser.add_argument('--rename-format', type=str, default="{series} S{season:02d}E{episode:02d}",
                        help='Format for the new filename. Available placeholders: {series}, {season}, {episode}. '
                             'Default: "{series} S{season:02d}E{episode:02d}"')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be renamed without actually renaming')
    
    args = parser.parse_args()
    
    # Load batch results
    try:
        with open(args.batch_results, 'r') as f:
            batch_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Batch results file not found: {args.batch_results}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in batch results file: {e}")
        return
    
    if not isinstance(batch_results, list):
        print("Error: Batch results must be an array of results")
        return
    
    # Rename files
    rename_results = rename_files_from_batch_results(
        batch_results,
        args.series_name,
        args.rename_format,
        dry_run=args.dry_run
    )
    
    # Print results
    print("\n--- File Rename Results ---")
    print(json.dumps(rename_results, indent=2))
    
    # Summary
    successful = sum(1 for r in rename_results if r.get("success", False))
    skipped = sum(1 for r in rename_results if r.get("skipped", False))
    failed = sum(1 for r in rename_results if not r.get("success") and not r.get("skipped") and not r.get("dry_run"))
    dry_runs = sum(1 for r in rename_results if r.get("dry_run", False))
    
    print(f"\nSummary:")
    if args.dry_run:
        print(f"  Would rename: {dry_runs}")
    else:
        print(f"  Successful renames: {successful}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
