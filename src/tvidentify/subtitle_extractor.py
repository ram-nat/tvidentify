import subprocess
import os
import argparse
import json
import logging
from typing import List, Dict, Optional, Any, Tuple

from .subtitle_handlers import SubtitleHandler, get_handler_for_codec, get_supported_codecs
from .utils import check_required_tools, setup_logging, add_logging_args

logger = logging.getLogger(__name__)




def get_subtitle_tracks(video_file: str) -> List[Dict[str, Any]]:
    """
    Uses ffprobe to get information about subtitle tracks in the video file.
    Returns the index of the first English subtitle stream, or 0 if not found.
    """
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        video_file
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Error getting subtitle track info: %s", e)
        return []
    
    subtitle_streams = []
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'subtitle':
            subtitle_streams.append(stream)
            
    return subtitle_streams


def find_subtitle_stream(
    video_file: str, 
    subtitle_track_index: Optional[int] = None
) -> Optional[Tuple[int, SubtitleHandler]]:
    """
    Find a suitable subtitle stream and its handler.

    - If subtitle_track_index is provided, validate and use it.
    - If not, find the first supported English subtitle stream.
    - If no English track is found, fall back to the first supported subtitle stream.
    - Return None if no supported subtitle tracks exist.
    
    Returns:
        Tuple of (stream_index, SubtitleHandler) or None if not found/unsupported.
    """
    subtitle_tracks = get_subtitle_tracks(video_file)

    if not subtitle_tracks:
        logger.warning("No subtitle tracks found.")
        return None

    def get_track_with_handler(track: Dict) -> Optional[Tuple[int, SubtitleHandler]]:
        """Helper to get a track's index and handler if supported."""
        codec_name = track.get('codec_name', '')
        handler = get_handler_for_codec(codec_name)
        if handler:
            return (track.get('index'), handler)
        return None

    # If a specific track index is provided, find and validate it
    if subtitle_track_index is not None:
        for track in subtitle_tracks:
            if track.get('index') == subtitle_track_index:
                result = get_track_with_handler(track)
                if result:
                    logger.info("Using user-specified subtitle stream at index %s", subtitle_track_index)
                    return result
                else:
                    codec = track.get('codec_name', 'unknown')
                    logger.error("Subtitle track %s has unsupported codec: %s", subtitle_track_index, codec)
                    return None
        logger.error("Specified subtitle track index %s not found.", subtitle_track_index)
        return None

    # Find the first supported English subtitle stream
    for track in subtitle_tracks:
        lang = track.get('tags', {}).get('language', 'eng')
        if lang.lower().startswith('en'):
            result = get_track_with_handler(track)
            if result:
                logger.info("Found English subtitle stream at index %s (codec: %s)", 
                           result[0], track.get('codec_name'))
                return result

    # Fall back to the first supported subtitle stream
    for track in subtitle_tracks:
        result = get_track_with_handler(track)
        if result:
            logger.warning("No English subtitle found. Using first supported stream at index %s", result[0])
            return result
    
    # No supported subtitle formats found
    codecs = [t.get('codec_name', 'unknown') for t in subtitle_tracks]
    logger.warning("No supported subtitle formats found. Available codecs: %s", codecs)
    return None



def extract_subtitles(
    video_file: str, 
    subtitle_track_index: Optional[int] = None, 
    offset_minutes: int = 0, 
    max_frames: Optional[int] = None, 
    scan_duration_minutes: int = 15, 
    output_dir: Optional[str] = None,
    debug_dir: Optional[str] = None
) -> List[str]:
    """
    Extracts subtitles from a video file.
    
    Automatically detects the subtitle format and uses the appropriate handler:
    - PGS (Blu-ray): Extracts bitmap subtitles and performs OCR
    - SRT (SubRip): Extracts text directly (no OCR needed)
    
    Args:
        video_file: Path to the video file.
        subtitle_track_index: The index of the subtitle track to use.
        offset_minutes: Skip the first N minutes of the video.
        max_frames: Maximum number of subtitles to extract.
        scan_duration_minutes: How many minutes of the video to scan.
        output_dir: Optional directory to save JSON output.
        debug_dir: Optional directory to save debug images (VobSub only).

    Returns:
        List of extracted subtitle strings.
    """
    if not os.path.exists(video_file):
        logger.error("Error: File not found at %s", video_file)
        return []

    # Find a suitable subtitle stream and its handler
    result = find_subtitle_stream(video_file, subtitle_track_index)
    if result is None:
        logger.error("Error: Could not find a suitable subtitle stream in the video file.")
        return []
    
    stream_index, handler = result
    
    # Delegate extraction to the handler
    logger.info("Extracting subtitles using %s...", handler.__class__.__name__)
    
    # Check if required tools for this specific handler are available
    if not handler.check_tools():
        logger.error("Required tools for %s are missing or not found.", handler.__class__.__name__)
        return []

    # VobSubHandler supports debug_dir for saving OCR debug images
    from .subtitle_handlers import VobSubHandler
    if isinstance(handler, VobSubHandler) and debug_dir:
        all_subtitles = handler.extract_text(
            video_file=video_file,
            stream_index=stream_index,
            offset_minutes=offset_minutes,
            scan_duration_minutes=scan_duration_minutes,
            max_subtitles=max_frames,
            debug_dir=debug_dir,
        )
    else:
        all_subtitles = handler.extract_text(
            video_file=video_file,
            stream_index=stream_index,
            offset_minutes=offset_minutes,
            scan_duration_minutes=scan_duration_minutes,
            max_subtitles=max_frames,
        )

    # Save to JSON if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_subtitles.json")
        
        output_data = {
            "source_file": video_file,
            "subtitle_track_index": stream_index,
            "offset_minutes": offset_minutes,
            "scan_duration_minutes": scan_duration_minutes,
            "subtitles": all_subtitles
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info("JSON output saved to: %s", output_file)
        except IOError as e:
            logger.error("Error saving JSON output: %s", e)

    return all_subtitles

def add_extraction_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds standard subtitle extraction arguments to the provided argparse parser.
    """
    group = parser.add_argument_group('Subtitle Extraction')
    group.add_argument('--max-frames', type=int, default=None, help='Maximum number of subtitles to extract.')
    group.add_argument('--subtitle-track', type=int, default=None, help='The subtitle track index to use. If not specified, finds the first English track. If no english subtitle tracks are available, finds the first subtitle track.')
    group.add_argument('--offset', type=int, default=0, help='Skip the first N minutes of the video.')
    group.add_argument('--debug-dir', type=str, default=None, help='Directory to save debug images (VobSub OCR).')
    group.add_argument('--scan-duration', type=int, default=15, help='How many minutes of the video to scan for subtitles from the offset (default: 15).')
    group.add_argument('--output-dir', type=str, default=None, help='Optional directory to save JSON output instead of printing to console.')

def main():
    parser = argparse.ArgumentParser(description='Extract subtitles from a video file using FFmpeg and OCR.')
    parser.add_argument('input_file', help='The input video file.')
    
    add_extraction_args(parser)
    add_logging_args(parser)

    args = parser.parse_args()

    # Determine logging level
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    setup_logging(console_level=log_level, log_file=args.log_file, file_level=log_level)

    # Check for required tools once at startup
    logger.info("Checking for required tools...")
    if not check_required_tools():
        logger.error("Error: Not all required tools are available. Please install the missing tools and try again.")
        return

    extracted_subtitles = extract_subtitles(
        video_file=args.input_file,
        subtitle_track_index=args.subtitle_track,
        offset_minutes=args.offset,
        max_frames=args.max_frames,
        scan_duration_minutes=args.scan_duration,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir
    )

    logger.info("--- All Extracted Subtitles ---")
    if not extracted_subtitles:
        logger.info("No subtitles were found.")
    else:
        for sub in extracted_subtitles:
            logger.info("%s", sub)


if __name__ == '__main__':
    main()
