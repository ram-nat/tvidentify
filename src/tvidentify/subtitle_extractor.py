import subprocess
import os
import cv2
import pytesseract
import numpy as np
import re
import json
import tempfile
from PIL import Image
from .pgsreader import PGSReader
from .imagemaker import make_image


def clean_subtitle_text(text):
    """
    Cleans OCR output: fixes |/I errors, removes SDH tags, and strips whitespace.
    """
    if not text: 
        return ""
    
    text = text.strip()
    
    # Fix common | vs I errors at start of lines
    text = re.sub(r'^\|', 'I', text) 
    text = re.sub(r'(?<=\n)\|', 'I', text)
    
    # Fix common "l" vs "I" errors
    text = text.replace("l'm", "I'm").replace("l'll", "I'll")

    # Remove SDH (Hearing Impaired) tags like (Music), [Screams]
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    
    # Remove musical notes
    text = text.replace('♪', '')

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def ocr_image(cv_img):
    """
    Performs OCR on a single PGS bitmap (OpenCV format).
    """
    # 1. Handle Transparency (PGS is RGBA)
    # We invert the alpha channel: Text (opaque) -> Black, Background (transparent) -> White
    if cv_img.shape[2] == 4:
        alpha = cv_img[:, :, 3]
        processed_img = cv2.bitwise_not(alpha)
    else:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        processed_img = cv2.bitwise_not(processed_img)

    # 2. Upscale (Critical for accuracy)
    scale_factor = 3
    height, width = processed_img.shape
    processed_img = cv2.resize(processed_img, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

    # 3. Add Padding (White Border)
    processed_img = cv2.copyMakeBorder(processed_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    # 4. Run OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    
    return clean_subtitle_text(text)


def get_subtitle_tracks(video_file):
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
        print(f"Error getting subtitle track info: {e}")
        return []
    
    subtitle_streams = []
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'subtitle':
            subtitle_streams.append(stream)
            
    return subtitle_streams


def find_english_subtitle_stream(video_file):
    """
    Find the first English subtitle stream index using ffprobe.
    Returns the stream index, or None if not found.
    """
    subtitle_tracks = get_subtitle_tracks(video_file)
    
    if not subtitle_tracks:
        print("No subtitle tracks found.")
        return None
    
    print("Found subtitle tracks:")
    for i, track in enumerate(subtitle_tracks):
        codec_name = track.get('codec_name')
        lang = track.get('tags', {}).get('language', 'eng')
        stream_index = track.get('index')
        print(f"  Stream {stream_index} (Track {i}): Codec: {codec_name}, Language: {lang}")
        
        # Check if this is an English subtitle stream
        if lang.lower().startswith('en'):
            print(f"Using English subtitle stream at index {stream_index}")
            return stream_index
    
    # If no English subtitle found, use the first one
    if subtitle_tracks:
        stream_index = subtitle_tracks[0].get('index')
        print(f"No English subtitle found. Using first subtitle stream at index {stream_index}")
        return stream_index
    
    return None


def extract_sup_file(video_file, output_sup_path, subtitle_stream_index, offset_minutes=0, scan_duration_minutes=15):
    """
    Use ffmpeg to extract a subtitle stream to a SUP file.
    
    Args:
        video_file: Path to the input video file
        output_sup_path: Path where the SUP file should be saved
        subtitle_stream_index: The ffprobe stream index of the subtitle (e.g., 0:s:1 for second subtitle stream)
        offset_minutes: Skip the first N minutes
        scan_duration_minutes: How many minutes to scan for subtitles
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Calculate start and end times
        start_time = offset_minutes * 60  # Convert to seconds
        duration = scan_duration_minutes * 60  # Convert to seconds
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_file,
            '-t', str(duration),
            '-map', f'0:{subtitle_stream_index}',
            '-c', 'copy',
            '-f', 'sup',
            output_sup_path,
            '-y'
        ]
        
        print(f"Extracting subtitle stream to SUP file...")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        
        if os.path.exists(output_sup_path) and os.path.getsize(output_sup_path) > 0:
            print(f"Successfully created SUP file: {output_sup_path}")
            return True
        else:
            print(f"Failed to create SUP file or file is empty.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error extracting SUP file (ffmpeg exited with code {e.returncode}).")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in your PATH. Please install it.")
        return False


def extract_text_from_sup(sup_file_path, max_subtitles=None):
    """
    Extracts text from SUP file up to `max_subtitles` entries.
    
    Args:
        sup_file_path: Path to the SUP file
        max_subtitles: Maximum number of subtitles to extract
    
    Returns:
        list[str]: List of extracted subtitle strings
    """
    try:
        pgs = PGSReader(sup_file_path)
        subtitles = []
        count = 0
        
        # Iterate through Display Sets
        for ds in pgs.iter_displaysets():
            
            # Stop if we hit the limit
            if max_subtitles is not None and count >= max_subtitles:
                break

            # Only process if this display set has an image (start of a subtitle)
            if ds.has_image:
                try:
                    pil_image = make_image(
                        ods=ds.ods[0],
                        pds=ds.pds[0],
                    )
                    
                    if pil_image:
                        # Convert PIL (RGBA) -> OpenCV (BGRA)
                        pil_image = pil_image.convert("RGBA")
                        open_cv_image = np.array(pil_image)
                        open_cv_image = open_cv_image[:, :, ::-1].copy()
                        
                        # Perform OCR
                        text = ocr_image(open_cv_image)
                        
                        # Only add if we actually got text back (ignores empty glitches)
                        if text:
                            subtitles.append(text)
                            print(f"  Extracted subtitle {count + 1}: \"{text}\"")
                            count += 1
                except Exception as e:
                    print(f"  Error processing display set: {e}")
                    continue
        
        return subtitles
        
    except Exception as e:
        print(f"Error reading SUP file: {e}")
        return []


def extract_subtitles(video_file, subtitle_track_index=0, offset_minutes=0, max_frames=None, scan_duration_minutes=15, output_dir=None):
    """
    Extracts subtitles from a video file using FFmpeg and OCR.
    
    This function:
    1. Uses ffprobe to find the English subtitle stream
    2. Uses ffmpeg to extract the subtitle stream to a SUP file
    3. Extracts text from the SUP file using PGSReader and OCR
    
    Args:
        video_file (str): Path to the video file.
        subtitle_track_index (int): The index of the subtitle track to use (ignored, finds English automatically).
        offset_minutes (int): Skip the first N minutes of the video.
        max_frames (int): Maximum number of subtitles to extract.
        scan_duration_minutes (int): How many minutes of the video to scan for subtitles.
        output_dir (str): Optional directory to save JSON output. If None, prints to console.

    Returns:
        list[str]: A list of extracted subtitle strings.
    """
    if not os.path.exists(video_file):
        print(f"Error: File not found at {video_file}")
        return []

    # Find the English subtitle stream
    subtitle_stream_index = find_english_subtitle_stream(video_file)
    if subtitle_stream_index is None:
        print("Error: Could not find a subtitle stream in the video file.")
        return []

    all_subtitles = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sup_file_path = os.path.join(temp_dir, "extracted.sup")
        
        # Extract the subtitle stream to a SUP file
        if not extract_sup_file(
            video_file,
            sup_file_path,
            subtitle_stream_index,
            offset_minutes=offset_minutes,
            scan_duration_minutes=scan_duration_minutes
        ):
            print("Failed to extract subtitle stream to SUP file.")
            return []
        
        # Extract text from the SUP file
        print("\nPerforming OCR on subtitle frames...")
        all_subtitles = extract_text_from_sup(sup_file_path, max_subtitles=max_frames)

    # Save to JSON if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Create a safe filename from the video file
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_subtitles.json")
        
        output_data = {
            "source_file": video_file,
            "subtitle_track_index": subtitle_stream_index,
            "offset_minutes": offset_minutes,
            "scan_duration_minutes": scan_duration_minutes,
            "subtitles": all_subtitles
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nJSON output saved to: {output_file}")
        except IOError as e:
            print(f"Error saving JSON output: {e}")

    return all_subtitles


def get_subtitle_events(video_file, subtitle_track_index, read_interval=None):
    """
    Stub function to maintain compatibility with batch_identifier.py.
    The new FFmpeg-based extractor doesn't need this function.
    """
    # This is kept for backward compatibility but not used by the new extractor
    pass


def check_required_tools():
    """
    Check if required tools are installed: ffmpeg, ffprobe, and tesseract.
    
    Returns:
        bool: True if all tools are available, False otherwise
    """
    tools = [
        ('ffmpeg', 'ffmpeg', '-version'),
        ('ffprobe', 'ffprobe', '-version'),
        ('tesseract', 'Tesseract OCR', '--version')
    ]
    
    all_available = True
    for tool_cmd, tool_name, arg in tools:
        try:
            subprocess.run([tool_cmd, arg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {tool_name} is available")
        except FileNotFoundError:
            print(f"✗ Error: {tool_name} is not installed or not in your PATH. Please install it.")
            all_available = False
    
    return all_available


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract subtitles from a video file using FFmpeg and OCR.')
    parser.add_argument('input_file', help='The input video file.')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of subtitles to extract.')
    parser.add_argument('--subtitle-track', type=int, default=0, help='The subtitle track index to use (ignored, finds English automatically).')
    parser.add_argument('--offset', type=int, default=0, help='Skip the first N minutes of the video.')
    parser.add_argument('--scan-duration', type=int, default=15, help='How many minutes of the video to scan for subtitles from the offset (default: 15).')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional directory to save JSON output instead of printing to console.')

    args = parser.parse_args()

    # Check for required tools once at startup
    print("Checking for required tools...")
    if not check_required_tools():
        print("\nError: Not all required tools are available. Please install the missing tools and try again.")
        return

    extracted_subtitles = extract_subtitles(
        video_file=args.input_file,
        subtitle_track_index=args.subtitle_track,
        offset_minutes=args.offset,
        max_frames=args.max_frames,
        scan_duration_minutes=args.scan_duration,
        output_dir=args.output_dir
    )

    print("\n--- All Extracted Subtitles ---")
    if not extracted_subtitles:
        print("No subtitles were found.")
    else:
        for sub in extracted_subtitles:
            print(sub)
            print("---")


if __name__ == '__main__':
    main()
