"""
Subtitle format handlers for different subtitle codecs.

This module provides an extensible architecture for handling different subtitle formats:
- PGS (Blu-ray bitmap subtitles) - requires OCR
- SRT (SubRip text subtitles) - direct text extraction
- Future: VobSub, ASS, SSA, etc.
"""

import subprocess
import os
import re
import tempfile
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

from PIL import Image

from .pgsreader import PGSReader
from .imagemaker import make_image
from .imagemaker import make_image
from .vobsubreader import VobSubReader
from .utils import check_ocr_tools, check_vobsub_tools

logger = logging.getLogger(__name__)


# =============================================================================
# Subtitle Handler Base Class
# =============================================================================

class SubtitleHandler(ABC):
    """Abstract base class for subtitle format handlers."""
    
    @abstractmethod
    def extract_text(
        self,
        video_file: str,
        stream_index: int,
        offset_minutes: int = 0,
        scan_duration_minutes: int = 15,
        max_subtitles: Optional[int] = None,
    ) -> List[str]:
        """
        Extract subtitle text from a video file.
        
        Args:
            video_file: Path to the video file
            stream_index: The ffprobe stream index of the subtitle track
            offset_minutes: Skip the first N minutes
            scan_duration_minutes: How many minutes to scan
            max_subtitles: Maximum number of subtitles to extract
            
        Returns:
            List of extracted subtitle strings
        """
        return []

    def check_tools(self) -> bool:
        """
        Check if required tools for this handler are available.
        Can be overridden by subclasses to add specific checks.
        
        Returns:
            bool: True if tools are available, False otherwise.
        """
        return True


# =============================================================================
# PGS Handler (Blu-ray bitmap subtitles)
# =============================================================================

def clean_subtitle_text(text: str) -> str:
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


def ocr_image(cv_img: np.ndarray) -> str:
    """
    Performs OCR on a single PGS bitmap (OpenCV format).
    """
    # 1. Handle Transparency (PGS is RGBA)
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
    processed_img = cv2.resize(
        processed_img, 
        (width * scale_factor, height * scale_factor), 
        interpolation=cv2.INTER_CUBIC
    )

    # 3. Add Padding (White Border)
    processed_img = cv2.copyMakeBorder(
        processed_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )

    # 4. Run OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    
    return clean_subtitle_text(text)


def ocr_vobsub_image(pil_img: Image.Image, debug_dir: Optional[str] = None, image_name: str = "") -> str:
    """
    OCR a VobSub bitmap with preprocessing optimized for the improved decoder.
    
    The VobSub decoder produces clean RGBA images where:
    - Text has alpha=255 (fully opaque)
    - Background has alpha=0 (transparent)
    
    Pipeline:
    1. Extract alpha channel (text mask)
    2. Upscale 3x for ~300 DPI
    3. Threshold and invert (black text on white)
    4. Add padding
    5. OCR with Tesseract
    
    Args:
        pil_img: PIL Image (RGBA from VobSubReader)
        debug_dir: Optional directory to save debug images
        image_name: Prefix for debug image filenames (e.g., "sub_00012345")
        
    Returns:
        Extracted and cleaned text string
    """

    def alpha_is_mask(alpha: np.ndarray) -> bool:
        # A real subtitle mask should have background near 0 and text near 255.
        return (alpha.min() <= 5) and (alpha.max() >= 250) and (alpha.min() != alpha.max())

    # Save raw image if debug mode enabled
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        raw_path = os.path.join(debug_dir, f"{image_name}_raw.png")
        pil_img.save(raw_path)
    
    img_array = np.array(pil_img)
    
    # Handle different input formats
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA image - use alpha channel as text mask
        alpha = img_array[:, :, 3]
        
        # Skip if fully transparent
        if np.max(alpha) == 0:
            return ""
        
        if alpha_is_mask(alpha):
            gray = alpha
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
    elif len(img_array.shape) == 3:
        # RGB without alpha - convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        # Already grayscale
        gray = img_array
    
    # Upscale 3x for better OCR (target ~300 DPI)
    scale_factor = 3
    upscaled = cv2.resize(
        gray,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    
    # Threshold to binary
    _, binary = cv2.threshold(upscaled, 127, 255, cv2.THRESH_BINARY)

    # after threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    # if text is white on black (binary): close first, then open
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Invert: Tesseract expects black text on white background
    # Alpha channel has text=255 (white), we need text=0 (black)
    inverted = cv2.bitwise_not(binary)

    # Add padding (Tesseract needs space around text)
    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, 
        cv2.BORDER_CONSTANT, value=255
    )
    
    # Save processed image if debug mode enabled
    if debug_dir:
        processed_path = os.path.join(debug_dir, f"{image_name}_processed.png")
        cv2.imwrite(processed_path, padded)
    
    # OCR with optimal config
    # OEM 3: LSTM + legacy (best accuracy)
    # PSM 6: Assume uniform block of text (works better for multi-line subtitles)
    text = pytesseract.image_to_string(padded, config='--oem 3 --psm 6')
    
    return clean_subtitle_text(text)


class PGSHandler(SubtitleHandler):
    """Handler for PGS/SUP bitmap subtitles (Blu-ray). Requires OCR."""
    
    def check_tools(self) -> bool:
        """Checks for OCR tools (Tesseract)."""
        return check_ocr_tools()

    def extract_text(
        self,
        video_file: str,
        stream_index: int,
        offset_minutes: int = 0,
        scan_duration_minutes: int = 15,
        max_subtitles: Optional[int] = None,
    ) -> List[str]:
        """Extract text from PGS subtitles using FFmpeg extraction and OCR."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sup_file_path = os.path.join(temp_dir, "extracted.sup")
            
            # Extract SUP file using FFmpeg
            if not self._extract_sup_file(
                video_file, sup_file_path, stream_index,
                offset_minutes, scan_duration_minutes
            ):
                return []
            
            # OCR the SUP file
            return self._extract_text_from_sup(sup_file_path, max_subtitles)
    
    def _extract_sup_file(
        self,
        video_file: str,
        output_sup_path: str,
        stream_index: int,
        offset_minutes: int,
        scan_duration_minutes: int,
    ) -> bool:
        """Use ffmpeg to extract a subtitle stream to a SUP file."""
        try:
            start_time = offset_minutes * 60
            duration = scan_duration_minutes * 60
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_file,
                '-t', str(duration),
                '-map', f'0:{stream_index}',
                '-c', 'copy',
                '-f', 'sup',
                output_sup_path,
                '-y'
            ]
            
            logger.info("Extracting PGS subtitle stream to SUP file...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(output_sup_path) and os.path.getsize(output_sup_path) > 0:
                logger.debug("Successfully created SUP file: %s", output_sup_path)
                return True
            else:
                logger.error("Failed to create SUP file or file is empty.")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error("Error extracting SUP file: %s", e.stderr)
            return False
        except FileNotFoundError:
            logger.error("ffmpeg is not installed or not in your PATH.")
            return False
    
    def _extract_text_from_sup(
        self, 
        sup_file_path: str, 
        max_subtitles: Optional[int] = None
    ) -> List[str]:
        """Extracts text from SUP file using PGSReader and OCR."""
        try:
            pgs = PGSReader(sup_file_path)
            subtitles = []
            count = 0
            
            for ds in pgs.iter_displaysets():
                if max_subtitles is not None and count >= max_subtitles:
                    break

                if ds.has_image:
                    try:
                        pil_image = make_image(ods=ds.ods[0], pds=ds.pds[0])
                        
                        if pil_image:
                            # Convert PIL (RGBA) -> OpenCV (BGRA)
                            pil_image = pil_image.convert("RGBA")
                            open_cv_image = np.array(pil_image)
                            open_cv_image = open_cv_image[:, :, ::-1].copy()
                            
                            text = ocr_image(open_cv_image)
                            
                            if text:
                                subtitles.append(text)
                                logger.debug("Extracted subtitle %d: \"%s\"", count + 1, text)
                                count += 1
                    except Exception as e:
                        logger.warning("Error processing display set: %s", e)
                        continue
            
            return subtitles
            
        except Exception as e:
            logger.error("Error reading SUP file: %s", e)
            return []


# =============================================================================
# VobSub Handler (DVD bitmap subtitles)
# =============================================================================

class VobSubHandler(SubtitleHandler):
    """Handler for VobSub (DVD) bitmap subtitles. Requires OCR."""
    
    def check_tools(self) -> bool:
        """Checks for OCR tools (Tesseract) and VobSub extraction tools (mkvextract)."""
        # Run all checks so user sees all missing dependencies at once
        ocr_ok = check_ocr_tools()
        vobsub_tools_ok = check_vobsub_tools()
        return ocr_ok and vobsub_tools_ok

    def extract_text(
        self,
        video_file: str,
        stream_index: int,
        offset_minutes: int = 0,
        scan_duration_minutes: int = 15,
        max_subtitles: Optional[int] = None,
        debug_dir: Optional[str] = None,
    ) -> List[str]:
        """Extract text from VobSub subtitles using mkvextract and OCR."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            idx_file_path = os.path.join(temp_dir, "extracted.idx")
            sub_file_path = os.path.join(temp_dir, "extracted.sub")
            
            # Extract VobSub files using mkvextract
            if not self._extract_vobsub_files(
                video_file, idx_file_path, stream_index,
                offset_minutes, scan_duration_minutes
            ):
                return []
            
            # OCR the VobSub files
            return self.extract_from_idx(
                idx_file_path, 
                max_subtitles=max_subtitles,
                offset_seconds=offset_minutes * 60,
                duration_seconds=scan_duration_minutes * 60,
                debug_dir=debug_dir
            )
    
    def _extract_vobsub_files(
        self,
        video_file: str,
        output_idx_path: str,
        stream_index: int,
        offset_minutes: int,
        scan_duration_minutes: int,
    ) -> bool:
        """
        Use mkvextract to extract a VobSub subtitle stream to idx/sub files.
        
        Note: mkvextract uses track IDs (from mkvmerge --identify) which differ
        from ffmpeg stream indices. We need to find the correct track ID.
        """
        try:
            import json
            
            # Step 1: Get track ID from mkvmerge --identify
            track_id = self._get_mkvextract_track_id(video_file, stream_index)
            if track_id is None:
                logger.error("Could not find track ID for stream index %d", stream_index)
                return False
            
            logger.info("Found mkvextract track ID %d for ffmpeg stream %d", track_id, stream_index)
            
            # Step 2: Extract using mkvextract
            # mkvextract outputs .idx and .sub files when extracting VobSub
            output_base = output_idx_path.rsplit('.', 1)[0]
            
            mkvextract_cmd = [
                'mkvextract', 'tracks', video_file,
                f'{track_id}:{output_base}'
            ]
            
            logger.info("Extracting VobSub with mkvextract...")
            result = subprocess.run(
                mkvextract_cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            # mkvextract creates .idx and .sub files
            idx_path = output_base + '.idx'
            sub_path = output_base + '.sub'
            
            if os.path.exists(idx_path) and os.path.exists(sub_path):
                logger.debug("Successfully created VobSub files: %s", idx_path)
                return True
            else:
                logger.error("mkvextract did not create expected VobSub files")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error("Error extracting VobSub files: %s", e.stderr)
            return False
        except FileNotFoundError:
            logger.error("mkvextract is not installed or not in your PATH. Install mkvtoolnix.")
            return False
    
    def _get_mkvextract_track_id(self, video_file: str, ffmpeg_stream_index: int) -> Optional[int]:
        """
        Get the mkvextract track ID corresponding to an ffmpeg stream index.
        
        mkvextract track IDs are 0-based and only count actual tracks,
        while ffmpeg stream indices include all streams.
        
        We use mkvmerge --identify -J to get track info.
        """
        import json
        
        try:
            result = subprocess.run(
                ['mkvmerge', '--identify', '-J', video_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            info = json.loads(result.stdout)
            tracks = info.get('tracks', [])
            
            # Find subtitle tracks and match to ffmpeg stream index
            # ffmpeg orders streams as: video, audio, subtitle
            # mkvmerge reports tracks in file order
            
            # Count how many streams of each type come before our target
            # to find the matching mkvextract track ID
            for track in tracks:
                track_id = track.get('id')
                track_type = track.get('type')
                properties = track.get('properties', {})
                
                # mkvmerge track ID + 1 typically equals stream position
                # But subtitle tracks specifically: we need to find the one
                # that matches our ffmpeg stream index
                
                # Simple approach: subtitle track with matching codec
                if track_type == 'subtitles':
                    codec = properties.get('codec_id', '')
                    # VobSub is identified as S_VOBSUB
                    if 'VOBSUB' in codec.upper():
                        # Check if this track's position matches ffmpeg index
                        # mkvmerge lists tracks in order, so we can use track_id
                        # as an approximation
                        
                        # For MKV files, track IDs typically align with 
                        # ffmpeg stream indices for subtitles
                        # We compare the number property if available
                        track_num = properties.get('number', track_id + 1)
                        
                        # ffmpeg stream index for subtitles in MKV is usually
                        # the track_id + 1 (accounting for 0-indexing)
                        if track_id == ffmpeg_stream_index or track_num - 1 == ffmpeg_stream_index:
                            return track_id
            
            # Fallback: find any subtitle track with matching index pattern
            subtitle_count = 0
            for track in tracks:
                if track.get('type') == 'subtitles':
                    track_id = track.get('id')
                    # Try direct match
                    if track_id == ffmpeg_stream_index:
                        return track_id
                    subtitle_count += 1
            
            # Last resort: assume track ID equals stream index
            logger.warning("Could not verify track ID, using stream index as fallback")
            return ffmpeg_stream_index
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.error("Error identifying tracks: %s", e)
            return None
    
    def extract_from_idx(
        self,
        idx_file_path: str,
        max_subtitles: Optional[int] = None,
        offset_seconds: int = 0,
        duration_seconds: Optional[int] = None,
        debug_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Extract text from standalone VobSub idx/sub files.
        
        This is useful for directly OCR'ing extracted VobSub files
        without going through a video file.
        
        Args:
            idx_file_path: Path to the .idx file (expects .sub in same directory)
            max_subtitles: Maximum number of subtitles to extract
            offset_seconds: Skip subtitles before this timestamp
            duration_seconds: Only process subtitles within this duration from offset
            debug_dir: Optional directory to save debug images
            
        Returns:
            List of extracted subtitle strings
        """
        try:
            reader = VobSubReader(idx_file_path)
            subtitles = []
            count = 0
            
            end_time_ms = None
            if duration_seconds is not None:
                end_time_ms = (offset_seconds + duration_seconds) * 1000
            
            for event in reader.iter_events(max_events=None):
                if max_subtitles is not None and count >= max_subtitles:
                    break
                
                # Apply time filters
                if event.timestamp_ms < offset_seconds * 1000:
                    continue
                if end_time_ms is not None and event.timestamp_ms > end_time_ms:
                    break
                
                if event.image:
                    try:
                        # Use timestamp for unique debug image names
                        timestamp_str = f"{event.timestamp_ms:08d}"
                        text = ocr_vobsub_image(
                            event.image,
                            debug_dir=debug_dir,
                            image_name=f"sub_{timestamp_str}"
                        )
                        
                        if text:
                            subtitles.append(text)
                            logger.debug(
                                "Extracted VobSub subtitle %d at %dms: \"%s\"",
                                count + 1, event.timestamp_ms, text
                            )
                            count += 1
                    except Exception as e:
                        logger.warning("Error OCR'ing VobSub image: %s", e)
                        continue
            
            logger.info("Extracted %d subtitles from VobSub", len(subtitles))
            return subtitles
            
        except Exception as e:
            logger.error("Error reading VobSub files: %s", e)
            return []


# =============================================================================
# SRT Handler (Text-based subtitles)
# =============================================================================

class SRTHandler(SubtitleHandler):
    """Handler for SRT/SubRip text subtitles. No OCR needed."""
    
    def extract_text(
        self,
        video_file: str,
        stream_index: int,
        offset_minutes: int = 0,
        scan_duration_minutes: int = 15,
        max_subtitles: Optional[int] = None,
    ) -> List[str]:
        """Extract text from SRT subtitles using FFmpeg."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            srt_file_path = os.path.join(temp_dir, "extracted.srt")
            
            # Extract SRT file using FFmpeg
            if not self._extract_srt_file(
                video_file, srt_file_path, stream_index,
                offset_minutes, scan_duration_minutes
            ):
                return []
            
            # Parse the SRT file
            return self._parse_srt_file(srt_file_path, max_subtitles)
    
    def _extract_srt_file(
        self,
        video_file: str,
        output_srt_path: str,
        stream_index: int,
        offset_minutes: int,
        scan_duration_minutes: int,
    ) -> bool:
        """Use ffmpeg to extract a subtitle stream to an SRT file."""
        try:
            start_time = offset_minutes * 60
            duration = scan_duration_minutes * 60
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_file,
                '-t', str(duration),
                '-map', f'0:{stream_index}',
                '-c:s', 'srt',
                output_srt_path,
                '-y'
            ]
            
            logger.info("Extracting SRT subtitle stream...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                logger.debug("Successfully created SRT file: %s", output_srt_path)
                return True
            else:
                logger.error("Failed to create SRT file or file is empty.")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error("Error extracting SRT file: %s", e.stderr)
            return False
        except FileNotFoundError:
            logger.error("ffmpeg is not installed or not in your PATH.")
            return False
    
    def _parse_srt_file(
        self, 
        srt_file_path: str, 
        max_subtitles: Optional[int] = None
    ) -> List[str]:
        """Parse an SRT file and extract subtitle text."""
        subtitles = []
        
        try:
            with open(srt_file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(srt_file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # SRT format: blocks separated by blank lines
        # Each block: index, timestamp, text (can be multiple lines)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if max_subtitles is not None and len(subtitles) >= max_subtitles:
                break
            
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Skip index (line 0) and timestamp (line 1), get text (lines 2+)
                text_lines = lines[2:]
                text = ' '.join(text_lines).strip()
                
                # Clean the text (remove HTML tags, etc.)
                text = self._clean_srt_text(text)
                
                if text:
                    subtitles.append(text)
                    logger.debug("Extracted subtitle: \"%s\"", text)
        
        return subtitles
    
    def _clean_srt_text(self, text: str) -> str:
        """Clean SRT subtitle text."""
        # Remove HTML tags like <i>, </i>, <b>, etc.
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove SDH tags like (Music), [Screams]
        text = re.sub(r'[\(\[].*?[\)\]]', '', text)
        
        # Remove musical notes
        text = text.replace('♪', '')
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# =============================================================================
# Handler Factory
# =============================================================================

# Mapping of codec names to handler classes
CODEC_HANDLERS = {
    'hdmv_pgs_subtitle': PGSHandler,
    'subrip': SRTHandler,
    'dvd_subtitle': VobSubHandler,
    # Future handlers:
    # 'ass': ASSHandler,
    # 'ssa': SSAHandler,
}


def get_handler_for_codec(codec_name: str) -> Optional[SubtitleHandler]:
    """
    Factory function to get the appropriate handler for a subtitle codec.
    
    Args:
        codec_name: The codec name from ffprobe (e.g., 'hdmv_pgs_subtitle', 'subrip')
        
    Returns:
        A SubtitleHandler instance, or None if the codec is not supported.
    """
    handler_class = CODEC_HANDLERS.get(codec_name)
    if handler_class:
        return handler_class()
    return None


def get_supported_codecs() -> List[str]:
    """Return a list of supported subtitle codec names."""
    return list(CODEC_HANDLERS.keys())
