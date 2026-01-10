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

from .pgsreader import PGSReader
from .imagemaker import make_image

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
        pass


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


class PGSHandler(SubtitleHandler):
    """Handler for PGS/SUP bitmap subtitles (Blu-ray). Requires OCR."""
    
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
    # Future handlers:
    # 'dvd_subtitle': VobSubHandler,
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
