#!/usr/bin/env python3
"""
VobSub (DVD subtitle) reader for parsing .idx and .sub files.

VobSub format consists of:
- .idx file: Text file with timing, palette, and file positions
- .sub file: Binary file with RLE-encoded subtitle bitmaps

Based on SubtitleEdit's implementation:
https://github.com/SubtitleEdit/subtitleedit/tree/main/src/libse/VobSub
"""

import re
import logging
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass, field
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum dimensions for a valid subtitle image (from SubtitleEdit MergeVobSubPacks)
MIN_IMAGE_WIDTH = 4
MIN_IMAGE_HEIGHT = 3


@dataclass
class VobSubPalette:
    """16-color palette from idx file (RGBA format)."""
    colors: List[Tuple[int, int, int, int]]  # 16 RGBA tuples


@dataclass
class SubPictureInfo:
    """Parsed control sequence data for a subtitle."""
    width: int = 0
    height: int = 0
    colors_used: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    alpha_values: List[int] = field(default_factory=lambda: [255, 255, 255, 255])
    top_field_offset: int = 0
    bottom_field_offset: int = 0
    forced: bool = False


@dataclass
class VobSubEvent:
    """A single subtitle event with timing and image."""
    timestamp_ms: int
    image: Optional[Image.Image]


class VobSubReader:
    """
    Parser for VobSub (.idx + .sub) subtitle files.
    
    Usage:
        reader = VobSubReader("subtitles.idx")  # .sub must be in same directory
        for event in reader.iter_events():
            if event.image:
                # Process event.image (PIL Image)
    """
    
    def __init__(self, idx_filepath: str):
        """
        Initialize the reader with an idx file path.
        The .sub file is expected to be in the same location with same basename.
        """
        self.idx_path = idx_filepath
        self.sub_path = idx_filepath.rsplit('.', 1)[0] + '.sub'
        
        # Parse idx file
        self.palette = self._parse_palette()
        self.events = self._parse_idx_timestamps()
        
        # Load sub file
        with open(self.sub_path, 'rb') as f:
            self.sub_data = f.read()
    
    def _parse_palette(self) -> VobSubPalette:
        """Parse the palette from the idx file."""
        colors = [(0, 0, 0, 255)] * 16  # Default: black, opaque
        
        try:
            with open(self.idx_path, 'r') as f:
                for line in f:
                    if line.startswith('palette:'):
                        # Format: palette: 000000, ffffff, 808080, ...
                        hex_colors = line.split(':', 1)[1].strip()
                        for i, hex_val in enumerate(hex_colors.split(',')):
                            if i >= 16:
                                break
                            hex_val = hex_val.strip()
                            if len(hex_val) == 6:
                                r = int(hex_val[0:2], 16)
                                g = int(hex_val[2:4], 16)
                                b = int(hex_val[4:6], 16)
                                colors[i] = (r, g, b, 255)
                        break
        except Exception as e:
            logger.warning("Failed to parse palette: %s", e)
        
        return VobSubPalette(colors=colors)
    
    def _parse_idx_timestamps(self) -> List[Tuple[int, int]]:
        """
        Parse timestamps and file positions from idx file.
        
        Returns:
            List of (timestamp_ms, filepos) tuples
        """
        events = []
        
        # Pattern: timestamp: HH:MM:SS:mmm, filepos: XXXXXXXX
        pattern = re.compile(
            r'timestamp:\s*(\d{2}):(\d{2}):(\d{2}):(\d{3}),\s*filepos:\s*([0-9a-fA-F]+)'
        )
        
        try:
            with open(self.idx_path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        h, m, s, ms = map(int, match.groups()[:4])
                        timestamp_ms = (h * 3600 + m * 60 + s) * 1000 + ms
                        filepos = int(match.group(5), 16)
                        events.append((timestamp_ms, filepos))
        except Exception as e:
            logger.error("Failed to parse idx file: %s", e)
        
        return events
    
    def iter_events(self, max_events: Optional[int] = None) -> Iterator[VobSubEvent]:
        """
        Iterate through subtitle events, decoding images.
        
        Filters out bad packets based on SubtitleEdit's MergeVobSubPacks criteria:
        - Images with width <= 3 or height <= 2 are discarded.
        
        Args:
            max_events: Maximum number of valid events to yield
            
        Yields:
            VobSubEvent with timestamp and decoded image
        """
        count = 0
        for i, (timestamp_ms, filepos) in enumerate(self.events):
            if max_events is not None and count >= max_events:
                break
            
            # Determine the size of this subtitle packet
            if i + 1 < len(self.events):
                next_pos = self.events[i + 1][1]
                packet_data = self.sub_data[filepos:next_pos]
            else:
                packet_data = self.sub_data[filepos:]
            
            # Decode the subtitle image
            try:
                image = self._decode_subtitle_packet(packet_data)
                
                # Filter bad packets (SubtitleEdit MergeVobSubPacks criteria)
                if image is None:
                    continue
                if image.width < MIN_IMAGE_WIDTH or image.height < MIN_IMAGE_HEIGHT:
                    logger.debug(
                        "Skipping small image at %dms: %dx%d",
                        timestamp_ms, image.width, image.height
                    )
                    continue
                
                yield VobSubEvent(timestamp_ms=timestamp_ms, image=image)
                count += 1
            except Exception as e:
                logger.warning("Failed to decode subtitle at %d: %s", filepos, e)
                continue
    
    def _decode_subtitle_packet(self, data: bytes) -> Optional[Image.Image]:
        """
        Decode a VobSub subtitle packet into a PIL Image.
        
        VobSub packets are MPEG-PS PES packets containing RLE-encoded bitmaps.
        A single subtitle can span multiple PES packets, so we need to merge them.
        """
        if len(data) < 4:
            return None
        
        # Merge multiple PES packets into a single subtitle data buffer
        merged_data = self._merge_pes_packets(data)
        if not merged_data or len(merged_data) < 4:
            return None
        
        # Parse subtitle packet header (from merged data)
        # First 2 bytes: total subtitle packet size
        # Next 2 bytes: offset to control sequence
        packet_size = (merged_data[0] << 8) | merged_data[1]
        ctrl_offset = (merged_data[2] << 8) | merged_data[3]
        
        # Validate control offset
        if ctrl_offset >= len(merged_data):
            logger.debug("Control offset %d beyond data length %d", ctrl_offset, len(merged_data))
            return None
        
        # Parse control sequence to get image info
        sub_info = self._parse_control_sequence(merged_data, ctrl_offset)
        
        if sub_info.width <= 0 or sub_info.height <= 0:
            return None
        
        # Decode RLE bitmap using interlaced field decoding
        # RLE offsets in sub_info are relative to start of merged_data
        return self._decode_rle_bitmap_interlaced(
            data=merged_data,
            base_offset=0,  # Already relative to start of merged_data
            sub_info=sub_info
        )
    
    def _merge_pes_packets(self, data: bytes) -> bytes:
        """
        Merge multiple PES packets into a single subtitle data buffer.
        
        VobSub subtitles can span multiple MPEG-PS PES packets (each max ~2028 bytes).
        This function extracts the payload from each packet and concatenates them.
        """
        result = bytearray()
        pos = 0
        
        while pos < len(data):
            # Check for MPEG-2 Pack header (00 00 01 BA)
            if pos + 14 <= len(data) and data[pos:pos+4] == b'\x00\x00\x01\xba':
                # Skip pack header (14 bytes + stuffing)
                stuffing = data[pos + 13] & 0x07
                pos += 14 + stuffing
                continue
            
            # Check for PES start code (00 00 01 xx)
            if pos + 9 <= len(data) and data[pos:pos+3] == b'\x00\x00\x01':
                stream_id = data[pos + 3]
                
                # Check for padding stream (BE) or program end (B9) - skip
                if stream_id in (0xBE, 0xB9):
                    if stream_id == 0xB9:
                        # Program end
                        break
                    # Padding stream - skip
                    if pos + 6 <= len(data):
                        pes_length = (data[pos + 4] << 8) | data[pos + 5]
                        pos += 6 + pes_length
                    else:
                        pos += 1
                    continue
                
                # Private stream 1 (BD) contains subtitle data
                if stream_id == 0xBD:
                    if pos + 9 > len(data):
                        break
                    
                    pes_length = (data[pos + 4] << 8) | data[pos + 5]
                    pes_header_len = data[pos + 8]
                    
                    # Payload starts after PES header
                    payload_start = pos + 9 + pes_header_len
                    # +1 to skip the substream ID byte 
                    payload_start += 1
                    
                    # Payload ends at PES packet end
                    payload_end = pos + 6 + pes_length
                    
                    if payload_start < payload_end and payload_end <= len(data):
                        result.extend(data[payload_start:payload_end])
                    
                    pos = payload_end
                    continue
            
            # No valid header found, move forward
            pos += 1
        
        return bytes(result)
    
    def _skip_pes_header(self, data: bytes) -> int:
        """Skip the PES packet header and return offset to payload. (Legacy method)"""
        if len(data) < 9:
            return len(data)
        
        # data[0:3] = start code (00 00 01)
        # data[3] = stream id
        # data[4:6] = PES packet length
        # data[6] = flags
        # data[7] = more flags  
        # data[8] = header data length
        
        if len(data) > 8:
            header_len = data[8]
            return 9 + header_len
        return len(data)
    
    def _parse_control_sequence(self, data: bytes, offset: int) -> SubPictureInfo:
        """
        Parse control sequence to extract image dimensions, colors, alpha, and RLE offsets.
        
        Control commands:
            0x00 - Force display
            0x01 - Start display
            0x02 - Stop display
            0x03 - Set colors (4 nibbles = palette indices)
            0x04 - Set contrast/alpha (4 nibbles)
            0x05 - Set display area (6 bytes)
            0x06 - Set RLE data offsets (4 bytes: top field, bottom field)
            0xFF - End of control sequence
        """
        info = SubPictureInfo()
        
        pos = offset
        while pos < len(data):
            if pos >= len(data):
                break
                
            cmd = data[pos]
            pos += 1
            
            if cmd == 0x00:
                # Force display
                info.forced = True
                continue
            elif cmd == 0x01:
                # Start display
                continue
            elif cmd == 0x02:
                # Stop display
                continue
            elif cmd == 0x03:
                # Set colors (4 nibbles = 4 palette indices)
                # Order: color3, color2, color1, color0
                if pos + 2 <= len(data):
                    info.colors_used = [
                        (data[pos] >> 4) & 0x0F,
                        data[pos] & 0x0F,
                        (data[pos + 1] >> 4) & 0x0F,
                        data[pos + 1] & 0x0F,
                    ]
                    pos += 2
            elif cmd == 0x04:
                # Set contrast/alpha (4 nibbles)
                # Values: 0x0 = transparent, 0xF = opaque
                # Convert to 0-255 by multiplying by 17
                if pos + 2 <= len(data):
                    info.alpha_values = [
                        ((data[pos] >> 4) & 0x0F) * 17,
                        (data[pos] & 0x0F) * 17,
                        ((data[pos + 1] >> 4) & 0x0F) * 17,
                        (data[pos + 1] & 0x0F) * 17,
                    ]
                    pos += 2
            elif cmd == 0x05:
                # Set display area (coordinates)
                if pos + 6 <= len(data):
                    x1 = (data[pos] << 4) | ((data[pos + 1] >> 4) & 0x0F)
                    x2 = ((data[pos + 1] & 0x0F) << 8) | data[pos + 2]
                    y1 = (data[pos + 3] << 4) | ((data[pos + 4] >> 4) & 0x0F)
                    y2 = ((data[pos + 4] & 0x0F) << 8) | data[pos + 5]
                    info.width = x2 - x1 + 1
                    info.height = y2 - y1 + 1
                    pos += 6
            elif cmd == 0x06:
                # Set RLE data offsets (2 bytes each for top/bottom field)
                if pos + 4 <= len(data):
                    info.top_field_offset = (data[pos] << 8) | data[pos + 1]
                    info.bottom_field_offset = (data[pos + 2] << 8) | data[pos + 3]
                    pos += 4
            elif cmd == 0xFF:
                # End of control sequence
                break
            else:
                # Unknown command, try to continue
                continue
        
        return info
    
    def _decode_rle_bitmap_interlaced(
        self,
        data: bytes,
        base_offset: int,
        sub_info: SubPictureInfo
    ) -> Optional[Image.Image]:
        """
        Decode interlaced RLE bitmap to a PIL RGBA image.
        
        DVD subtitles are interlaced:
        - Top field contains even lines (0, 2, 4, ...)
        - Bottom field contains odd lines (1, 3, 5, ...)
        
        Each field is RLE-encoded separately with its own offset.
        """
        width = sub_info.width
        height = sub_info.height
        
        if width <= 0 or height <= 0:
            return None
        
        # Create RGBA image
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        pixels = image.load()
        
        # Decode even lines from top field
        top_addr = base_offset + sub_info.top_field_offset
        self._decode_field(
            data=data,
            start_addr=top_addr,
            pixels=pixels,
            width=width,
            height=height,
            start_y=0,
            y_step=2,
            colors_used=sub_info.colors_used,
            alpha_values=sub_info.alpha_values
        )
        
        # Decode odd lines from bottom field
        bottom_addr = base_offset + sub_info.bottom_field_offset
        self._decode_field(
            data=data,
            start_addr=bottom_addr,
            pixels=pixels,
            width=width,
            height=height,
            start_y=1,
            y_step=2,
            colors_used=sub_info.colors_used,
            alpha_values=sub_info.alpha_values
        )
        
        return image
    
    def _decode_field(
        self,
        data: bytes,
        start_addr: int,
        pixels,
        width: int,
        height: int,
        start_y: int,
        y_step: int,
        colors_used: List[int],
        alpha_values: List[int]
    ) -> None:
        """
        Decode one field (top or bottom) of an interlaced subtitle.
        
        RLE encoding (from SubtitleEdit's DecodeRle):
        - 4-bit:  nncc (n=1-3, c=color)
        - 8-bit:  00nnnncc (n=4-15)
        - 12-bit: 0000nnnnnncc (n=16-63)
        - 16-bit: 000000nnnnnnnncc (n=64-255)
        - All zeros: end of line (align to byte)
        """
        pos = start_addr
        only_half = False  # Tracks nibble alignment
        y = start_y
        x = 0
        
        while y < height and pos + 2 < len(data):
            color, run_len, bytes_consumed, only_half, rest_of_line = self._decode_rle(
                data, pos, only_half
            )
            pos += bytes_consumed
            
            if rest_of_line:
                run_len = width - x
            
            if run_len == 0 and not rest_of_line:
                # End of line marker
                if only_half:
                    only_half = False
                    pos += 1
                x = 0
                y += y_step
                continue
            
            # Get color from palette with alpha
            rgba = self._get_color_with_alpha(color, colors_used, alpha_values)
            
            # Draw the run
            for _ in range(run_len):
                if x >= width:
                    # Handle line overflow
                    if only_half:
                        only_half = False
                        pos += 1
                    x = 0
                    y += y_step
                    if y >= height:
                        return
                
                if y < height:
                    pixels[x, y] = rgba
                x += 1
    
    def _decode_rle(
        self,
        data: bytes,
        index: int,
        only_half: bool
    ) -> Tuple[int, int, int, bool, bool]:
        """
        Decode one RLE code from the data stream.
        
        Based on SubtitleEdit's DecodeRle function.
        
        Returns:
            (color, run_length, bytes_consumed, new_only_half, rest_of_line)
        """
        rest_of_line = False
        
        if index + 2 > len(data):
            return 0, 0, 0, only_half, True
        
        b1 = data[index]
        b2 = data[index + 1] if index + 1 < len(data) else 0
        
        # Handle half-nibble alignment
        if only_half:
            b3 = data[index + 2] if index + 2 < len(data) else 0
            b1 = ((b1 & 0x0F) << 4) | ((b2 & 0xF0) >> 4)
            b2 = ((b2 & 0x0F) << 4) | ((b3 & 0xF0) >> 4)
        
        # Decode based on leading bits
        if (b1 >> 2) == 0:
            # 16-bit: 000000nnnnnnnncc
            run_len = (b1 << 6) | (b2 >> 2)
            color = b2 & 0x03
            if run_len == 0:
                # Rest of line + align
                rest_of_line = True
                if only_half:
                    return color, 0, 3, False, True
            return color, run_len, 2, only_half, rest_of_line
        
        if (b1 >> 4) == 0:
            # 12-bit: 0000nnnnnncc
            run_len = (b1 << 2) | (b2 >> 6)
            color = (b2 & 0x30) >> 4
            if only_half:
                return color, run_len, 2, False, rest_of_line
            return color, run_len, 1, True, rest_of_line
        
        if (b1 >> 6) == 0:
            # 8-bit: 00nnnncc
            run_len = b1 >> 2
            color = b1 & 0x03
            return color, run_len, 1, only_half, rest_of_line
        
        # 4-bit: nncc (n=1-3)
        run_len = b1 >> 6
        color = (b1 & 0x30) >> 4
        
        if only_half:
            return color, run_len, 1, False, rest_of_line
        return color, run_len, 0, True, rest_of_line
    
    def _get_color_with_alpha(
        self,
        color_idx: int,
        colors_used: List[int],
        alpha_values: List[int]
    ) -> Tuple[int, int, int, int]:
        """Get RGBA color from palette with proper alpha."""
        if color_idx < len(colors_used):
            palette_idx = colors_used[color_idx]
            if palette_idx < len(self.palette.colors):
                base_color = self.palette.colors[palette_idx]
                alpha = alpha_values[color_idx] if color_idx < len(alpha_values) else 255
                return (base_color[0], base_color[1], base_color[2], alpha)
        
        return (255, 255, 255, 255)
