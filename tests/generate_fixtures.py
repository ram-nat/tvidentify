"""
Generate synthetic test fixtures for integration tests.

This script creates:
1. A subtitle-style PNG image for OCR testing
2. A minimal SUP (PGS) file for subtitle extraction testing

Run this script to regenerate fixtures if needed.
"""

import os
import struct
from pathlib import Path

# Ensure PIL is available (it comes with opencv-python)
from PIL import Image, ImageDraw, ImageFont


def create_subtitle_test_image():
    """Create a subtitle-style image (white text on transparent background)."""
    width, height = 600, 60
    # Transparent background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Try to get a readable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # White text (typical subtitle style)
    text = "I am the one who knocks"
    draw.text((20, 15), text, fill=(255, 255, 255, 255), font=font)
    
    return img, text


def create_sup_segment(segment_type, pts, data):
    """
    Create a PGS segment with the given type.
    
    Format:
    - 2 bytes: "PG" magic
    - 4 bytes: PTS (presentation timestamp)
    - 4 bytes: DTS (decoding timestamp, usually 0)
    - 1 byte: segment type
    - 2 bytes: segment size
    - N bytes: segment data
    """
    magic = b'PG'
    pts_bytes = struct.pack('>I', pts)  # Big-endian 4 bytes
    dts_bytes = struct.pack('>I', 0)
    type_byte = struct.pack('B', segment_type)
    size_bytes = struct.pack('>H', len(data))
    
    return magic + pts_bytes + dts_bytes + type_byte + size_bytes + data


def create_minimal_sup_file(width=200, height=40):
    """
    Create a minimal valid SUP file with a simple white rectangle subtitle.
    
    PGS format requires:
    - PCS (Presentation Composition Segment) - 0x16
    - WDS (Window Definition Segment) - 0x17
    - PDS (Palette Definition Segment) - 0x14
    - ODS (Object Definition Segment) - 0x15
    - END (End of Display Set) - 0x80
    """
    segments = []
    pts = 90000  # 1 second in 90kHz clock
    
    # 1. PCS - Presentation Composition Segment
    pcs_data = bytearray()
    pcs_data += struct.pack('>H', 1920)  # Video width
    pcs_data += struct.pack('>H', 1080)  # Video height
    pcs_data += struct.pack('B', 0x10)   # Frame rate
    pcs_data += struct.pack('>H', 1)     # Composition number
    pcs_data += struct.pack('B', 0x80)   # Composition state: Epoch Start
    pcs_data += struct.pack('B', 0)      # Palette update flag
    pcs_data += struct.pack('B', 0)      # Palette ID
    pcs_data += struct.pack('B', 1)      # Number of composition objects
    # Composition object
    pcs_data += struct.pack('>H', 0)     # Object ID
    pcs_data += struct.pack('B', 0)      # Window ID
    pcs_data += struct.pack('B', 0)      # Object cropped flag
    pcs_data += struct.pack('>H', 100)   # X position
    pcs_data += struct.pack('>H', 500)   # Y position
    segments.append(create_sup_segment(0x16, pts, bytes(pcs_data)))
    
    # 2. WDS - Window Definition Segment
    wds_data = bytearray()
    wds_data += struct.pack('B', 1)      # Number of windows
    wds_data += struct.pack('B', 0)      # Window ID
    wds_data += struct.pack('>H', 100)   # X position
    wds_data += struct.pack('>H', 500)   # Y position
    wds_data += struct.pack('>H', width) # Width
    wds_data += struct.pack('>H', height) # Height
    segments.append(create_sup_segment(0x17, pts, bytes(wds_data)))
    
    # 3. PDS - Palette Definition Segment
    pds_data = bytearray()
    pds_data += struct.pack('B', 0)      # Palette ID
    pds_data += struct.pack('B', 0)      # Palette version
    # Palette entry 0: transparent (Y=0, Cr=128, Cb=128, Alpha=0)
    pds_data += struct.pack('BBBBB', 0, 16, 128, 128, 0)
    # Palette entry 1: white (Y=235, Cr=128, Cb=128, Alpha=255)
    pds_data += struct.pack('BBBBB', 1, 235, 128, 128, 255)
    segments.append(create_sup_segment(0x14, pts, bytes(pds_data)))
    
    # 4. ODS - Object Definition Segment with RLE-encoded image
    # Create a simple pattern: white rectangle that tesseract can read as text
    # We'll draw "TEST" using a simple pixel pattern
    ods_data = bytearray()
    ods_data += struct.pack('>H', 0)     # Object ID
    ods_data += struct.pack('B', 0)      # Object version
    ods_data += struct.pack('B', 0xC0)   # Sequence flag: First and Last
    
    # RLE data placeholder - we'll build the image
    rle_data = create_rle_text_image(width, height)
    
    ods_data += struct.pack('>I', len(rle_data) + 4)[1:]  # Data length (3 bytes)
    ods_data += struct.pack('>H', width)   # Width
    ods_data += struct.pack('>H', height)  # Height
    ods_data += rle_data
    segments.append(create_sup_segment(0x15, pts, bytes(ods_data)))
    
    # 5. END - End of Display Set
    segments.append(create_sup_segment(0x80, pts, b''))
    
    return b''.join(segments)


def create_rle_text_image(width, height):
    """
    Create RLE-encoded image data with text "TEST" that tesseract can read.
    
    PGS RLE format:
    - Non-zero byte: single pixel of that color
    - 0x00 0x00: End of line
    - 0x00 0x01-0x3F: Run of N transparent pixels
    - 0x00 0x40-0x7F: Run of ((N-0x40)<<8 + next_byte) transparent pixels
    - 0x00 0x80-0xBF: Run of (N-0x80) pixels of next color
    - 0x00 0xC0-0xFF: Run of ((N-0xC0)<<8 + next_byte) pixels of color after that
    """
    # Create PIL image with text
    img = Image.new('L', (width, height), 0)  # Black/transparent background
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    draw.text((10, 8), "TEST", fill=1, font=font)  # Color index 1 = white
    
    # Convert to RLE
    rle = bytearray()
    pixels = list(img.getdata())
    
    for y in range(height):
        row = pixels[y * width:(y + 1) * width]
        x = 0
        while x < width:
            color = row[x]
            # Count run length
            run_length = 1
            while x + run_length < width and row[x + run_length] == color and run_length < 255:
                run_length += 1
            
            if color == 0:  # Transparent
                if run_length < 64:
                    rle += bytes([0x00, run_length])
                else:
                    rle += bytes([0x00, 0x40 + (run_length >> 8), run_length & 0xFF])
            else:  # Color pixel
                if run_length == 1:
                    rle += bytes([color])
                elif run_length < 64:
                    rle += bytes([0x00, 0x80 + run_length, color])
                else:
                    rle += bytes([0x00, 0xC0 + (run_length >> 8), run_length & 0xFF, color])
            
            x += run_length
        
        # End of line
        rle += bytes([0x00, 0x00])
    
    return bytes(rle)



def main():
    """Generate all test fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    # 1. Create subtitle-style image for OCR test
    img, _ = create_subtitle_test_image()
    img.save(fixtures_dir / "subtitle_test.png")
    print(f"Created: {fixtures_dir / 'subtitle_test.png'}")
    
    # 2. Create SUP file for subtitle extraction test
    sup_data = create_minimal_sup_file()
    sup_path = fixtures_dir / "test_subtitle.sup"
    with open(sup_path, 'wb') as f:
        f.write(sup_data)
    print(f"Created: {sup_path} ({len(sup_data)} bytes)")
    
    print("\nAll fixtures created successfully!")


if __name__ == "__main__":
    main()
