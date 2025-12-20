# TVIdentify

Python tool for automatically identifying and renaming TV Show episodes for Plex given the video files and series name. Identifies TV show episodes from video files using OCR on PGS subtitles and LLM analysis of subtitles.

## Get Started

### To just install the package and use it as a utility
```bash
python3 -m venv tvidentify
cd tvidentify
source bin/activate
pip install tvidentify
```

### To modify the sources or build/work from source
```bash
git clone https://github.com/ram-nat/tvidentify tvidentify
cd tvidentify
python3 -m venv venv
source venv/bin/activate
pip install -e .
tvidentify /path/to/TVShows/Game\ Of\ Thrones/Season\ 02/ --max-frames 10 --offset 3 --series-name "Game Of Thrones" --scan-duration 5 --output-dir ~/gots2 --model gemini-3-pro-preview --rename --skip-already-named
```

### Usage
```bash
usage: tvidentify [-h] --series-name SERIES_NAME [--size-threshold SIZE_THRESHOLD] [--provider {google,openai,perplexity}] [--model MODEL] [--max-frames MAX_FRAMES] [--subtitle-track SUBTITLE_TRACK] [--offset OFFSET] [--scan-duration SCAN_DURATION] [--output-dir OUTPUT_DIR]
                  [--rename] [--rename-format RENAME_FORMAT] [--skip-already-named]
                  input_dir

Batch identify TV show episodes in a directory.

positional arguments:
  input_dir             The directory containing video files.

options:
  -h, --help            show this help message and exit
  --series-name SERIES_NAME
                        The name of the TV series.
  --size-threshold SIZE_THRESHOLD
                        Size similarity threshold for filtering episodes (default: 0.7).
  --provider {google,openai,perplexity}
                        LLM provider to use (default: google).
  --model MODEL         Model name. If not provided, defaults based on provider.
  --max-frames MAX_FRAMES
                        Maximum number of subtitle events to process (default: 10).
  --subtitle-track SUBTITLE_TRACK
                        The subtitle track index to use (default: 0).
  --offset OFFSET       Skip the first N minutes for subtitle extraction (default: 0).
  --scan-duration SCAN_DURATION
                        How many minutes to scan for subtitles from the offset (default: 15).
  --output-dir OUTPUT_DIR
                        Optional directory to save JSON output files (one per video) instead of printing to console.
  --rename              Rename files to "<series_name> S<season>E<episode>" format if identification is successful.
  --rename-format RENAME_FORMAT
                        Format for renamed files. Available placeholders: {{series}}, {{season}}, {{episode}}. Default: "{{series}} S{{season:02d}}E{{episode:02d}}"
  --skip-already-named  Skip files that are already in the expected naming format (only when --rename is specified).
```

## Features

- **Subtitle Extraction**: 
  - `subtitle_extractor.py` is the stand-alone module for this.
  - Extracts English subtitle stream (expects and handles PGS only)
  - You can specify starting offset and duration of subtitle stream to extract (so entire file is not processed). You can also specify the maximum number of subtitle events to extract.
    - `--offset` to specify starting offset in minutes
    - `--scan-duration` to specify how many minutes from starting offset you want to extract.
    - `--max-frames` to specify how many subtitle events to extract.
  - Uses OCR (`pytesseract`) and some very basic regex clean-up to get subtitle text.
  - PGS parsing code is from https://github.com/EzraBC/pgsreader
  - Use `--output-dir` to store output in json format.
- **Episode Identification**
  - episode_identifier.py is the stand-alone module for this.
  - With extracted subtitles and the series name, use LLMs to identify the episode of the series.
  - Supports different LLM providers - Google Gemini, OpenAI or Perplexity
  - Use `--model` to pass the model to use for episode identification
  - You can pass an MKV (or other container format) file or the json output from subtitle_extractor.py as input to this stage.
  - Use `--output-dir` to store output in json format.
- **Batch Identification**
  - `batch_identifier.py` is the stand-alone module for this.
  - Pass an entire season folder to identify all episodes in folder.
  - Identifies and ignores non-episode files (assumes largest files are episodes)
  - Identifies and does not process duplicate episode files (uses subtitle similarity for duplicates)
  - Use `--rename` option to rename identified episodes to match Plex episode naming requirements.
  - Use `--output-dir` to store output in json format. Stores both batch results and results for individual files.
- **File Renaming**
  - `file_renamer.py` is the stand-alone module for this.
  - Use `--rename-format` to specify the rename format. Series, season and episode are the available variables for the format string.

## Installation

1. Clone the repository
2. Set up the Python virtual environment (already configured)
3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

Set the appropriate API key environment variables:

```bash
# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Perplexity
export PERPLEXITY_API_KEY="your-perplexity-api-key"
```

## Usage

### Extract Subtitles from Video

```bash
python subtitle_extractor.py /path/to/video.mkv \
  --output-dir ./subtitles
```

### Identify Episode from Video File

```bash
python episode_identifier.py /path/to/video.mkv \
  --series-name "Game of Thrones" \
  --provider google
```

### Identify Episode from Pre-extracted Subtitles

```bash
python episode_identifier.py \
  --series-name "Game of Thrones" \
  --subtitles-json subtitles.json \
  --provider openai
```

### Batch Processing an Entire Season

```bash
python batch_identifier.py /path/to/episodes/directory \
  --series-name "Game of Thrones" \
  --provider google \
  --rename
```

### Command-line Options

#### subtitle_extractor.py
- `input_file`: Path to the video file to extract subtitles from
- `--max-frames`: Maximum number of subtitle events to extract
- `--subtitle-track`: Subtitle track index to use (default: 0)
- `--offset`: Skip first N minutes (default: 0)
- `--scan-duration`: Minutes to scan from offset (default: 15)
- `--output-dir`: Directory to save JSON output file

#### episode_identifier.py
- `input_file` (optional): Path to video file (required if `--subtitles-json` not provided)
- `--series-name` (required): Name of the TV series
- `--provider`: LLM provider (default: google). Options: google, openai, perplexity
- `--model`: Model name. Defaults: gemini-2.5-flash (google), gpt-4 (openai), sonar (perplexity)
- `--subtitles-json`: Path to JSON file with pre-extracted subtitles (alternative to video input)
- `--max-frames`: Maximum number of subtitle events to process (default: 10)
- `--subtitle-track`: Subtitle track index to use (default: 0)
- `--offset`: Skip first N minutes (default: 0)
- `--scan-duration`: Minutes to scan from offset (default: 15)
- `--output-dir`: Directory to save JSON output file

#### batch_identifier.py
- `input_dir`: Directory containing video files to process
- `--series-name` (required): Name of the TV series
- `--size-threshold`: File size similarity threshold for filtering episodes (default: 0.7)
- `--provider`: LLM provider (default: google). Options: google, openai, perplexity
- `--model`: Model name. Defaults: gemini-2.5-flash (google), gpt-4 (openai), sonar-pro (perplexity)
- `--max-frames`: Maximum number of subtitle events to process (default: 10)
- `--subtitle-track`: Subtitle track index to use (default: 0)
- `--offset`: Skip first N minutes (default: 0)
- `--scan-duration`: Minutes to scan from offset (default: 15)
- `--output-dir`: Directory to save JSON output files
- `--rename`: Rename identified episodes to match Plex naming format
- `--rename-format`: Format string for renamed files (default: `{series} S{season:02d}E{episode:02d}`)
- `--skip-already-named`: Skip files that are already in the expected naming format (only when `--rename` is specified)

#### file_renamer.py
- `--batch-results` (required): Path to batch_results.json from batch_identifier
- `--series-name` (required): Name of the TV series
- `--rename-format`: Format string for renamed files. Available placeholders: `{series}`, `{season}`, `{episode}` (default: `{series} S{season:02d}E{episode:02d}`)
- `--dry-run`: Show what would be renamed without actually renaming

## Components

### subtitle_extractor.py
Handles extraction of subtitles from video files:
- Detects subtitle tracks using ffprobe
- Extracts frames for each subtitle event
- Performs OCR on frames using Tesseract
- Filters gibberish using character pattern analysis

### episode_identifier.py
Identifies TV show episodes from subtitles:
- Loads subtitles from video or JSON file
- Sends subtitles to LLM with identifying prompt
- Parses LLM response for season/episode information
- Supports multiple LLM providers

### batch_identifier.py
Processes multiple video files:
- Discovers episode files by size similarity
- Processes each file with episode_identifier
- Outputs results in JSON format

## Requirements

### System Dependencies
- `ffmpeg` - For video processing
- `ffprobe` - For reading video metadata (comes with ffmpeg)
- `tesseract-ocr` - For optical character recognition (OCR) on subtitle images

Install on Ubuntu/Debian:
```bash
sudo apt-get install ffmpeg tesseract-ocr
```

Install on macOS:
```bash
brew install ffmpeg tesseract
```

### Python Dependencies
See requirements.txt - includes:
- `opencv-python-headless` - For video frame processing
- `pytesseract` - Python interface to Tesseract OCR
- `openai` - OpenAI API client
- `google-genai` - Google Generative AI client

## Example Output

```json
{
  "season": 1,
  "episode": 2,
  "subtitles": [
    "Sorry, Your Grace.",
    "My deepest apologies.",
    "No. No, Your Grace."
  ],
  "provider": "google",
  "model": "gemini-2.5-flash"
}
```

## Notes

- PGS subtitles are image-based, so OCR quality depends on video resolution and subtitle clarity
- In my tests, `gemini-3-pro-preview` has been the best model at identifying episodes consistently and correctly.
- About 5 minutes of subtitle input has been sufficient to identify GOT episodes in my testing.

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
