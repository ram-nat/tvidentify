
import logging
import sys
import os
import subprocess
import argparse
from typing import Optional, Dict

# --- Constants ---

DEFAULT_MODELS = {
    "google": "gemini-2.5-flash",
    "openai": "gpt-4",
    "perplexity": "sonar"
}

# The single source of truth for the LLM prompt
EPISODE_IDENTIFICATION_PROMPT = """
You are an expert TV series database assistant. Your task is to identify a TV episode strictly based on the provided subtitle snippet and series name.

Instructions:
1. Analyze the subtitle text below.
2. Identify specific character names, unique plot points, or dialogue lines.
3. Match these details to your internal knowledge of the series "{series_name}".
4. DO NOT perform a web search unless absolutely necessary. Rely on your training data.
5. If the text is generic (e.g., "Hello", "How are you"), return null.
6. You must provide a confidence score (0-100) indicating how certain you are about the match.
7. Provide a brief reasoning for your identification based on the subtitle content.

Subtitles:
---
{subtitle_text}
---

Output Format:
Return ONLY a raw JSON object with the format below. Do not output markdown code blocks:
{{
  "season": <int or null>,
  "episode": <int or null>,
  "confidence_score": <0-100>,
  "reasoning": "<brief explanation of which line confirmed the match>"
}}
"""


def setup_logging(console_level: int = logging.INFO, log_file: Optional[str] = None, file_level: int = logging.INFO) -> None:
    """
    Configures the root logger with dual handlers:
    1. Console Handler: Human-friendly format (message only for INFO, prefixed for errors).
    2. File Handler (Optional): Detailed format with timestamps for machine debugging.
    
    Args:
        console_level: Logging level for the console output (default: INFO)
        log_file: Path to a log file. If provided, logs are written here.
        file_level: Logging level for the log file (default: INFO).
    """
    root_log = logging.getLogger()
    root_log.setLevel(logging.DEBUG)  # Capture everything at the root level

    # Clear existing handlers to prevent duplicate logs on re-import/re-run
    # Do not iterate over the list while modifying it
    for handler in list(root_log.handlers):
        root_log.removeHandler(handler)

    # --- 1. Console Handler (Human Friendly) ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)

    class HumanFormatter(logging.Formatter):
        def format(self, record):
            # Get the message with arguments applied (e.g. "Found %d files" -> "Found 5 files")
            message = record.getMessage()
            
            # For critical errors, make them pop.
            if record.levelno >= logging.ERROR:
                return f"❌ Error: {message}"
            # For warnings, add a small prefix
            elif record.levelno >= logging.WARNING:
                return f"⚠️  Warning: {message}"
            # For INFO and below, just print the message cleanly (like print())
            return message

    console_handler.setFormatter(HumanFormatter())
    root_log.addHandler(console_handler)

    # --- 2. File Handler (Machine Friendly / Audit Trail) ---
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(file_level)  # Capture logs at correct level on disk
            
            # Standard detailed log format: Timestamp [Level] LoggerName (Func:Line): Message
            detailed_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s (%(funcName)s:%(lineno)d): %(message)s'
            )
            file_handler.setFormatter(detailed_formatter)
            root_log.addHandler(file_handler)
        except Exception as e:
            # Fallback if we can't write to the log file (e.g. permissions)
            sys.stderr.write(f"⚠️  Warning: Could not open log file '{log_file}': {e}\n")

def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds standard logging arguments to the provided argparse parser.
    
    Args:
        parser: The argparse parser to add arguments to.
    """
    group = parser.add_argument_group('Logging')
    group.add_argument('--log-file', type=str, default=None,
                        help='Path to a file to write detailed debug logs to.')
    group.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output (INFO level is default, this is largely for symmetry).')
    group.add_argument('--debug', action='store_true',
                        help='Enable debug output to console.')

def check_api_key(provider: str) -> bool:
    """
    Checks if the required environment variable for the provider is set.
    Logs an error if missing.
    
    Args:
        provider: 'google', 'openai', or 'perplexity'
        
    Returns:
        bool: True if key exists, False otherwise.
    """
    logger = logging.getLogger(__name__)
    key_map = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY"
    }
    
    env_var = key_map.get(provider.lower())
    if not env_var:
        logger.error("Unknown provider: %s", provider)
        return False
        
    if not os.environ.get(env_var):
        logger.error("Missing API key for %s. Please set %s.", provider, env_var)
        return False
        
    return True

def check_required_tools() -> bool:
    """
    Check if required tools are installed: ffmpeg, ffprobe, and tesseract.
    
    Returns:
        bool: True if all tools are available, False otherwise
    """
    logger = logging.getLogger(__name__)
    tools = [
        ('ffmpeg', 'ffmpeg', '-version'),
        ('ffprobe', 'ffprobe', '-version'),
        ('tesseract', 'Tesseract OCR', '--version')
    ]
    
    all_available = True
    for tool_cmd, tool_name, arg in tools:
        try:
            subprocess.run([tool_cmd, arg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug("%s is available", tool_name)
        except FileNotFoundError:
            logger.error("%s is not installed or not in your PATH. Please install it.", tool_name)
            all_available = False
    
    return all_available
