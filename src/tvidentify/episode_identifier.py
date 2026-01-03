import argparse
import json
import os
import re
import logging
from typing import List, Dict, Optional, Any, Union

from google.genai import Client
from openai import OpenAI
from .utils import (
    check_required_tools, 
    setup_logging, 
    check_api_key, 
    DEFAULT_MODELS, 
    EPISODE_IDENTIFICATION_PROMPT,
    add_logging_args
)
from .subtitle_extractor import add_extraction_args, extract_subtitles

logger = logging.getLogger(__name__)


def identify_episode(series_name: str, subtitles: List[str], model: str = "gemini-2.5-flash", provider: str = "google") -> Dict[str, Any]:
    """
    Uses an LLM to identify the season and episode number from a list of subtitles.

    Args:
        series_name (str): The name of the TV series.
        subtitles (list[str]): A list of subtitle strings.
        model (str): The model to use (e.g., "gemini-2.5-flash", "gpt-4", "sonar").
        provider (str): The LLM provider - "google", "openai", or "perplexity".

    Returns:
        dict: A dictionary containing the season and episode, or an error message.
    """
    if not subtitles:
        return {"error": "Could not identify episode: No subtitles provided."}
    
    # Check for API key before proceeding
    if not check_api_key(provider):
        return {"error": f"Missing API key for provider: {provider}"}

    # Join the subtitles into a single block of text for the prompt
    subtitle_text = "\n".join(subtitles)

    # Use the centralized prompt template
    prompt = EPISODE_IDENTIFICATION_PROMPT.format(
        series_name=series_name,
        subtitle_text=subtitle_text
    )
    try:
        if provider.lower() == "google":
            return _identify_episode_google(prompt, model)
        elif provider.lower() == "openai":
            return _identify_episode_openai(prompt, model)
        elif provider.lower() == "perplexity":
            return _identify_episode_perplexity(prompt, model)
        else:
            return {"error": f"Unknown provider: {provider}. Supported providers: google, openai, perplexity"}
    except Exception as e:
        return {
            "season": None,
            "episode": None,
            "error": f"An error occurred: {e}"
        }


def _parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Helper function to parse JSON from LLM response text.
    Handles responses wrapped in markdown code blocks or plain JSON.
    """
    # Try markdown code block first
    json_match = re.search(r"```json\n({.*?})\n```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fallback for plain JSON output
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return None
        json_str = json_match.group()
    
    return json.loads(json_str)


def _identify_episode_google(prompt: str, model: str) -> Dict[str, Any]:
    """
    Identify episode using Google Gemini API.
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        # Note: redundant check as check_api_key handles this, but kept for type safety locally
        if not api_key:
            return {"error": "GOOGLE_API_KEY environment variable not set"}
        
        with Client(api_key=api_key) as  gemini_client:
            logger.info("Asking %s to identify the episode...", model)
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )        
            parsed = _parse_json_response(response.text)
            if parsed:
                return parsed
            else:
                return {
                    "season": None,
                    "episode": None,
                    "error": "LLM did not return a valid JSON object.",
                    "llm_response": response.text,
                }
    except json.JSONDecodeError as e:
        return {
            "season": None,
            "episode": None,
            "error": f"Failed to parse JSON from LLM response: {e}",
        }


def _identify_episode_openai(prompt: str, model: str) -> Dict[str, Any]:
    """
    Identify episode using OpenAI API.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY environment variable not set"}
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        logger.info("Asking %s to identify the episode...", model)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        response_text = response.choices[0].message.content
        parsed = _parse_json_response(response_text)
        if parsed:
            return parsed
        else:
            return {
                "season": None,
                "episode": None,
                "error": "LLM did not return a valid JSON object.",
                "llm_response": response_text,
            }
    except json.JSONDecodeError as e:
        return {
            "season": None,
            "episode": None,
            "error": f"Failed to parse JSON from LLM response: {e}",
        }


def _identify_episode_perplexity(prompt: str, model: str) -> Dict[str, Any]:
    """
    Identify episode using Perplexity API (OpenAI-compatible).
    """
    try:
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            return {"error": "PERPLEXITY_API_KEY environment variable not set"}
        
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
        logger.info("Asking %s to identify the episode...", model)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        response_text = response.choices[0].message.content
        parsed = _parse_json_response(response_text)
        if parsed:
            return parsed
        else:
            return {
                "season": None,
                "episode": None,
                "error": "LLM did not return a valid JSON object.",
                "llm_response": response_text,
            }
    except json.JSONDecodeError as e:
        return {
            "season": None,
            "episode": None,
            "error": f"Failed to parse JSON from LLM response: {e}",
        }


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds standard LLM related arguments to the provided argparse parser.
    """
    group = parser.add_argument_group('LLM Configuration')
    group.add_argument('--provider', type=str, default='google', choices=['google', 'openai', 'perplexity'],
                        help='LLM provider to use (default: google).')
    group.add_argument('--model', type=str, default=None,
                        help='Model name. If not provided, defaults based on provider (google: gemini-2.5-flash, openai: gpt-4, perplexity: sonar).')
    group.add_argument('--series-name', required=True, help='The name of the TV series.')

def main():
    parser = argparse.ArgumentParser(description='Identify the season and episode of a TV show from a video file or provided subtitles.')
    parser.add_argument('input_file', nargs='?', help='The input video file (optional if --subtitles-json is provided).')
    parser.add_argument('--subtitles-json', type=str, default=None,
                        help='Path to a JSON file containing subtitle strings (array of strings). If provided, skips subtitle extraction.')
    
    add_llm_args(parser)
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

    # Set default model based on provider
    if args.model is None:
        args.model = DEFAULT_MODELS.get(args.provider, "gemini-2.5-flash")

    # Determine where to get subtitles from
    subtitles = None
    
    if args.subtitles_json:
        # Load subtitles from JSON file
        try:
            with open(args.subtitles_json, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats:
            # 1. Array of strings (direct subtitle array)
            if isinstance(data, list):
                subtitles = data
            # 2. Object with 'subtitles' key (output from subtitle_extractor or user-provided)
            elif isinstance(data, dict) and 'subtitles' in data:
                subtitles = data['subtitles']
                if not isinstance(subtitles, list):
                    logger.error("Error: 'subtitles' key must contain an array of strings.")
                    return
            # 3. Empty or invalid object
            else:
                logger.error("Error: JSON file must be either:")
                logger.error("  - An array of subtitle strings")
                logger.error("  - An object with a 'subtitles' key containing an array of strings")
                return
            
            logger.info("Loaded %d subtitles from %s", len(subtitles), args.subtitles_json)
        except FileNotFoundError:
            logger.error("Error: Subtitles JSON file not found: %s", args.subtitles_json)
            return
        except json.JSONDecodeError as e:
            logger.error("Error: Invalid JSON in subtitles file: %s", e)
            return
    else:
        # Extract subtitles from video file
        if not args.input_file:
            logger.error("Error: Either input_file or --subtitles-json must be provided.")
            parser.print_help()
            return
        
        # Step 1: Extract subtitles from the video file
        subtitles = extract_subtitles(
            video_file=args.input_file,
            subtitle_track_index=args.subtitle_track,
            offset_minutes=args.offset,
            max_frames=args.max_frames,
            scan_duration_minutes=args.scan_duration
        )

    # Step 2: Identify the episode using the extracted subtitles
    if subtitles:
        # Check API key before making the call
        if not check_api_key(args.provider):
            return

        result = identify_episode(args.series_name, subtitles, model=args.model, provider=args.provider)
        result["subtitles"] = subtitles
        result["provider"] = args.provider
        result["model"] = args.model
        
        # Save to JSON if output_dir is specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Create a safe filename from the series name and input file
            if args.input_file:
                base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                output_file = os.path.join(args.output_dir, f"{base_name}_identification.json")
            else:
                # If using subtitles JSON, use that filename
                base_name = os.path.splitext(os.path.basename(args.subtitles_json))[0]
                output_file = os.path.join(args.output_dir, f"{base_name}_identification.json")
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info("JSON output saved to: %s", output_file)
            except IOError as e:
                logger.error("Error saving JSON output: %s", e)
        else:
            # Print to console if no output_dir specified
            logger.info("--- LLM Identification Result ---")
            print(json.dumps(result, indent=2)) # Keep print for JSON output pipeability
    else:
        logger.error("Could not extract any subtitles to send to the LLM.")


if __name__ == '__main__':
    # Check for required API key based on provider
    main()
