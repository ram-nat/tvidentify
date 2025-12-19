import argparse
import json
import os
import re

from google.genai import Client
from openai import OpenAI
from .subtitle_extractor import extract_subtitles, check_required_tools


# Configure the API keys
# IMPORTANT: You must set the appropriate API key environment variables.
# For Google Gemini: export GOOGLE_API_KEY="YOUR_API_KEY"
# For OpenAI: export OPENAI_API_KEY="YOUR_API_KEY"
# For Perplexity: export PERPLEXITY_API_KEY="YOUR_API_KEY"


def identify_episode(series_name, subtitles, model="gemini-2.5-flash", provider="google"):
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

    # Join the subtitles into a single block of text for the prompt
    subtitle_text = "\n".join(subtitles)

    prompt = f"""
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


def _parse_json_response(response_text):
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


def _identify_episode_google(prompt, model):
    """
    Identify episode using Google Gemini API.
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "GOOGLE_API_KEY environment variable not set"}
        
        with Client(api_key=api_key) as  gemini_client:
            print(f"Asking {model} to identify the episode...")
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


def _identify_episode_openai(prompt, model):
    """
    Identify episode using OpenAI API.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY environment variable not set"}
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        print(f"Asking {model} to identify the episode...")
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


def _identify_episode_perplexity(prompt, model):
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
        
        print(f"Asking {model} to identify the episode...")
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


def main():
    parser = argparse.ArgumentParser(description='Identify the season and episode of a TV show from a video file or provided subtitles.')
    parser.add_argument('input_file', nargs='?', help='The input video file (optional if --subtitles-json is provided).')
    parser.add_argument('--series-name', required=True, help='The name of the TV series.')
    parser.add_argument('--subtitles-json', type=str, default=None,
                        help='Path to a JSON file containing subtitle strings (array of strings). If provided, skips subtitle extraction.')
    parser.add_argument('--provider', type=str, default='google', choices=['google', 'openai', 'perplexity'],
                        help='LLM provider to use (default: google).')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name. If not provided, defaults based on provider (google: gemini-2.5-flash, openai: gpt-4, perplexity: sonar).')
    # Arguments for subtitle extraction
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Maximum number of subtitle events to process (default: 10). This controls how much text is sent to the LLM.')
    parser.add_argument('--subtitle-track', type=int, default=0, help='The subtitle track index to use.')
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip the first N minutes of the video for subtitle extraction.')
    parser.add_argument('--scan-duration', type=int, default=15,
                        help='How many minutes of the video to scan for subtitles from the offset (default: 15).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Optional directory to save JSON output instead of printing to console.')

    args = parser.parse_args()

    # Check for required tools once at startup
    print("Checking for required tools...")
    if not check_required_tools():
        print("\nError: Not all required tools are available. Please install the missing tools and try again.")
        return

    # Set default model based on provider
    if args.model is None:
        if args.provider == 'google':
            args.model = 'gemini-2.5-flash'
        elif args.provider == 'openai':
            args.model = 'gpt-4'
        elif args.provider == 'perplexity':
            args.model = 'sonar'

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
                    print("Error: 'subtitles' key must contain an array of strings.")
                    return
            # 3. Empty or invalid object
            else:
                print("Error: JSON file must be either:")
                print("  - An array of subtitle strings")
                print("  - An object with a 'subtitles' key containing an array of strings")
                return
            
            print(f"Loaded {len(subtitles)} subtitles from {args.subtitles_json}")
        except FileNotFoundError:
            print(f"Error: Subtitles JSON file not found: {args.subtitles_json}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in subtitles file: {e}")
            return
    else:
        # Extract subtitles from video file
        if not args.input_file:
            print("Error: Either input_file or --subtitles-json must be provided.")
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
                print(f"\nJSON output saved to: {output_file}")
            except IOError as e:
                print(f"Error saving JSON output: {e}")
        else:
            # Print to console if no output_dir specified
            print("\n--- LLM Identification Result ---")
            print(json.dumps(result, indent=2))
    else:
        print("Could not extract any subtitles to send to the LLM.")


if __name__ == '__main__':
    # Check for required API key based on provider
    main()
