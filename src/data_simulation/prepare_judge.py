import os
import requests
import json
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# control the number of thread workers
NUM_THREADS = 10

# --- Configuration ---
RAW_DIR = 'data/synthetic_feedback/raw'
SYNTHESIS_DIR = 'data/synthetic_feedback/output'
JUDGE_DIR = 'data/synthetic_feedback/llm_judge'
LITELLM_ENDPOINT = os.getenv('LITELLM_BASE_URL') + '/v1/chat/completions' # Use the same env var as before
MODEL_NAME = 'xai/grok-3-mini-fast-beta' # Or choose another model suitable for judging
TEMPERATURE = 0 # Lower temperature might be better for judging consistency
MAX_TOKENS = 2000 # Adjust as needed for the JSON response

MAX_RETRIES = 3
INITIAL_BACKOFF = 1 # seconds

# Get API Key from environment variable
API_KEY = os.getenv('LITELLM_API_KEY')

# --- Prompt Template for Judging ---
# Asks for a JSON response for easier parsing
JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating the relevance of a synthesized customer feedback based on the original customer data.

Original Customer Data:
---
{original_content}
---

Synthesized Customer Feedback:
---
{synthesized_content}
---

Is the synthesized feedback relevant to the original customer data? Please respond ONLY with a JSON object containing two keys:
1. "relevant": boolean (true if relevant, false otherwise)
2. "reasoning": string (a brief explanation for your judgment)

Example JSON response:
{{
  "relevant": true,
  "reasoning": "The synthesized feedback directly addresses the points mentioned in the original data regarding login frequency and offer preference."
}}

Your JSON response:"""

# --- Helper Functions (reusing make_api_request) ---

def make_api_request(session, payload, headers):
    """Sends the request to the LiteLLM API with retry logic."""
    # This function is identical to the one in prepare_synthesis.py
    # Included here for completeness, assuming this script might be run independently.
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(LITELLM_ENDPOINT, headers=headers, json=payload, timeout=60) # 60 second timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Request timed out (attempt {attempt + 1}/{MAX_RETRIES}). Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                 if e.response.status_code == 429 or e.response.status_code >= 500:
                     print("Potential transient error detected.")
                 else:
                     print(f"Non-retryable client error: {e.response.status_code}")
                     return None
            else:
                 print("Network error without response.")

        if attempt < MAX_RETRIES - 1:
            backoff_time = INITIAL_BACKOFF * (2 ** attempt)
            print(f"Waiting {backoff_time}s before next retry...")
            time.sleep(backoff_time)
        else:
            print("Max retries reached. Request failed.")
            return None
    return None

def judge_file_pair(filename, session, headers):
    """Reads raw and synthesized files, sends for judging, saves the judgment."""
    raw_filepath = os.path.join(RAW_DIR, filename)
    synthesis_filepath = os.path.join(SYNTHESIS_DIR, filename)
    judge_filepath = os.path.join(JUDGE_DIR, filename)

    # Skip if output file already exists
    if os.path.exists(judge_filepath):
        return filename, "skipped"

    try:
        # Read original content
        with open(raw_filepath, 'r', encoding='utf-8') as f_raw:
            original_content = f_raw.read()

        # Read synthesized content
        with open(synthesis_filepath, 'r', encoding='utf-8') as f_synth:
            synthesized_content = f_synth.read()

        # Construct the prompt
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            original_content=original_content,
            synthesized_content=synthesized_content
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
             # Instructing some models to output JSON
            "response_format": { "type": "json_object" }
        }

        api_response = make_api_request(session, payload, headers)

        if api_response:
            try:
                # Extract the judgment content
                judgment_content = api_response['choices'][0]['message']['content']

                # Attempt to validate if it's JSON (optional but good practice)
                try:
                    json.loads(judgment_content) # Try parsing
                    is_json = True
                except json.JSONDecodeError:
                    print(f"Warning: Response for {filename} is not valid JSON: {judgment_content[:100]}...") # Log first 100 chars
                    is_json = False

                # Write the judgment (raw string from LLM) to the output file
                with open(judge_filepath, 'w', encoding='utf-8') as outfile:
                    outfile.write(judgment_content)
                return filename, "success"

            except (KeyError, IndexError, TypeError) as e:
                print(f"Error parsing API response structure for {filename}: {e}. Response: {api_response}")
                return filename, "parse_error"
        else:
            print(f"API request failed for {filename} after retries.")
            return filename, "api_failed"

    except FileNotFoundError as e:
        print(f"Error: Missing file for {filename}: {e}")
        return filename, "missing_file"
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}")
        return filename, "error"

# --- Main Execution ---
if __name__ == "__main__":
    if not API_KEY:
        print("Error: LITELLM_API_KEY environment variable not set.")
        exit(1)
    if not LITELLM_ENDPOINT:
        print("Error: LITELLM_BASE_URL environment variable not set.")
        exit(1)

    if not os.path.isdir(RAW_DIR):
        print(f"Error: Raw directory '{RAW_DIR}' not found.")
        exit(1)
    if not os.path.isdir(SYNTHESIS_DIR):
        print(f"Error: Synthesis directory '{SYNTHESIS_DIR}' not found.")
        exit(1)

    os.makedirs(JUDGE_DIR, exist_ok=True)
    print(f"Ensured judge directory '{JUDGE_DIR}' exists.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Get list of files to process (using synthesis dir as reference)
    try:
        files_to_judge = [f for f in os.listdir(SYNTHESIS_DIR) if f.endswith('.txt')]
        if not files_to_judge:
             print(f"No .txt files found in {SYNTHESIS_DIR} to judge.")
             exit(0)
        print(f"Found {len(files_to_judge)} files to judge based on '{SYNTHESIS_DIR}'.")
    except Exception as e:
        print(f"Error listing files in {SYNTHESIS_DIR}: {e}")
        exit(1)


    success_count = 0
    skipped_count = 0
    error_count = 0
    total_files = len(files_to_judge)
    processed_count = 0

    # Use a session object for connection pooling
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # Submit tasks
            futures = {executor.submit(judge_file_pair, filename, session, headers): filename for filename in files_to_judge}

            print(f"Submitted {len(futures)} judging tasks to {NUM_THREADS} threads...")

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                processed_count += 1
                try:
                    _ , status = future.result()
                    if status == "success":
                        success_count += 1
                    elif status == "skipped":
                        skipped_count += 1
                    else:
                        error_count += 1
                        # Error details are printed within the worker function
                        # print(f"Failed ({status}): {filename}")

                    # Print progress update
                    if processed_count % 100 == 0 or processed_count == total_files:
                         print(f"Progress: {processed_count}/{total_files} | Success: {success_count} | Skipped: {skipped_count} | Errors: {error_count}")

                except Exception as e:
                    error_count += 1
                    print(f"Error getting result for {filename}: {e}")

    print("\n--- Judging Complete ---")
    print(f"Total file pairs: {total_files}")
    print(f"Successfully judged: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Judgment files saved in '{JUDGE_DIR}'.")
