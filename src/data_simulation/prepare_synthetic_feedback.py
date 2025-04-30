import os
import requests
import json
import time
import re # Import regex module
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())

# control the number of thread workers
NUM_THREADS = 10

# --- Configuration ---
RAW_DIR = 'data/synthetic_feedback/raw'
SYNTHESIS_DIR = 'data/synthetic_feedback/output'
LITELLM_ENDPOINT = os.getenv('LITELLM_BASE_URL') + '/v1/chat/completions'
API_KEY = os.getenv('LITELLM_API_KEY')

MODEL_NAME = 'xai/grok-3-beta' # As per your curl example
TEMPERATURE = 0.7
MAX_TOKENS = 300

MAX_RETRIES = 3
INITIAL_BACKOFF = 1 # seconds


# --- Prompt Template ---
PROMPT_TEMPLATE = """You are tasked with generating realistic, detailed customer feedback based on the provided customer data profile.

Analyze the entire customer profile below, paying attention to all sections (feedback category, complaints, engagement, offers, recency, customer info).

Synthesize a specific, plausible customer feedback statement that reflects the original feedback category but elaborates on it, potentially incorporating details from other parts of the profile. Aim for a feedback statement a real customer might provide.

Output ONLY the synthesized feedback statement, enclosed within <response></response> tags. Do not add any explanation or introductory text before or after the tags.

Customer Data Profile:
---
{original_content}
---

Synthesized Feedback:"""
# Note: The script itself handles the 'No reason specified' case before calling the API.
# The LLM will generate the content that goes inside <response></response> tags.

# --- Helper Functions ---

def make_api_request(session, payload, headers):
    """Sends the request to the LiteLLM API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(LITELLM_ENDPOINT, headers=headers, json=payload, timeout=60) # 60 second timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Request timed out (attempt {attempt + 1}/{MAX_RETRIES}). Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            # Specific check for potentially transient errors like 5xx or rate limits (429)
            if hasattr(e, 'response') and e.response is not None:
                 if e.response.status_code == 429 or e.response.status_code >= 500:
                     print("Potential transient error detected.")
                 else:
                     # Don't retry for other client errors (e.g., 400, 401, 403)
                     print(f"Non-retryable client error: {e.response.status_code}")
                     return None # Indicate failure
            else:
                 # Network error without response, could be transient
                 print("Network error without response.")

        if attempt < MAX_RETRIES - 1:
            backoff_time = INITIAL_BACKOFF * (2 ** attempt)
            print(f"Waiting {backoff_time}s before next retry...")
            time.sleep(backoff_time)
        else:
            print("Max retries reached. Request failed.")
            return None # Indicate failure after max retries
    return None # Should not be reached, but ensures a return value

def process_file(filepath, session, headers):
    """Reads a file, sends its content to the API, and saves the response."""
    filename = os.path.basename(filepath)
    output_filepath = os.path.join(SYNTHESIS_DIR, filename)

    # Skip if output file already exists
    if os.path.exists(output_filepath):
        # print(f"Skipping {filename}: Output already exists.")
        return filename, "skipped"

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Extract feedback category using regex
        match = re.search(r'<feedback_category>(.*?)</feedback_category>', original_content, re.IGNORECASE | re.DOTALL)
        feedback_category = match.group(1).strip() if match else None

        # Check if feedback category is 'No reason specified'
        if feedback_category == 'No reason specified':
            # Write the specific response and skip API call
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                outfile.write('<response>No reason specified</response>')
            return filename, "no_reason_specified" # Return a new status

        # If not 'No reason specified', proceed with API call
        prompt = PROMPT_TEMPLATE.format(original_content=original_content)

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }

        api_response = make_api_request(session, payload, headers)

        if api_response:
            try:
                # Extract the synthesized feedback (adjust path if necessary based on actual API response structure)
                synthesized_feedback = api_response['choices'][0]['message']['content']

                # Write the synthesized feedback to the output file
                with open(output_filepath, 'w', encoding='utf-8') as outfile:
                    outfile.write(synthesized_feedback)
                # print(f"Successfully processed {filename}")
                return filename, "success"
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error parsing API response for {filename}: {e}. Response: {api_response}")
                return filename, "parse_error"
        else:
            print(f"API request failed for {filename} after retries.")
            return filename, "api_failed"

    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return filename, "not_found"
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}")
        return filename, "error"

# --- Main Execution ---
if __name__ == "__main__":
    if not API_KEY:
        print("Error: LITELLM_API_KEY environment variable not set.")
        exit(1)

    if not os.path.isdir(RAW_DIR):
        print(f"Error: Raw directory '{RAW_DIR}' not found.")
        exit(1)

    os.makedirs(SYNTHESIS_DIR, exist_ok=True)
    print(f"Ensured synthesis directory '{SYNTHESIS_DIR}' exists.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Get list of files to process
    try:
        files_to_process = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith('.txt')]
        if not files_to_process:
             print(f"No .txt files found in {RAW_DIR}")
             exit(0)
        print(f"Found {len(files_to_process)} files to process in '{RAW_DIR}'.")
    except Exception as e:
        print(f"Error listing files in {RAW_DIR}: {e}")
        exit(1)


    success_count = 0
    skipped_count = 0
    no_reason_count = 0 # Add counter for the new status
    error_count = 0
    total_files = len(files_to_process)
    processed_count = 0

    # Use a session object for connection pooling
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # Submit tasks
            futures = {executor.submit(process_file, filepath, session, headers): filepath for filepath in files_to_process}

            print(f"Submitted {len(futures)} tasks to {NUM_THREADS} threads...")

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                filepath = futures[future]
                filename = os.path.basename(filepath)
                processed_count += 1
                try:
                    _ , status = future.result()
                    if status == "success":
                        success_count += 1
                    elif status == "skipped":
                        skipped_count += 1
                    elif status == "no_reason_specified":
                        no_reason_count += 1
                    else: # Covers 'api_failed', 'parse_error', 'not_found', 'error'
                        error_count += 1
                        print(f"Failed ({status}): {filename}")

                    # Print progress update
                    if processed_count % 100 == 0 or processed_count == total_files:
                         print(f"Progress: {processed_count}/{total_files} | Success: {success_count} | Skipped: {skipped_count} | No Reason: {no_reason_count} | Errors: {error_count}")

                except Exception as e:
                    error_count += 1
                    print(f"Error getting result for {filename}: {e}")

    print("\n--- Processing Complete ---")
    print(f"Total files: {total_files}")
    print(f"Successfully synthesized (API call): {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Skipped (No reason specified): {no_reason_count}")
    print(f"Errors: {error_count}")
    print(f"Synthesized files saved in '{SYNTHESIS_DIR}'.")
