import os
import requests
import json
import time
import re # Import regex for tag extraction
import concurrent.futures
import functools # For partial function application
import threading # Import the correct module for thread identification

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# Number of threads to use
MAX_WORKERS = 20

# --- Configuration ---
LITELLM_ENDPOINT = os.getenv('LITELLM_BASE_URL') + '/v1/embeddings'
# IMPORTANT: Replace with your actual token or load from environment variables
API_KEY = os.getenv('LITELLM_API_KEY')
MODEL_NAME = "model2vec/potion-base-8m"
INPUT_DIR = "data/synthetic_feedback/output"
OUTPUT_DIR = "data/embeddings/distil_sbert_embeddings"
REQUEST_TIMEOUT = 60 # seconds
RETRY_DELAY = 5 # seconds
MAX_RETRIES = 3
# --- End Configuration ---

# Status constants for return values
STATUS_PROCESSED = "processed"
STATUS_SKIPPED_EXISTS = "skipped_exists"
STATUS_SKIPPED_NO_CONTENT = "skipped_no_content"
STATUS_SKIPPED_NO_TAG = "skipped_no_tag"
STATUS_SKIPPED_EMPTY_TAG = "skipped_empty_tag"
STATUS_FAILED_READ = "failed_read"
STATUS_FAILED_API = "failed_api"
STATUS_FAILED_SAVE = "failed_save"

def get_embeddings(texts_batch):
    """Sends a batch of texts to the embedding API and returns embeddings."""
    # Add thread identifier to logs for clarity (optional but helpful)
    thread_id = threading.get_ident() # Use threading.get_ident()
    print(f"[Thread-{thread_id}] Requesting embedding for batch size {len(texts_batch)}...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "input": texts_batch,
        "model": MODEL_NAME
    })

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(
                LITELLM_ENDPOINT,
                headers=headers,
                data=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            print(f"[Thread-{thread_id}] API call successful.")
            return response.json().get("data", []) # Assuming response structure {"data": [...]}
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"[Thread-{thread_id}] Error calling API: {e}. Retrying ({retries}/{MAX_RETRIES})...")
            if retries >= MAX_RETRIES:
                print(f"[Thread-{thread_id}] Max retries reached for batch. Skipping.")
                return None # Indicate failure
            time.sleep(RETRY_DELAY)
        except json.JSONDecodeError as e:
             print(f"[Thread-{thread_id}] Error decoding API response: {e}. Response text: {response.text}")
             return None # Indicate failure

def process_file(input_filepath, output_dir):
    """
    Reads a file, extracts content, gets embedding (if output doesn't exist), saves.
    Returns a status string indicating the outcome.
    """
    thread_id = threading.get_ident() # Use threading.get_ident()
    print(f"[Thread-{thread_id}] Processing {input_filepath}...")

    input_file_base_name = os.path.splitext(os.path.basename(input_filepath))[0]
    output_filename = f"{input_file_base_name}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    # --- Check if output file already exists ---
    if os.path.exists(output_filepath):
        print(f"[Thread-{thread_id}] Output file {output_filepath} already exists. Skipping.")
        return STATUS_SKIPPED_EXISTS
    # --- End check ---

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            full_content = f.read()
    except Exception as e:
        print(f"[Thread-{thread_id}] Error reading file {input_filepath}: {e}")
        return STATUS_FAILED_READ

    if not full_content.strip():
        print(f"[Thread-{thread_id}] No content found in {input_filepath}. Skipping.")
        return STATUS_SKIPPED_NO_CONTENT

    match = re.search(r'<response>(.*?)</response>', full_content, flags=re.IGNORECASE | re.DOTALL)

    if not match:
        print(f"[Thread-{thread_id}] Alert: No <response>...</response> tags found in {input_filepath}. Skipping file.")
        return STATUS_SKIPPED_NO_TAG

    extracted_content = match.group(1).strip()

    if not extracted_content:
        print(f"[Thread-{thread_id}] Content inside <response> tags is empty in {input_filepath}. Skipping.")
        return STATUS_SKIPPED_EMPTY_TAG

    # print(f"[Thread-{thread_id}] Content extracted from <response> tag. Requesting single embedding...") # Moved to get_embeddings

    embeddings_data_list = get_embeddings([extracted_content])

    if embeddings_data_list is None:
        print(f"[Thread-{thread_id}] Failed to get embedding for {input_filepath}. Skipping.")
        return STATUS_FAILED_API

    if len(embeddings_data_list) != 1:
         print(f"[Thread-{thread_id}] Warning: Expected 1 embedding for {input_filepath}, but received {len(embeddings_data_list)}. Skipping.")
         # Treat this as an API failure for counting purposes
         return STATUS_FAILED_API

    embedding_data = embeddings_data_list[0]

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2)
        print(f"[Thread-{thread_id}] Embedding saved to {output_filepath}")
        return STATUS_PROCESSED
    except Exception as e:
        print(f"[Thread-{thread_id}] Error saving embedding to {output_filepath}: {e}")
        return STATUS_FAILED_SAVE


def main():
    """Main function to orchestrate the embedding generation using multithreading."""
    print(f"Starting embedding generation with {MAX_WORKERS} workers...")
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        filenames = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt") and os.path.isfile(os.path.join(INPUT_DIR, f))]
        filepaths = [os.path.join(INPUT_DIR, fname) for fname in filenames]
    except OSError as e:
        print(f"Error listing files in {INPUT_DIR}: {e}")
        return

    if not filepaths:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    print(f"Found {len(filepaths)} .txt files to process.")

    start_time = time.time()
    results = {
        STATUS_PROCESSED: 0,
        STATUS_SKIPPED_EXISTS: 0,
        STATUS_SKIPPED_NO_CONTENT: 0,
        STATUS_SKIPPED_NO_TAG: 0,
        STATUS_SKIPPED_EMPTY_TAG: 0,
        STATUS_FAILED_READ: 0,
        STATUS_FAILED_API: 0,
        STATUS_FAILED_SAVE: 0,
        "failed_unexpected": 0 # For errors within the executor/map itself
    }

    # Create a partial function to pass the fixed output_dir argument
    process_func = functools.partial(process_file, output_dir=OUTPUT_DIR)

    # Use ThreadPoolExecutor to process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use executor.map to apply the function to each filepath
        # map returns an iterator of results in the order the tasks were submitted
        future_to_filepath = {executor.submit(process_func, fp): fp for fp in filepaths}
        for future in concurrent.futures.as_completed(future_to_filepath):
            filepath = future_to_filepath[future]
            try:
                status = future.result()
                if status in results:
                    results[status] += 1
                else:
                    print(f"Warning: Unknown status '{status}' returned for file {filepath}")
                    results["failed_unexpected"] += 1
            except Exception as exc:
                print(f"Error processing file {filepath}: {exc}")
                results["failed_unexpected"] += 1

    # --- Aggregated Counts ---
    total_processed = results[STATUS_PROCESSED]
    total_skipped = (results[STATUS_SKIPPED_EXISTS] +
                     results[STATUS_SKIPPED_NO_CONTENT] +
                     results[STATUS_SKIPPED_NO_TAG] +
                     results[STATUS_SKIPPED_EMPTY_TAG])
    total_failed = (results[STATUS_FAILED_READ] +
                    results[STATUS_FAILED_API] +
                    results[STATUS_FAILED_SAVE] +
                    results["failed_unexpected"])

    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Embedding Generation Finished ---")
    print(f"Total files processed successfully: {total_processed}")
    print(f"Total files skipped:")
    print(f"  - Already existed: {results[STATUS_SKIPPED_EXISTS]}")
    print(f"  - No content: {results[STATUS_SKIPPED_NO_CONTENT]}")
    print(f"  - No <response> tag: {results[STATUS_SKIPPED_NO_TAG]}")
    print(f"  - Empty <response> tag: {results[STATUS_SKIPPED_EMPTY_TAG]}")
    print(f"Total files failed:")
    print(f"  - Read error: {results[STATUS_FAILED_READ]}")
    print(f"  - API/Embedding error: {results[STATUS_FAILED_API]}")
    print(f"  - Save error: {results[STATUS_FAILED_SAVE]}")
    print(f"  - Unexpected error: {results['failed_unexpected']}")
    print("-" * 20)
    print(f"Overall Skipped: {total_skipped}")
    print(f"Overall Failed: {total_failed}")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
