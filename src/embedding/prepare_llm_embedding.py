import os
import requests
import json
import time
import re # Import regex for tag extraction

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# --- Configuration ---
LITELLM_ENDPOINT = os.getenv('LITELLM_BASE_URL') + '/v1/embeddings'
# IMPORTANT: Replace with your actual token or load from environment variables
API_KEY = os.getenv('LITELLM_API_KEY')
MODEL_NAME = "gemini/text-embedding-004"
INPUT_DIR = "data/synthesis/"
OUTPUT_DIR = "data/embeddings/llm_embeddings"
# BATCH_SIZE = 100 # No longer needed
REQUEST_TIMEOUT = 60 # seconds
RETRY_DELAY = 5 # seconds
MAX_RETRIES = 3
# MAX_WORKERS = 10 # No longer needed
# --- End Configuration ---

def get_embeddings(texts_batch):
    """Sends a batch of texts to the embedding API and returns embeddings."""
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
            return response.json().get("data", []) # Assuming response structure {"data": [...]}
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Error calling API: {e}. Retrying ({retries}/{MAX_RETRIES})...")
            if retries >= MAX_RETRIES:
                print(f"  Max retries reached for batch. Skipping.")
                return None # Indicate failure
            time.sleep(RETRY_DELAY)
        except json.JSONDecodeError as e:
             print(f"Error decoding API response: {e}. Response text: {response.text}")
             return None # Indicate failure

# process_single_line function removed

def process_file(input_filepath, output_dir):
    """
    Reads an entire file, extracts content within the first <response> tag,
    gets a single embedding for that extracted content if the output file doesn't exist,
    and saves the embedding to a corresponding JSON file (e.g., 0.json).
    """
    print(f"Processing {input_filepath}...")
    # Output directory is assumed to exist (created in main)

    # Extract the base name of the input file for naming the output file
    input_file_base_name = os.path.splitext(os.path.basename(input_filepath))[0]
    output_filename = f"{input_file_base_name}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    # --- Check if output file already exists ---
    if os.path.exists(output_filepath):
        print(f"  Output file {output_filepath} already exists. Skipping.")
        return
    # --- End check ---

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            # Read the entire file content
            full_content = f.read()
    except Exception as e:
        print(f"  Error reading file {input_filepath}: {e}")
        return

    if not full_content.strip():
        print(f"  No content found in {input_filepath}. Skipping.")
        return

    # Extract content within the first <response>...</response> block
    # Use re.search to find the first match. Group 1 captures the content inside.
    # DOTALL allows '.' to match newlines, IGNORECASE makes it case-insensitive.
    match = re.search(r'<response>(.*?)</response>', full_content, flags=re.IGNORECASE | re.DOTALL)

    if not match:
        print(f"Alert: No <response>...</response> tags found in {input_filepath}. Skipping file.")
        return # Skip if tags are not found

    # Get the captured group (the content inside the tags)
    extracted_content = match.group(1).strip()

    if not extracted_content:
        print(f"Content inside <response> tags is empty in {input_filepath}. Skipping.")
        return

    print(f"Content extracted from <response> tag. Requesting single embedding...")

    # Get embedding for the extracted content
    # Note: get_embeddings expects a list, so wrap the content in a list
    embeddings_data_list = get_embeddings([extracted_content])

    # Check if the API call failed
    if embeddings_data_list is None:
        print(f"Failed to get embedding for {input_filepath}. Skipping.")
        return

    # Check if we got exactly one embedding back
    if len(embeddings_data_list) != 1:
         print(f"Warning: Expected 1 embedding for {input_filepath}, but received {len(embeddings_data_list)}. Skipping.")
         return

    # Extract the single embedding object
    embedding_data = embeddings_data_list[0]

    # Save the single embedding
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2)
        print(f"Embedding saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving embedding to {output_filepath}: {e}")


def main():
    """Main function to orchestrate the embedding generation."""
    print("Starting embedding generation...")
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        # Get all .txt files, handling potential errors during listing
        files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt") and os.path.isfile(os.path.join(INPUT_DIR, f))]
    except OSError as e:
        print(f"Error listing files in {INPUT_DIR}: {e}")
        return

    if not files_to_process:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    print(f"Found {len(files_to_process)} .txt files to process.")

    start_time = time.time()
    processed_count = 0
    skipped_count = 0 # Count skipped files
    failed_count = 0

    for filename in files_to_process:
        input_filepath = os.path.join(INPUT_DIR, filename)
        # Check if output exists *before* calling process_file to update counts correctly
        output_filename = f"{os.path.splitext(filename)[0]}.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_filepath):
             print(f"Processing {input_filepath}...") # Print initial message even if skipping
             print(f"  Output file {output_filepath} already exists. Skipping.")
             skipped_count += 1
             print("-" * 20) # Separator between files
             continue # Move to the next file

        # If output doesn't exist, proceed with processing
        try:
             # Pass the main OUTPUT_DIR directly to process_file
             process_file(input_filepath, OUTPUT_DIR)
             processed_count +=1 # Count as processed only if not skipped and no exception
        except Exception as e:
             print(f"Unexpected error processing file {filename}: {e}")
             failed_count += 1
        print("-" * 20) # Separator between files


    end_time = time.time()
    total_time = end_time - start_time
    print("\nEmbedding generation finished.")
    print(f"Total files processed: {processed_count}")
    print(f"Total files skipped (already exist): {skipped_count}")
    print(f"Total files failed: {failed_count}")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
