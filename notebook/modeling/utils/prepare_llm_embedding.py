import os
import requests
import json
import time
import threading # For thread identification in logs
import pandas as pd # Added for type hinting and series processing
import concurrent.futures # Added for multi-threading
import gzip
import csv # Added for CSV writing
from typing import Union, List, Optional, Dict, Any # For type hinting
from dotenv import load_dotenv, find_dotenv # Ensure dotenv is imported

_ = load_dotenv(find_dotenv())

# --- Configuration ---
LITELLM_ENDPOINT = os.getenv('LITELLM_BASE_URL') + '/v1/embeddings' # type: ignore
# IMPORTANT: Replace with your actual token or load from environment variables
API_KEY = os.getenv('LITELLM_API_KEY')
MODEL_NAME = os.getenv('LITELLM_EMBEDDING_MODEL', "gemini/text-embedding-004") # Added default and env var
REQUEST_TIMEOUT = 60  # seconds
RETRY_DELAY = 5  # seconds
MAX_RETRIES = 3
# --- End Configuration ---

def get_embeddings(text_to_embed: str) -> Optional[List[float]]:
    """
    Sends a single text to the embedding API.
    Returns an embedding vector (list of floats), or None if the request failed.
    The API is expected to receive {"input": [text_to_embed], ...}
    """
    thread_id = threading.get_ident()
    if not text_to_embed: # Should be pre-validated, but good practice
        return None
        
    print(f"[Thread-{thread_id}] Requesting embedding for text: \"{text_to_embed[:50]}...\"")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # API expects a list of strings, even for a single item
    payload = json.dumps({
        "input": [text_to_embed], 
        "model": MODEL_NAME
    })

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(
                LITELLM_ENDPOINT, # type: ignore
                headers=headers,
                data=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            print(f"[Thread-{thread_id}] API call successful for text.")

            api_embeddings_data = response_data.get("data", [])
            
            if api_embeddings_data and len(api_embeddings_data) == 1:
                embedding_object = api_embeddings_data[0]
                embedding_vector = embedding_object.get("embedding")
                if embedding_vector:
                    return embedding_vector
                else:
                    print(f"[Thread-{thread_id}] Warning: API returned an embedding object without an 'embedding' field for text.")
                    return None
            else:
                print(f"[Thread-{thread_id}] Warning: API did not return expected data structure (1 embedding object). Found {len(api_embeddings_data)} items.")
                return None

        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"[Thread-{thread_id}] Error calling API for text: {e}. Retrying ({retries}/{MAX_RETRIES})...")
            if retries >= MAX_RETRIES:
                print(f"[Thread-{thread_id}] Max retries reached for text. Skipping.")
                return None
            time.sleep(RETRY_DELAY)
        except json.JSONDecodeError as e:
            print(f"[Thread-{thread_id}] Error decoding API response for text: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
            return None # Indicate failure for this text
    
    return None # Fallback after retries

# Helper function to prepare individual embedding tasks from the series
def _prepare_embedding_tasks(
    processed_texts_series: pd.Series,
    id_series: Optional[pd.Series],
    additional_data: Optional[Dict[str, Union[List[Any], pd.Series]]],
    ordered_additional_column_names: List[str], # To maintain order
    output_csv_path: Optional[str],
    csv_writer_lock: Optional[threading.Lock],
    stats: Dict[str, int],
    all_embeddings_placeholders: Optional[List[Optional[List[float]]]]
) -> List[Dict[str, Any]]:
    """
    Prepares individual embedding tasks, including any additional data.
    Identifies valid texts and handles immediate logging of skipped invalid texts if in CSV mode.
    """
    tasks = []
    num_texts = len(processed_texts_series)

    for i in range(num_texts):
        text_content = processed_texts_series.iloc[i]
        current_id = id_series.iloc[i] if id_series is not None else processed_texts_series.index[i]
        
        additional_values_for_row = []
        if additional_data:
            for col_name in ordered_additional_column_names:
                data_source = additional_data[col_name]
                if isinstance(data_source, pd.Series):
                    additional_values_for_row.append(data_source.iloc[i])
                else: # Assumed to be a list
                    additional_values_for_row.append(data_source[i])

        if isinstance(text_content, str) and text_content.strip():
            tasks.append({
                "text_to_embed": text_content,
                "id": current_id,
                "original_series_global_index": i,
                "additional_values_ordered": additional_values_for_row
            })
        else:
            stats["skipped_invalid_text"] += 1
            if output_csv_path and csv_writer_lock:
                row_to_write = [current_id, text_content, None, "skipped_invalid_text"] + additional_values_for_row
                with csv_writer_lock:
                    with gzip.open(output_csv_path, 'at', newline='', encoding='utf-8') as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow(row_to_write)
            elif all_embeddings_placeholders is not None:
                # Note: additional_values are not stored in-memory for skipped texts if not writing to CSV
                all_embeddings_placeholders[i] = None
        
    return tasks

# Helper function to process the result of a single completed embedding task
def _process_completed_task(
    task_payload: Dict[str, Any],
    embedding_result: Optional[List[float]], 
    is_exception: bool,
    output_csv_path: Optional[str],
    csv_writer_lock: Optional[threading.Lock],
    all_embeddings_placeholders: Optional[List[Optional[List[float]]]],
    stats: Dict[str, int]
):
    """
    Processes the result for a single completed text embedding task.
    Updates stats and writes to CSV (including additional data) or in-memory list.
    """
    text_id = task_payload["id"]
    text_content = task_payload["text_to_embed"]
    original_series_global_index = task_payload["original_series_global_index"]
    additional_values_ordered = task_payload["additional_values_ordered"]

    status_message = ""
    final_embedding_for_text = None

    if is_exception:
        status_message = "failed_exception_in_processing"
        print(f"Error processing text ID {text_id}: {status_message}.")
    elif embedding_result is None:
        status_message = "failed_embedding_api"
        print(f"Failed to get embedding for text ID {text_id}.")
    else:
        status_message = "success"
        final_embedding_for_text = embedding_result
        print(f"Successfully got embedding for text ID {text_id}.")

    if final_embedding_for_text:
        stats["processed_successfully"] += 1
    else:
        stats["failed_embedding_or_processing"] += 1
    
    if output_csv_path and csv_writer_lock:
        embedding_json = json.dumps(final_embedding_for_text) if final_embedding_for_text else None
        row_to_write = [text_id, text_content, embedding_json, status_message] + additional_values_ordered
        with csv_writer_lock:
            with gzip.open(output_csv_path, 'at', newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(row_to_write)
    elif all_embeddings_placeholders is not None:
        # Note: additional_values are not stored with in-memory results
        all_embeddings_placeholders[original_series_global_index] = final_embedding_for_text

def generate_embeddings_from_series(
    processed_texts_series: pd.Series,
    id_series: Optional[pd.Series] = None,
    additional_data: Optional[Dict[str, Union[List[Any], pd.Series]]] = None, # New parameter
    max_workers: int = 10,
    output_csv_path: Optional[str] = None
) -> Union[List[Optional[List[float]]], Dict[str, Any]]:
    """
    Generates embeddings for each text in a pandas Series using multi-threading.
    Each text is sent individually to the embedding API.
    Optionally saves results incrementally to a CSV file, including additional custom columns.
    Allows providing a custom id_series for the output CSV.
    """
    # --- Input Validation ---
    if not isinstance(processed_texts_series, pd.Series):
        raise TypeError("Input 'processed_texts_series' must be a pandas Series.")
    num_texts = len(processed_texts_series)

    if id_series is not None:
        if not isinstance(id_series, pd.Series):
            raise TypeError("Input 'id_series' must be a pandas Series if provided.")
        if len(id_series) != num_texts:
            raise ValueError("Inputs 'id_series' and 'processed_texts_series' must have the same length.")

    fixed_headers = ["id", "text_content", "embedding_json", "status"]
    ordered_additional_column_names: List[str] = []
    if additional_data is not None:
        if not isinstance(additional_data, dict):
            raise TypeError("'additional_data' must be a dictionary if provided.")
        for col_name, col_values in additional_data.items():
            if not isinstance(col_name, str):
                raise TypeError("Keys in 'additional_data' must be strings (column names).")
            if col_name in fixed_headers:
                raise ValueError(f"Column name '{col_name}' in 'additional_data' conflicts with a fixed column name.")
            if not isinstance(col_values, (list, pd.Series)):
                raise TypeError(f"Values in 'additional_data' (for column '{col_name}') must be lists or pandas Series.")
            if len(col_values) != num_texts:
                raise ValueError(
                    f"Length of data for additional column '{col_name}' ({len(col_values)}) "
                    f"does not match length of 'processed_texts_series' ({num_texts})."
                )
            ordered_additional_column_names.append(col_name) # Store in a fixed order

    # --- Initialization ---
    stats = {"processed_successfully": 0, "failed_embedding_or_processing": 0, "skipped_invalid_text": 0}
    if num_texts == 0:
        # Still write header if CSV path is given for an empty operation
        if output_csv_path:
            try:
                with gzip.open(output_csv_path, 'wt', newline='', encoding='utf-8') as f_header:
                    csv.writer(f_header).writerow(fixed_headers + ordered_additional_column_names)
            except IOError as e:
                 print(f"Warning: Could not write CSV header for empty series to {output_csv_path}: {e}")
        return [] if output_csv_path is None else {**stats, "output_path": output_csv_path, "total_texts_in_series": num_texts}

    all_embeddings_placeholders: Optional[List[Optional[List[float]]]] = None
    csv_writer_lock: Optional[threading.Lock] = None
    final_csv_header = fixed_headers + ordered_additional_column_names

    if output_csv_path:
        print(f"Starting embedding generation for {num_texts} texts. Results will be saved to {output_csv_path}.")
        csv_writer_lock = threading.Lock()
        try:
            with gzip.open(output_csv_path, 'wt', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(final_csv_header)
        except IOError as e:
            print(f"Error: Could not write CSV header to {output_csv_path}: {e}")
            raise
    else:
        all_embeddings_placeholders = [None] * num_texts
        print(f"Starting embedding generation for {num_texts} texts. Results will be stored in memory.")
    
    print(f"Using up to {max_workers} concurrent API calls...")

    # --- Prepare Tasks ---
    embedding_tasks = _prepare_embedding_tasks(
        processed_texts_series, id_series, additional_data, ordered_additional_column_names,
        output_csv_path, csv_writer_lock, stats, all_embeddings_placeholders
    )
    
    # --- Execute Tasks Concurrently ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_payload = {
            executor.submit(get_embeddings, task_payload["text_to_embed"]): task_payload
            for task_payload in embedding_tasks # Each task is for one text
        }

        for future in concurrent.futures.as_completed(future_to_task_payload):
            task_payload = future_to_task_payload[future]
            try:
                embedding_result = future.result() # This is Optional[List[float]]
                _process_completed_task(
                    task_payload, embedding_result, False, # is_exception = False
                    output_csv_path, csv_writer_lock, all_embeddings_placeholders, stats
                )
            except Exception as exc:
                print(f"Task for text ID {task_payload['id']} generated an exception: {exc}")
                _process_completed_task(
                    task_payload, None, True, # embedding_result = None, is_exception = True
                    output_csv_path, csv_writer_lock, all_embeddings_placeholders, stats
                )
            
    print(f"Embedding generation finished.")
    if output_csv_path:
        summary = {**stats, "output_path": output_csv_path, "total_texts_in_series": num_texts}
        print(f"Summary: {summary}")
        return summary
    else:
        if all_embeddings_placeholders and len(all_embeddings_placeholders) != num_texts:
            print(f"Critical Warning: Final number of in-memory results ({len(all_embeddings_placeholders)}) does not match input texts ({num_texts}).")
        print(f"Total embeddings/placeholders generated in memory: {len(all_embeddings_placeholders) if all_embeddings_placeholders else 0}")
        return all_embeddings_placeholders if all_embeddings_placeholders is not None else []