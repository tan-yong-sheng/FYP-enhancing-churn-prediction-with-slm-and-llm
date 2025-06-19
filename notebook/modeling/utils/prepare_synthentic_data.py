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
LITELLM_ENDPOINT = os.getenv('LITELLM_BASE_URL') + '/v1/chat/completions' # type: ignore
# IMPORTANT: Replace with your actual token or load from environment variables
API_KEY = os.getenv('LITELLM_API_KEY')
MODEL_NAME = os.getenv('LITELLM_MODEL', "gemini/gemini-2.0-flash") # Added default and env var
REQUEST_TIMEOUT = 60  # seconds
RETRY_DELAY = 5  # seconds
MAX_RETRIES = 3
# --- End Configuration ---

def generate_text(prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
    """
    Sends a prompt to the text generation API.
    Returns generated text (string), or None if the request failed.
    """
    thread_id = threading.get_ident()
    if not prompt: # Should be pre-validated, but good practice
        return None
        
    print(f"[Thread-{thread_id}] Requesting text generation for prompt: \"{prompt[:50]}...\"")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Prepare messages for chat completion
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
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
            print(f"[Thread-{thread_id}] API call successful for prompt.")

            choices = response_data.get("choices", [])
            
            if choices and len(choices) > 0:
                choice = choices[0]
                message = choice.get("message", {})
                generated_text = message.get("content")
                if generated_text:
                    return generated_text.strip()
                else:
                    print(f"[Thread-{thread_id}] Warning: API returned a choice without content for prompt.")
                    return None
            else:
                print(f"[Thread-{thread_id}] Warning: API did not return expected data structure (choices). Found {len(choices)} choices.")
                return None

        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"[Thread-{thread_id}] Error calling API for prompt: {e}. Retrying ({retries}/{MAX_RETRIES})...")
            if retries >= MAX_RETRIES:
                print(f"[Thread-{thread_id}] Max retries reached for prompt. Skipping.")
                return None
            time.sleep(RETRY_DELAY)
        except json.JSONDecodeError as e:
            print(f"[Thread-{thread_id}] Error decoding API response for prompt: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
            return None # Indicate failure for this prompt
    
    return None # Fallback after retries

# Helper function to prepare individual generation tasks from the series
def _prepare_generation_tasks(
    prompts_series: pd.Series,
    id_series: Optional[pd.Series],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    additional_data: Optional[Dict[str, Union[List[Any], pd.Series]]],
    ordered_additional_column_names: List[str], # To maintain order
    output_csv_path: Optional[str],
    csv_writer_lock: Optional[threading.Lock],
    stats: Dict[str, int],
    all_generations_placeholders: Optional[List[Optional[str]]]
) -> List[Dict[str, Any]]:
    """
    Prepares individual generation tasks, including any additional data.
    Identifies valid prompts and handles immediate logging of skipped invalid prompts if in CSV mode.
    """
    tasks = []
    num_prompts = len(prompts_series)

    for i in range(num_prompts):
        prompt_content = prompts_series.iloc[i]
        current_id = id_series.iloc[i] if id_series is not None else prompts_series.index[i]
        
        additional_values_for_row = []
        if additional_data:
            for col_name in ordered_additional_column_names:
                data_source = additional_data[col_name]
                if isinstance(data_source, pd.Series):
                    additional_values_for_row.append(data_source.iloc[i])
                else: # Assumed to be a list
                    additional_values_for_row.append(data_source[i])

        if isinstance(prompt_content, str) and prompt_content.strip():
            tasks.append({
                "prompt": prompt_content,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "id": current_id,
                "original_series_global_index": i,
                "additional_values_ordered": additional_values_for_row
            })
        else:
            stats["skipped_invalid_prompt"] += 1
            if output_csv_path and csv_writer_lock:
                row_to_write = [current_id, prompt_content, None, "skipped_invalid_prompt"] + additional_values_for_row
                with csv_writer_lock:
                    with gzip.open(output_csv_path, 'at', newline='', encoding='utf-8') as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow(row_to_write)
            elif all_generations_placeholders is not None:
                # Note: additional_values are not stored in-memory for skipped prompts if not writing to CSV
                all_generations_placeholders[i] = None
        
    return tasks

# Helper function to process the result of a single completed generation task
def _process_completed_task(
    task_payload: Dict[str, Any],
    generation_result: Optional[str], 
    is_exception: bool,
    output_csv_path: Optional[str],
    csv_writer_lock: Optional[threading.Lock],
    all_generations_placeholders: Optional[List[Optional[str]]],
    stats: Dict[str, int]
):
    """
    Processes the result for a single completed text generation task.
    Updates stats and writes to CSV (including additional data) or in-memory list.
    """
    prompt_id = task_payload["id"]
    prompt_content = task_payload["prompt"]
    original_series_global_index = task_payload["original_series_global_index"]
    additional_values_ordered = task_payload["additional_values_ordered"]

    status_message = ""
    final_generation_for_prompt = None

    if is_exception:
        status_message = "failed_exception_in_processing"
        print(f"Error processing prompt ID {prompt_id}: {status_message}.")
    elif generation_result is None:
        status_message = "failed_generation_api"
        print(f"Failed to get generation for prompt ID {prompt_id}.")
    else:
        status_message = "success"
        final_generation_for_prompt = generation_result
        print(f"Successfully got generation for prompt ID {prompt_id}.")

    if final_generation_for_prompt:
        stats["processed_successfully"] += 1
    else:
        stats["failed_generation_or_processing"] += 1
    
    if output_csv_path and csv_writer_lock:
        row_to_write = [prompt_id, prompt_content, final_generation_for_prompt, status_message] + additional_values_ordered
        with csv_writer_lock:
            with gzip.open(output_csv_path, 'at', newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(row_to_write)
    elif all_generations_placeholders is not None:
        # Note: additional_values are not stored with in-memory results
        all_generations_placeholders[original_series_global_index] = final_generation_for_prompt

def generate_text_from_series(
    prompts_series: pd.Series,
    id_series: Optional[pd.Series] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    additional_data: Optional[Dict[str, Union[List[Any], pd.Series]]] = None,
    max_workers: int = 10,
    output_csv_path: Optional[str] = None
) -> Union[List[Optional[str]], Dict[str, Any]]:
    """
    Generates text for each prompt in a pandas Series using multi-threading.
    Each prompt is sent individually to the text generation API.
    Optionally saves results incrementally to a CSV file, including additional custom columns.
    Allows providing a custom id_series for the output CSV.
    
    Args:
        prompts_series: pandas Series containing prompts for text generation
        id_series: Optional pandas Series with custom IDs for each prompt
        system_prompt: Optional system prompt to include in all requests
        max_tokens: Maximum number of tokens to generate per request
        temperature: Temperature for text generation (0.0 to 2.0)
        additional_data: Optional dictionary of additional columns to include in output
        max_workers: Maximum number of concurrent API calls
        output_csv_path: Optional path to save results as CSV (gzipped)
    
    Returns:
        List of generated texts (if no CSV path) or dictionary with stats (if CSV path provided)
    """
    # --- Input Validation ---
    if not isinstance(prompts_series, pd.Series):
        raise TypeError("Input 'prompts_series' must be a pandas Series.")
    num_prompts = len(prompts_series)

    if id_series is not None:
        if not isinstance(id_series, pd.Series):
            raise TypeError("Input 'id_series' must be a pandas Series if provided.")
        if len(id_series) != num_prompts:
            raise ValueError("Inputs 'id_series' and 'prompts_series' must have the same length.")

    fixed_headers = ["id", "prompt", "generated_text", "status"]
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
            if len(col_values) != num_prompts:
                raise ValueError(
                    f"Length of data for additional column '{col_name}' ({len(col_values)}) "
                    f"does not match length of 'prompts_series' ({num_prompts})."
                )
            ordered_additional_column_names.append(col_name) # Store in a fixed order

    # --- Initialization ---
    stats = {"processed_successfully": 0, "failed_generation_or_processing": 0, "skipped_invalid_prompt": 0}
    if num_prompts == 0:
        # Still write header if CSV path is given for an empty operation
        if output_csv_path:
            try:
                with gzip.open(output_csv_path, 'wt', newline='', encoding='utf-8') as f_header:
                    csv.writer(f_header).writerow(fixed_headers + ordered_additional_column_names)
            except IOError as e:
                 print(f"Warning: Could not write CSV header for empty series to {output_csv_path}: {e}")
        return [] if output_csv_path is None else {**stats, "output_path": output_csv_path, "total_prompts_in_series": num_prompts}

    all_generations_placeholders: Optional[List[Optional[str]]] = None
    csv_writer_lock: Optional[threading.Lock] = None
    final_csv_header = fixed_headers + ordered_additional_column_names

    if output_csv_path:
        print(f"Starting text generation for {num_prompts} prompts. Results will be saved to {output_csv_path}.")
        csv_writer_lock = threading.Lock()
        try:
            with gzip.open(output_csv_path, 'wt', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(final_csv_header)
        except IOError as e:
            print(f"Error: Could not write CSV header to {output_csv_path}: {e}")
            raise
    else:
        all_generations_placeholders = [None] * num_prompts
        print(f"Starting text generation for {num_prompts} prompts. Results will be stored in memory.")
    
    print(f"Using up to {max_workers} concurrent API calls...")

    # --- Prepare Tasks ---
    generation_tasks = _prepare_generation_tasks(
        prompts_series, id_series, system_prompt, max_tokens, temperature, additional_data, 
        ordered_additional_column_names, output_csv_path, csv_writer_lock, stats, all_generations_placeholders
    )
    
    # --- Execute Tasks Concurrently ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_payload = {
            executor.submit(
                generate_text, 
                task_payload["prompt"], 
                task_payload["system_prompt"],
                task_payload["max_tokens"],
                task_payload["temperature"]
            ): task_payload
            for task_payload in generation_tasks # Each task is for one prompt
        }

        for future in concurrent.futures.as_completed(future_to_task_payload):
            task_payload = future_to_task_payload[future]
            try:
                generation_result = future.result() # This is Optional[str]
                _process_completed_task(
                    task_payload, generation_result, False, # is_exception = False
                    output_csv_path, csv_writer_lock, all_generations_placeholders, stats
                )
            except Exception as exc:
                print(f"Task for prompt ID {task_payload['id']} generated an exception: {exc}")
                _process_completed_task(
                    task_payload, None, True, # generation_result = None, is_exception = True
                    output_csv_path, csv_writer_lock, all_generations_placeholders, stats
                )
            
    print(f"Text generation finished.")
    if output_csv_path:
        summary = {**stats, "output_path": output_csv_path, "total_prompts_in_series": num_prompts}
        print(f"Summary: {summary}")
        return summary
    else:
        if all_generations_placeholders and len(all_generations_placeholders) != num_prompts:
            print(f"Critical Warning: Final number of in-memory results ({len(all_generations_placeholders)}) does not match input prompts ({num_prompts}).")
        print(f"Total generations/placeholders generated in memory: {len(all_generations_placeholders) if all_generations_placeholders else 0}")
        return all_generations_placeholders if all_generations_placeholders is not None else []

# Example usage
if __name__ == "__main__":
    # Example with pandas Series
    import pandas as pd
    
    # Create sample prompts
    sample_prompts = pd.Series([
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "Create a recipe for chocolate cake.",
        "Write a haiku about spring flowers."
    ])
    
    # Generate text (in-memory)
    results = generate_text_from_series(
        prompts_series=sample_prompts,
        system_prompt="You are a helpful assistant.",
        max_tokens=500,
        temperature=0.7,
        max_workers=2
    )
    
    print("\nGenerated texts:")
    for i, result in enumerate(results):
        if result:
            print(f"\nPrompt {i+1}: {sample_prompts.iloc[i][:50]}...")
            print(f"Generated: {result[:100]}...")
        else:
            print(f"\nPrompt {i+1}: Failed to generate")
    
    # Generate text (save to CSV)
    # results_csv = generate_text_from_series(
    #     prompts_series=sample_prompts,
    #     system_prompt="You are a helpful assistant.",
    #     max_tokens=500,
    #     temperature=0.7,
    #     max_workers=2,
    #     output_csv_path="generated_texts.csv.gz"
    # )
    # print(f"\nCSV Results: {results_csv}")