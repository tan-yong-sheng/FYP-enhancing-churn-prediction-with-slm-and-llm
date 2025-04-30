'''
Utility script to synchronize a local data folder (e.g., `data/`) with a Hugging Face Dataset repository.

**Rationale:**
Standard Git hosting platforms like GitHub are not ideal for versioning large data files due to storage
and bandwidth limitations (especially with Git LFS on free tiers). Hugging Face Hub, particularly its
Dataset repositories, is specifically designed to handle large files efficiently.

This script provides command-line functions to easily upload local data to a Hugging Face Dataset repo
or download the data from the repo back to the local machine, keeping the main project's Git repository
(e.g., on GitHub) focused on code while the data resides on Hugging Face.

**Prerequisites:**
1. Install huggingface_hub: `pip install huggingface_hub`

**Example Usage:**

Assuming your HF Dataset repo is `tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm`:

- Upload local `data/` folder:
  `python utils/hf_data_sync.py upload --repo-id tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm --local-path data --verbose`

- Download from repo to local `data/` folder:
  `python utils/hf_data_sync.py download --repo-id tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm --local-path data --verbose`

- Use `--verbose` or `-v` for detailed progress output.
'''

import os
import argparse
from huggingface_hub import HfApi, snapshot_download
import logging
from huggingface_hub import login
from huggingface_hub.utils import logging as hf_logging

from dotenv import load_dotenv, find_dotenv

# Set default logging level (can be overridden by verbose flag)
hf_logging.set_verbosity_warning()
# Also configure root logger slightly for better formatting if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


_ = load_dotenv(find_dotenv())

# Automatically login with token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN not set. Upload/download may fail if not authenticated.")


def upload_data(local_path: str, repo_id: str, repo_type: str = "dataset"):
    """
    Uploads the contents of a local directory to a Hugging Face repository.

    Args:
        local_path (str): Path to the local directory to upload.
        repo_id (str): The Hugging Face repository ID (e.g., "username/repo-name").
        repo_type (str): The type of repository ('dataset', 'model', 'space'). Defaults to 'dataset'.
    """
    if not os.path.isdir(local_path):
        print(f"Error: Local path '{local_path}' does not exist or is not a directory.")
        return

    print(f"Attempting to upload contents of '{local_path}' to {repo_type} repo '{repo_id}' using upload_large_folder...")
    try:
        # Instantiate the HfApi client
        api = HfApi()

        # Use upload_large_folder for better handling of numerous files
        # Note: This function might have slightly different behavior or options
        # compared to upload_folder. We'll use the core parameters.
        # It might upload in chunks and handle retries automatically.
        api.upload_large_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            # commit_message parameter might not be directly available or work the same way.
            # We can add a default commit message later if needed, possibly via other API calls or parameters.
            # delete_patterns=["*"], # Check documentation if needed for equivalent functionality
            allow_patterns=["**"], # Ensure all files are considered
            ignore_patterns=None # Explicitly don't ignore anything within the folder
            # run_as_future=False # Set to False if you want the call to block until completion
        )
        print(f"Successfully initiated upload of '{local_path}' to '{repo_id}'. Check Hugging Face Hub for progress/completion.")
    except Exception as e:
        print(f"Error uploading data: {e}")

def download_data(local_path: str, repo_id: str, repo_type: str = "dataset"):
    """
    Downloads the contents of a Hugging Face repository to a local directory.

    Args:
        local_path (str): Path to the local directory to download into.
        repo_id (str): The Hugging Face repository ID (e.g., "username/repo-name").
        repo_type (str): The type of repository ('dataset', 'model', 'space'). Defaults to 'dataset'.
    """
    print(f"Attempting to download contents of {repo_type} repo '{repo_id}' to '{local_path}'...")
    try:
        # Ensure the target local directory exists
        os.makedirs(local_path, exist_ok=True)

        # snapshot_download downloads the repo content and returns the path to the cached version.
        # We want to download *into* our specified local_path, so we use local_dir.
        # Setting local_dir_use_symlinks=False ensures files are copied, not symlinked (safer on Windows).
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_path,
            local_dir_use_symlinks=False, # Use False for copying, True for symlinks (if supported)
            force_download=True, # Ensure fresh download
            # Consider adding ignore_patterns if you want to exclude specific files/folders
        )
        print(f"Successfully downloaded '{repo_id}' to '{local_path}'.")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload or download data to/from Hugging Face Hub.")
    parser.add_argument("action", choices=["upload", "download"], help="Action to perform: 'upload' or 'download'.")
    parser.add_argument("--local-path", default="data", help="Path to the local data directory (default: 'data').")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID (e.g., 'username/dataset-name').")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"], help="Type of Hugging Face repository (default: 'dataset').")
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity to show detailed progress.")

    args = parser.parse_args()

    # Set verbosity based on the flag
    if args.verbose:
        print("Verbose mode enabled. Setting logging level to INFO.")
        hf_logging.set_verbosity_info()
    else:
        hf_logging.set_verbosity_warning()


    # Ensure you are logged in: run `huggingface-cli login` in your terminal first.
    # You might need to install the library: pip install huggingface_hub

    if args.action == "upload":
        upload_data(local_path=args.local_path, repo_id=args.repo_id, repo_type=args.repo_type)
    elif args.action == "download":
        download_data(local_path=args.local_path, repo_id=args.repo_id, repo_type=args.repo_type)
