from huggingface_hub import HfApi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def upload_to_huggingface(folder_path: str, repo_id: str, repo_type: str = "dataset") -> None:
    """Uploads a local folder to the Hugging Face Hub.

    Args:
        folder_path (str): Path to the local folder.
        repo_id (str): Hugging Face repository ID.
        repo_type (str, optional): Repository type (default is "dataset").

    Raises:
        Exception: If an error occurs during upload.
    """
    api = HfApi()
    try:
        logging.info(f"Uploading folder '{folder_path}' to Hugging Face repo '{repo_id}' ({repo_type})...")
        api.upload_large_folder(folder_path=folder_path, repo_id=repo_id, repo_type=repo_type)
        logging.info("Upload completed successfully.")
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    print("Starting upload process to Hugging Face Hub...")
    upload_to_huggingface(folder_path="./checkpoints", repo_id="sirbastiano94/RawVesselsWeights")