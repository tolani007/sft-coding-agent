import argparse
import logging
import os
import sys
from pathlib import Path
import json

from datasets import load_dataset
from huggingface_hub import HfFileSystem, login

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def download_bucket(output_dir: Path):
    """Download data from the primary source: HuggingFace bucket."""
    logger.info("Downloading from HuggingFace bucket: hf://buckets/burtenshaw/sft-on-traces/example.jsonl")
    try:
        fs = HfFileSystem()
        source_path = "hf://buckets/burtenshaw/sft-on-traces/example.jsonl"
        dest_path = output_dir / "bucket_example.jsonl"
        
        if dest_path.exists():
            logger.info(f"File {dest_path} already exists. Skipping download.")
            return

        logger.info("Fetching bucket data...")
        # fsspec get handles streaming and some resume logic internally
        fs.get(source_path, str(dest_path))
        logger.info(f"Successfully downloaded to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download from bucket: {e}")

def download_dataset(output_dir: Path):
    """Download data from the secondary source: badlogicgames/pi-mono dataset."""
    logger.info("Downloading from HuggingFace dataset: badlogicgames/pi-mono")
    dest_path = output_dir / "dataset_pi_mono.jsonl"
    
    if dest_path.exists():
        logger.info(f"File {dest_path} already exists. Skipping download.")
        return

    try:
        dataset = load_dataset("badlogicgames/pi-mono", split="train")
        logger.info(f"Saving dataset to {dest_path}...")
        dataset.to_json(dest_path, orient="records", lines=True)
        logger.info(f"Successfully saved {len(dataset)} records to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset badlogicgames/pi-mono: {e}")

def download_chat(output_dir: Path):
    """Download data from the tertiary source: sergiopaniego/pi-mono-chat dataset."""
    logger.info("Downloading from HuggingFace dataset: sergiopaniego/pi-mono-chat")
    dest_path = output_dir / "chat_pi_mono.jsonl"
    
    if dest_path.exists():
        logger.info(f"File {dest_path} already exists. Skipping download.")
        return

    try:
        dataset = load_dataset("sergiopaniego/pi-mono-chat", split="train")
        logger.info(f"Saving dataset to {dest_path}...")
        dataset.to_json(dest_path, orient="records", lines=True)
        logger.info(f"Successfully saved {len(dataset)} records to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset sergiopaniego/pi-mono-chat: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download agent session traces from multiple sources.")
    parser.add_argument("--source", type=str, choices=["bucket", "dataset", "chat", "all"], default="all",
                        help="Data source to download from. Default: all")
    parser.add_argument("--output-dir", type=str, default="./data/raw/",
                        help="Output directory to save raw data. Default: ./data/raw/")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token for authentication (optional).")
    args = parser.parse_args()

    if args.token:
        logger.info("Logging into HuggingFace Hub...")
        login(token=args.token)
    elif "HF_TOKEN" in os.environ:
        logger.info("Logging into HuggingFace Hub using HF_TOKEN environment variable...")
        login(token=os.environ["HF_TOKEN"])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.source in ["bucket", "all"]:
        download_bucket(output_dir)
    
    if args.source in ["dataset", "all"]:
        download_dataset(output_dir)
        
    if args.source in ["chat", "all"]:
        download_chat(output_dir)
        
    logger.info("Download process completed.")

if __name__ == "__main__":
    main()
