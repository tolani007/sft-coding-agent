#!/usr/bin/env python3
"""
push_dataset.py - Push local processed JSONL data to Hugging Face Hub

Google Colab runs in the cloud and cannot directly access your Mac's hard drive.
To train on Colab, we first upload our 6,600+ compiled examples to the Hugging Face Hub.
"""
import os
from datasets import load_dataset
from pathlib import Path

# Load HF token
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ.setdefault("HF_TOKEN", line.split("=", 1)[1].strip())

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment or .env file")

# Target dataset repository
USERNAME = "focustiki"
REPO_NAME = f"{USERNAME}/sft-coding-agent-traces"

print("=" * 60)
print(f"PUSHING DATASET TO HUGGING FACE: {REPO_NAME}")
print("=" * 60)

# Load the merged dataset from local JSONL files
print("Loading local JSONL files...")
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "test": "data/eval.jsonl"
    }
)

print(f"Loaded:")
print(f"  Train: {len(dataset['train'])} examples")
print(f"  Test:  {len(dataset['test'])} examples")

print(f"\nUploading to https://huggingface.co/datasets/{REPO_NAME} ...")
dataset.push_to_hub(REPO_NAME, token=HF_TOKEN, private=False)

print("\n Dataset successfully pushed!")
print(f"You can now load it in Colab using: load_dataset('{REPO_NAME}')")
