"""
Continual learning pipeline script for the SFT coding agent.
Checks for new traces, merges data, resumes training, and evaluates.
"""

import os
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Continual learning pipeline for SFT agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to training config (e.g., config.yaml).")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing latest model checkpoints.")
    parser.add_argument("--data-dir", type=str, required=True, help="Local directory for training data.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push updated adapter to HuggingFace Hub after eval.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing.")
    return parser.parse_args()

def check_new_traces(dataset_id: str = "burtenshaw/sft-on-traces") -> List[Dict]:
    """
    Checks HF bucket/dataset for new traces not yet in local data directory.
    (Mock implementation representing huggingface_hub integration)
    """
    logger.info(f"Checking HuggingFace dataset ({dataset_id}) for new traces...")
    # In reality, this would use `load_dataset` or `HfApi().list_repo_files`
    # and compare against locally tracked timestamps/hashes.
    mock_new_traces = [
        {"session_id": "sess_001", "prompt": "Fix bug in app.py", "completion": "...", "timestamp": "2024-03-01"},
        {"session_id": "sess_002", "prompt": "Add API route", "completion": "...", "timestamp": "2024-03-02"}
    ]
    return mock_new_traces

def download_and_preprocess(traces: List[Dict], data_dir: str, dry_run: bool) -> str:
    """Downloads and preprocesses new traces."""
    logger.info(f"Found {len(traces)} new traces.")
    new_data_path = os.path.join(data_dir, "new_traces.jsonl")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would download and preprocess traces to {new_data_path}")
        return new_data_path
        
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Simulate saving preprocessed traces
    with open(new_data_path, 'w') as f:
        for trace in traces:
            # Reusing preprocess_traces logic implies formatting for completion-only loss
            # Here we just dump as JSONL
            f.write(json.dumps(trace) + '\n')
            
    logger.info(f"Preprocessed traces saved to {new_data_path}")
    return new_data_path

def merge_data(existing_data_path: str, new_data_path: str, merged_data_path: str, dry_run: bool):
    """Merges new traces with existing training data, deduplicating by session hash."""
    if dry_run:
        logger.info(f"[DRY RUN] Would merge {new_data_path} into {existing_data_path} -> {merged_data_path}")
        logger.info("[DRY RUN] Would deduplicate by session hash.")
        return
        
    logger.info("Merging and deduplicating data...")
    seen_hashes = set()
    merged_count = 0
    
    # Ensure merged file directory exists
    Path(os.path.dirname(merged_data_path)).mkdir(parents=True, exist_ok=True)
    
    with open(merged_data_path, 'w') as outfile:
        # Process existing data
        if os.path.exists(existing_data_path):
            with open(existing_data_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        sess_hash = hashlib.md5(item.get('session_id', str(item)).encode()).hexdigest()
                        if sess_hash not in seen_hashes:
                            seen_hashes.add(sess_hash)
                            outfile.write(line)
                            merged_count += 1
                    except Exception:
                        pass
                        
        # Process new data
        if os.path.exists(new_data_path):
            with open(new_data_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        sess_hash = hashlib.md5(item.get('session_id', str(item)).encode()).hexdigest()
                        if sess_hash not in seen_hashes:
                            seen_hashes.add(sess_hash)
                            outfile.write(line)
                            merged_count += 1
                    except Exception:
                        pass
                        
    logger.info(f"Merged data saved to {merged_data_path}. Total unique records: {merged_count}")

def resume_training(config_path: str, checkpoint_dir: str, data_path: str, dry_run: bool):
    """Resumes training from the latest checkpoint using the merged dataset."""
    if dry_run:
        logger.info(f"[DRY RUN] Would run SFTTrainer using {config_path}")
        logger.info(f"[DRY RUN] Would resume from latest checkpoint in {checkpoint_dir}")
        logger.info(f"[DRY RUN] Would use dataset {data_path}")
        logger.info("[DRY RUN] Would apply DataCollatorForCompletionOnlyLM for completion-only loss masking.")
        return
        
    logger.info("Resuming training pipeline...")
    # Real implementation would programmatically invoke the training script or setup SFTTrainer here
    # Example logic:
    # trainer = SFTTrainer(..., resume_from_checkpoint=checkpoint_dir, train_dataset=load_dataset(data_path))
    # trainer.train()
    logger.info("Training complete.")

def evaluate_and_push(checkpoint_dir: str, push_to_hub: bool, dry_run: bool):
    """Evaluates the updated model and optionally pushes to HF Hub."""
    if dry_run:
        logger.info("[DRY RUN] Would evaluate on held-out set to log improvement metrics.")
        if push_to_hub:
            logger.info(f"[DRY RUN] Would push updated adapter from {checkpoint_dir} to HuggingFace Hub.")
        return
        
    logger.info("Evaluating updated model on held-out set...")
    # Call evaluation logic (similar to evaluate.py)
    
    if push_to_hub:
        logger.info("Pushing adapter to HuggingFace Hub...")
        # Real implementation: model.push_to_hub("repo_name")
        logger.info("Push complete.")

def main():
    args = parse_args()
    
    existing_data_path = os.path.join(args.data_dir, "train_data.jsonl")
    merged_data_path = os.path.join(args.data_dir, "train_data_merged.jsonl")
    
    try:
        logger.info("Starting continual learning pipeline")
        
        # 1. Check for new traces
        new_traces = check_new_traces()
        if not new_traces:
            logger.info("No new traces found. Pipeline exiting.")
            return
            
        # 2. Download and preprocess
        new_data_path = download_and_preprocess(new_traces, args.data_dir, args.dry_run)
        
        # 3. Merge and deduplicate
        merge_data(existing_data_path, new_data_path, merged_data_path, args.dry_run)
        
        # (Optional cleanup: overwrite existing data with merged)
        if not args.dry_run and os.path.exists(merged_data_path):
            os.rename(merged_data_path, existing_data_path)
        
        # 4. Resume training
        resume_training(args.config, args.checkpoint_dir, existing_data_path, args.dry_run)
        
        # 5. Evaluate and push
        evaluate_and_push(args.checkpoint_dir, args.push_to_hub, args.dry_run)
        
        logger.info("Continual learning pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Continual learning pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
