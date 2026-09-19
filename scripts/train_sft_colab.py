#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import subprocess

# Auto-install dependencies in Colab/Kaggle environments
def install_dependencies():
    try:
        import trl
        import peft
        import datasets
        import bitsandbytes
    except ImportError:
        print("Installing required packages for training...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers", "trl", "peft", "datasets", "bitsandbytes", "accelerate", "pyyaml"
        ])

install_dependencies()

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Handle local-first trackio or fallback
try:
    import trackio as wandb
except ImportError:
    class DummyWandb:
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    wandb = DummyWandb()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TrackioCallback(TrainerCallback):
    """Custom callback to log metrics to trackio."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)

def get_gpu_type_and_params():
    """Auto-detects GPU type and adjusts hyperparameters for Colab/Kaggle environments."""
    if not torch.cuda.is_available():
        logger.warning("No GPU found! Fallback to conservative CPU/small params.")
        return "Unknown", 1, 1024, 1
    
    device_name = torch.cuda.get_device_name(0).upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if "T4" in device_name or vram_gb < 20:
        return "T4", 1, 2048, 16
    elif "L4" in device_name or (20 <= vram_gb < 35):
        return "L4", 2, 4096, 8
    elif "A100" in device_name or vram_gb >= 35:
        return "A100", 4, 8192, 4
    return "Unknown", 1, 2048, 16

def parse_args():
    parser = argparse.ArgumentParser(description="Colab/Kaggle-optimized SFT Training Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B", help="Model ID on HuggingFace")
    parser.add_argument("--push-to-hub", action="store_true", help="Push final model to Hub")
    parser.add_argument("--hub-model-id", type=str, help="Repository ID for Hub")
    return parser.parse_args()

def main():
    args = parse_args()

    # Mount Google Drive for checkpointing if running in Google Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        base_dir = "/content/drive/MyDrive/sft-checkpoints/"
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Mounted Google Drive. Output directory: {base_dir}")
    except ImportError:
        logger.info("Not running in Google Colab. Using local output directory.")
        base_dir = "./colab_output"
        os.makedirs(base_dir, exist_ok=True)

    gpu_type, batch_size, max_seq_length, grad_accum = get_gpu_type_and_params()
    logger.info(f"Detected GPU: {gpu_type}. Using batch_size={batch_size}, max_seq_length={max_seq_length}, grad_accum={grad_accum}")

    wandb.init(project="sft-coding-agent", config={"model_id": args.model, "gpu": gpu_type})

    # Setup Tokenizer
    model_id = args.model
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup DataCollator with Completion-only masking
    model_id_lower = model_id.lower()
    if "gemma" in model_id_lower:
        response_template = "<start_of_turn>model\n"
    elif "qwen" in model_id_lower:
        response_template = "<|im_start|>assistant\n"
    else:
        logger.warning("Unknown model family, using fallback response template")
        response_template = "Assistant: "

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # Setup 4-bit QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Target appropriate modules based on architecture
    if hasattr(model, "language_model"):
        target_modules = [
            "language_model.model.layers.*.self_attn.q_proj", 
            "language_model.model.layers.*.self_attn.v_proj"
        ]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Dataset Loading
    logger.info("Loading train dataset...")
    # Attempt to load local fallback, otherwise fetch directly for colab simplicity
    if os.path.exists("data/train.jsonl"):
        train_dataset = load_dataset("json", data_files={"train": "data/train.jsonl"})["train"]
    else:
        logger.warning("data/train.jsonl not found! Using remote fallback dataset from HF.")
        train_dataset = load_dataset("json", data_files={"train": "https://huggingface.co/datasets/badlogicgames/pi-mono/resolve/main/train.jsonl"})["train"]
    
    if "text" not in train_dataset.column_names:
        def apply_template(batch):
            texts = []
            for msgs in batch.get("messages", []):
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            return {"text": texts}
        train_dataset = train_dataset.map(apply_template, batched=True)

    # Configuration for standard Colab/Kaggle free tier run
    sft_config = SFTConfig(
        output_dir=base_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_seq_length=max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3, # Keep Colab drive clean
        dataset_text_field="text",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=sft_config,
        callbacks=[TrackioCallback()]
    )

    # Resume handling to survive Colab disconnects
    checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint")]
    resume_from = os.path.join(base_dir, max(checkpoints, key=lambda x: int(x.split("-")[-1]))) if checkpoints else None
    
    if resume_from:
        logger.info(f"Resuming training from checkpoint: {resume_from}")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info("Training complete. Saving final adapter...")
    trainer.model.save_pretrained(os.path.join(base_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(base_dir, "final_adapter"))

    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing model and tokenizer to HuggingFace Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)

    wandb.finish()
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
