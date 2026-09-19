#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys

import yaml
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

# Use trackio as wandb for experiment tracking
try:
    import trackio as wandb
except ImportError:
    logging.warning("trackio not found, using a dummy tracker.")
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

def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Pipeline for Coding Agents")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Model ID on HuggingFace")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--max-samples", type=int, help="Limit number of training samples for debugging")
    parser.add_argument("--push-to-hub", action="store_true", help="Push final model to Hub")
    parser.add_argument("--hub-model-id", type=str, help="Repository ID for Hub")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file {args.config} not found, relying on defaults/cli args.")

    model_id = args.model or config.get("model_id", "Qwen/Qwen1.5-7B")
    output_dir = args.output_dir or config.get("output_dir", "./output")
    
    # Initialize experiment tracking
    wandb.init(project="sft-coding-agent", config={**config, "model_id": model_id})

    # Tokenizer Setup
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DataCollator Setup (Completion-only masking)
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

    # 4-bit QLoRA Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Model Loading
    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Handle Gemma 4 multimodal model architecture if necessary
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
    logger.info("Loading datasets...")
    train_file = config.get("train_data", "data/train.jsonl")
    eval_file = config.get("eval_data", "data/eval.jsonl")
    
    dataset_files = {}
    if os.path.exists(train_file):
        dataset_files["train"] = train_file
    if os.path.exists(eval_file):
        dataset_files["test"] = eval_file
        
    if not dataset_files:
        logger.error(f"Datasets not found at {train_file} or {eval_file}")
        raise ValueError("No train or eval JSONL files found.")

    dataset = load_dataset("json", data_files=dataset_files)
    train_dataset = dataset["train"]
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
    eval_dataset = dataset.get("test")
    if eval_dataset and args.max_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))

    # Dataset preprocessing for Chat format if plain text is not available
    if "text" not in train_dataset.column_names:
        def apply_template(batch):
            texts = []
            for msgs in batch.get("messages", []):
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            return {"text": texts}
        train_dataset = train_dataset.map(apply_template, batched=True)
        if eval_dataset:
            eval_dataset = eval_dataset.map(apply_template, batched=True)

    # SFT Training Configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_seq_length=config.get("max_seq_length", 4096),
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        dataset_text_field="text",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=sft_config,
        callbacks=[TrackioCallback()]
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    logger.info("Training complete. Saving adapter...")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

    # Push to Hub
    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing model and tokenizer to HuggingFace Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)

    wandb.finish()
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
