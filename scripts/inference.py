"""
Interactive inference script for the fine-tuned coding agent.
Supports 4-bit quantization and multi-turn responses.
"""

import os
import sys
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    RED = '\033[91m'

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Inference for SFT Coding Agent.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained LoRA adapter.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B", help="Base model name or path.")
    parser.add_argument("--prompt", type=str, help="Single-shot mode: Provide prompt, get answer, exit.")
    parser.add_argument("--compare-base", action="store_true", help="Show base model response side-by-side.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens to generate.")
    return parser.parse_args()

def load_models(base_model_path: str, peft_model_path: str, compare: bool):
    logger.info("Initializing 4-bit quantization config")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model {base_model_path} with 4-bit quantization")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Loading fine-tuned adapter from {peft_model_path}")
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    return base_model if compare else None, ft_model, tokenizer

def generate_response(model, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=(args.temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def print_colored(text: str):
    """Pretty-prints output with color coding for reasoning and tool calls."""
    lines = text.split('\n')
    in_json = False
    for line in lines:
        if line.startswith('```json'):
            in_json = True
            print(f"{Colors.GREEN}{line}")
        elif in_json and line.startswith('```'):
            in_json = False
            print(f"{line}{Colors.RESET}")
        elif in_json or '"tool"' in line:
            print(f"{Colors.GREEN}{line}{Colors.RESET}")
        elif line.startswith('Thought:') or line.startswith('Reasoning:') or 'thinking' in line.lower():
            print(f"{Colors.BLUE}{line}{Colors.RESET}")
        elif line.startswith('Result:') or line.startswith('Output:'):
            print(f"{Colors.YELLOW}{line}{Colors.RESET}")
        else:
            print(line)

def main():
    args = parse_args()
    
    try:
        base_model, ft_model, tokenizer = load_models(args.model, args.checkpoint, args.compare_base)
        
        # Single-shot mode
        if args.prompt:
            print(f"\n{Colors.YELLOW}Prompt:{Colors.RESET} {args.prompt}\n")
            if args.compare_base:
                print(f"{Colors.RED}=== Base Model ==={Colors.RESET}")
                base_resp = generate_response(base_model, tokenizer, args.prompt, args)
                print(base_resp)
                print()
            
            print(f"{Colors.BLUE}=== Fine-Tuned Model ==={Colors.RESET}")
            ft_resp = generate_response(ft_model, tokenizer, args.prompt, args)
            print_colored(ft_resp)
            return

        # Interactive REPL mode
        print(f"{Colors.GREEN}Interactive Agent REPL started. Type 'exit' to quit.{Colors.RESET}")
        chat_history = ""
        
        while True:
            try:
                user_input = input(f"\n{Colors.YELLOW}User >{Colors.RESET} ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                    
                # Append to history for multi-turn
                chat_history += f"User: {user_input}\nAssistant: "
                    
                if args.compare_base:
                    print(f"\n{Colors.RED}=== Base Model ==={Colors.RESET}")
                    base_resp = generate_response(base_model, tokenizer, chat_history, args)
                    print(base_resp)
                
                print(f"\n{Colors.BLUE}=== Fine-Tuned Model ==={Colors.RESET}")
                ft_resp = generate_response(ft_model, tokenizer, chat_history, args)
                print_colored(ft_resp)
                
                # Update history with generated response
                chat_history += ft_resp + "\n"
                
            except KeyboardInterrupt:
                print("\nExiting REPL.")
                break
                    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
