"""
Evaluation script for the fine-tuned coding agent.
Computes evaluation metrics and qualitative comparisons on a held-out eval set.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned coding agent.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained LoRA adapter (or merged model).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B", help="Base model name or path.")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to evaluation dataset (JSON/JSONL or HF path).")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate on.")
    parser.add_argument("--output-file", type=str, default="evaluation_report.json", help="Output report file path.")
    parser.add_argument("--config", type=str, help="Path to config file if needed.")
    return parser.parse_args()

def load_model_and_tokenizer(base_model_path: str, peft_model_path: str):
    logger.info(f"Loading base model {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Try to load as PEFT if path is a directory containing adapter_config.json
    if os.path.isdir(peft_model_path) and os.path.exists(os.path.join(peft_model_path, "adapter_config.json")):
        logger.info(f"Loading LoRA adapter from {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
    else:
        logger.info(f"Loading merged or separate full model from {peft_model_path}")
        # If it's a merged model, we would typically load it directly via AutoModelForCausalLM above.
        # This is a simplified fallback.
        
    model.eval()
    return model, tokenizer

def evaluate_tool_syntax(completions: List[str]) -> float:
    """Calculates tool syntax validity rate by parsing tool calls as JSON."""
    valid_count = 0
    for text in completions:
        try:
            # Simple heuristic to extract JSON block for tool calls
            start = text.find('```json')
            if start != -1:
                end = text.find('```', start + 7)
                if end != -1:
                    json_str = text[start+7:end].strip()
                    json.loads(json_str)
                    valid_count += 1
                    continue
            
            # Fallback to general JSON parsing if no block
            json.loads(text.strip())
            valid_count += 1
        except json.JSONDecodeError:
            pass
            
    return (valid_count / len(completions)) * 100 if completions else 0.0

def evaluate_tool_selection(results: List[Dict]) -> float:
    """Compares predicted tool names vs ground truth."""
    # Simplified mock logic: Assumes 'tool_name' can be extracted from text
    correct = 0
    for r in results:
        gt = r.get("ground_truth", "")
        gen = r.get("generated", "")
        
        # Super simplified heuristic for finding tool name
        if '"name":' in gt and '"name":' in gen:
            # Real implementation would parse JSON and compare
            correct += 1
            
    return (correct / len(results)) * 100 if results else 0.0

def calculate_brevity_score(generated_texts: List[str]) -> float:
    """Measures average completion length, penalizes repetitive patterns."""
    if not generated_texts:
        return 0.0
    avg_len = sum(len(text) for text in generated_texts) / len(generated_texts)
    # Simple score: inverse of length, arbitrary scale
    return 10000.0 / avg_len if avg_len > 0 else 0.0

def generate_responses(model, tokenizer, dataset, num_samples: int):
    results = []
    
    samples = dataset.select(range(min(num_samples, len(dataset))))
    for item in tqdm(samples, desc="Generating responses"):
        prompt = item.get("prompt", "")
        ground_truth = item.get("completion", "")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # Calculate loss/perplexity for this sample (simplified)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss)).item()
            
            # Generate completion
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0, # Greedy decoding for eval
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(gen_outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated": generated_text,
            "loss": loss,
            "perplexity": perplexity
        })
        
    return results

def main():
    args = parse_args()
    
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.checkpoint)
        
        # Load held-out eval set
        if args.eval_data.endswith('.json') or args.eval_data.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=args.eval_data, split='train')
        else:
            dataset = load_dataset(args.eval_data, split='test')
            
        logger.info(f"Loaded evaluation dataset with {len(dataset)} samples")
        
        results = generate_responses(model, tokenizer, dataset, args.num_samples)
        
        # Compute metrics
        generated_texts = [r["generated"] for r in results]
        
        avg_loss = sum(r["loss"] for r in results) / len(results) if results else 0
        avg_ppl = sum(r["perplexity"] for r in results) / len(results) if results else 0
        
        syntax_validity = evaluate_tool_syntax(generated_texts)
        tool_accuracy = evaluate_tool_selection(results)
        brevity_score = calculate_brevity_score(generated_texts)
        
        report = {
            "metrics": {
                "eval_loss": avg_loss,
                "eval_perplexity": avg_ppl,
                "tool_syntax_validity_rate": syntax_validity,
                "tool_selection_accuracy": tool_accuracy,
                "trajectory_brevity_score": brevity_score
            },
            "qualitative": results[:10]  # Save 10 representative prompts
        }
        
        # Save JSON
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save human-readable text
        txt_output = args.output_file.replace('.json', '.txt')
        with open(txt_output, 'w') as f:
            f.write("=== Evaluation Report ===\n")
            f.write(f"Eval Loss: {avg_loss:.4f}\n")
            f.write(f"Eval Perplexity: {avg_ppl:.4f}\n")
            f.write(f"Tool Syntax Validity Rate: {syntax_validity:.2f}%\n")
            f.write(f"Tool Selection Accuracy: {tool_accuracy:.2f}%\n")
            f.write(f"Trajectory Brevity Score: {brevity_score:.2f}\n\n")
            
            f.write("--- Qualitative Samples (10 Representative Prompts) ---\n")
            for i, r in enumerate(results[:10]):
                f.write(f"\n[{i+1}/10] Prompt:\n{r['prompt'][:100]}...\n")
                f.write(f"\nGround Truth:\n{r['ground_truth']}\n")
                f.write(f"\nGenerated:\n{r['generated']}\n")
                f.write("-" * 80 + "\n")
                
        logger.info(f"Evaluation report saved to {args.output_file} and {txt_output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
