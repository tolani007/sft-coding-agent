import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert coding agent with access to the following tools:

- **bash**: Execute shell commands. Usage: {"name": "bash", "arguments": {"command": "<shell command>"}}
- **read**: Read file contents. Usage: {"name": "read", "arguments": {"path": "<file path>"}}
- **write**: Write content to a file. Usage: {"name": "write", "arguments": {"path": "<file path>", "content": "<file content>"}}
- **edit**: Edit a file with search/replace. Usage: {"name": "edit", "arguments": {"path": "<file path>", "search": "<text to find>", "replace": "<replacement text>"}}

Before taking action, reason step-by-step inside <think>...</think> tags. Then call the appropriate tool. After receiving tool results, continue reasoning and acting until the task is complete."""

# Basic regex for catching potential API keys/secrets to redact
API_KEY_REGEX = re.compile(r'(?i)(api[_-]?key|secret|token|password)[\s:=]+[\'"]?([a-zA-Z0-9\-_]{16,})[\'"]?')

def redact_secrets(text: str) -> str:
    """Check for residual API key patterns and redact them."""
    if not isinstance(text, str):
        return text
    # Replace matches with [REDACTED]
    return API_KEY_REGEX.sub(r'\1 = [REDACTED]', text)

def estimate_tokens(text: str) -> int:
    """Very rough estimation of tokens based on word count."""
    if not isinstance(text, str):
        return 0
    # ~1.3 tokens per word is a common heuristic
    return int(len(text.split()) * 1.3)

def parse_pi_mono_session(session_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse Pi-Mono event streams into standard SFT format.
    
    Expects a list of events belonging to a single session.
    Extracts messages, role mapping, tool calls, and <think> blocks.
    """
    messages = []
    # Inject system prompt
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    # Simple state machine for parsing events
    # This assumes events are roughly ordered
    for event in session_events:
        event_type = event.get("type", "")
        
        if event_type == "message" or "role" in event:
            role = event.get("role", "")
            content = event.get("content", "")
            content = redact_secrets(content)
            
            if role in ["user", "system"]:
                messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "assistant":
                # Look for tool calls and think blocks
                tool_calls = event.get("tool_calls", [])
                think_content = event.get("reasoning", "")
                
                # If think content not explicitly provided, try to extract from content
                if not think_content and "<think>" in content:
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if think_match:
                        think_content = think_match.group(1).strip()
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                formatted_content = ""
                if think_content:
                    formatted_content += f"<think>\n{think_content}\n</think>\n"
                if content:
                    formatted_content += content
                    
                msg = {
                    "role": "assistant",
                    "content": formatted_content.strip()
                }
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
                
            elif role in ["tool", "toolResult", "bashExecution"]:
                # Map various tool result formats to standard "tool" role
                tool_call_id = event.get("tool_call_id", event.get("id", "unknown"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content
                })
    
    return messages

def process_file(file_path: Path, min_turns: int, max_tokens: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Process a raw JSONL file and return formatted sessions and statistics."""
    formatted_sessions = []
    stats = {
        "total": 0,
        "filtered_short": 0,
        "filtered_long": 0,
        "kept": 0,
        "tokens": []
    }
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return formatted_sessions, stats
        
    logger.info(f"Processing file: {file_path}")
    
    # Read line by line
    with open(file_path, "r", encoding="utf-8") as f:
        # Some files might be one session per line, others might be event streams
        # Here we assume each line is a complete session dictionary (like in pi-mono dataset)
        for line in f:
            if not line.strip():
                continue
                
            stats["total"] += 1
            try:
                data = json.loads(line)
                
                # Different sources have different structures
                if "messages" in data:
                    # Already somewhat formatted
                    session_events = data["messages"]
                elif "events" in data:
                    session_events = data["events"]
                elif isinstance(data, list):
                    session_events = data
                else:
                    # Maybe it's a flat structure
                    session_events = [data]
                
                messages = parse_pi_mono_session(session_events)
                
                # Check turns (user + assistant exchanges)
                user_msgs = sum(1 for m in messages if m["role"] == "user")
                if user_msgs < min_turns:
                    # Check for failure recovery (tool errors + corrective reasoning)
                    # Simple heuristic: look for "error" in tool results
                    has_error = any("error" in str(m.get("content", "")).lower() for m in messages if m["role"] == "tool")
                    if not has_error:
                        stats["filtered_short"] += 1
                        continue
                        
                # Check tokens
                session_text = " ".join([str(m.get("content", "")) for m in messages])
                session_tokens = estimate_tokens(session_text)
                
                if session_tokens > max_tokens:
                    stats["filtered_long"] += 1
                    continue
                    
                stats["kept"] += 1
                stats["tokens"].append(session_tokens)
                
                formatted_sessions.append({"messages": messages})
                
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON line.")
            except Exception as e:
                logger.warning(f"Error processing session: {e}")
                
    return formatted_sessions, stats

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw traces into SFT-ready format.")
    parser.add_argument("--input-dir", type=str, default="./data/raw/",
                        help="Input directory with raw data files.")
    parser.add_argument("--output-dir", type=str, default="./data/",
                        help="Output directory for processed train/eval files.")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Fraction of data to use for training (default: 0.9).")
    parser.add_argument("--min-turns", type=int, default=3,
                        help="Minimum number of user turns per session (default: 3).")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Maximum estimated tokens per session (default: 32768).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to additional config file.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting.")
    args = parser.parse_args()

    random.seed(args.seed)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
        
    all_sessions = []
    total_stats = {
        "total": 0, "filtered_short": 0, "filtered_long": 0, "kept": 0, "tokens": []
    }
    
    for file_path in input_dir.glob("*.jsonl"):
        sessions, stats = process_file(file_path, args.min_turns, args.max_tokens)
        all_sessions.extend(sessions)
        
        for k in ["total", "filtered_short", "filtered_long", "kept"]:
            total_stats[k] += stats[k]
        total_stats["tokens"].extend(stats["tokens"])
        
    if not all_sessions:
        logger.warning("No valid sessions found to process.")
        return
        
    # Shuffle and split
    random.shuffle(all_sessions)
    split_idx = int(len(all_sessions) * args.train_split)
    train_sessions = all_sessions[:split_idx]
    eval_sessions = all_sessions[split_idx:]
    
    # Save files
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    
    logger.info(f"Saving {len(train_sessions)} sessions to {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_sessions:
            f.write(json.dumps(s) + "\n")
            
    logger.info(f"Saving {len(eval_sessions)} sessions to {eval_path}")
    with open(eval_path, "w", encoding="utf-8") as f:
        for s in eval_sessions:
            f.write(json.dumps(s) + "\n")
            
    # Print statistics
    logger.info("=== Preprocessing Statistics ===")
    logger.info(f"Total raw sessions processed: {total_stats['total']}")
    logger.info(f"Filtered (too short, < {args.min_turns} turns): {total_stats['filtered_short']}")
    logger.info(f"Filtered (too long, > {args.max_tokens} tokens): {total_stats['filtered_long']}")
    logger.info(f"Total kept sessions: {total_stats['kept']}")
    
    if total_stats["tokens"]:
        avg_tokens = sum(total_stats["tokens"]) / len(total_stats["tokens"])
        logger.info(f"Average tokens per kept session: {avg_tokens:.2f}")
    
    logger.info(f"Train set: {len(train_sessions)} sessions")
    logger.info(f"Eval set: {len(eval_sessions)} sessions")
    logger.info("================================")

if __name__ == "__main__":
    main()
