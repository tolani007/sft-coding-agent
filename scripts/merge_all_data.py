#!/usr/bin/env python3
"""
merge_all_data.py - Combine all downloaded datasets into unified train/eval splits.

This script handles 4 different data sources with different formats and merges
them into a single train.jsonl and eval.jsonl ready for SFTTrainer.

LEARNING INSIGHT - Why combine multiple datasets?
  Each dataset teaches the model different skills:
  - vojtavlas2/pi-agent-traces-sft: Full agent sessions with <think>, tool_calls, errors
  - badlogicgames/pi-mono: Raw real-world coding sessions (needs conversion)
  - sergiopaniego/pi-mono-chat: Clean conversational coding (no tools)
  - burtenshaw/sft-on-traces: Professional Codex-style agentic workflow

  Combining them prevents overfitting to one style and builds a more resilient agent.
"""

import json
import os
import random
import re
import hashlib
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (used when adding to datasets that lack one)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert coding agent with access to the following tools:

- **exec_command**: Execute shell commands in the workspace.
  Usage: {"name": "exec_command", "arguments": {"cmd": "<shell command>", "workdir": "<directory>"}}

- **read_file**: Read the contents of a file.
  Usage: {"name": "read_file", "arguments": {"path": "<file path>"}}

- **write_file**: Write content to a file.
  Usage: {"name": "write_file", "arguments": {"path": "<file path>", "content": "<content>"}}

- **edit_file**: Edit a file using search/replace.
  Usage: {"name": "edit_file", "arguments": {"path": "<path>", "search": "<old text>", "replace": "<new text>"}}

Before taking action, reason step-by-step inside <think>...</think> tags. Then call the appropriate tool.
After receiving tool results, continue reasoning and acting until the task is complete."""

# Privacy patterns
SECRET_PATTERNS = [
    re.compile(r'hf_[a-zA-Z0-9]{30,}'),
    re.compile(r'sk-[a-zA-Z0-9]{30,}'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),
]

def redact(text: str) -> str:
    for p in SECRET_PATTERNS:
        text = p.sub('[REDACTED]', text)
    return text

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            b.get('text', '') or b.get('content', '')
            for b in content if isinstance(b, dict)
        )
    return ''

def example_hash(ex: dict) -> str:
    """Hash an example for deduplication."""
    msgs = ex.get('messages', [])
    key = '|'.join(
        m.get('role', '') + ':' + (m.get('content', '') or '')[:200]
        for m in msgs[:5]
    )
    return hashlib.md5(key.encode()).hexdigest()

def estimate_tokens(messages: list[dict]) -> int:
    total = sum(
        len(m.get('content', '') or '') + len(json.dumps(m.get('tool_calls', [])))
        for m in messages
    )
    return total // 4


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: vojtavlas2/pi-agent-traces-sft (already SFT-ready!)
# ─────────────────────────────────────────────────────────────────────────────
def load_vojtavlas2(path: str, max_tokens: int = 24000) -> list[dict]:
    """Load pre-formatted SFT data. Already has system/user/assistant/tool roles."""
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get('messages', [])
            if len(msgs) < 3:
                skipped += 1
                continue
            # Redact any secrets
            for m in msgs:
                if m.get('content'):
                    m['content'] = redact(m['content'])
            
            est = estimate_tokens(msgs)
            if est > max_tokens:
                skipped += 1
                continue
            
            examples.append({"messages": msgs, "source": "vojtavlas2"})
    
    print(f"  vojtavlas2: {len(examples)} loaded, {skipped} skipped")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: badlogicgames/pi-mono (raw Pi events → chat format)
# ─────────────────────────────────────────────────────────────────────────────
def convert_pi_mono_session(events: list[dict]) -> dict | None:
    """Convert native Pi-mono events to chat messages."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for event in events:
        etype = event.get('type')
        
        if etype == 'message':
            msg = event.get('message', {})
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            content = extract_text(msg.get('content', ''))
            content = redact(content.strip())
            
            if not content:
                continue
            
            if role == 'user':
                messages.append({"role": "user", "content": content})
            elif role == 'assistant':
                # Check for tool_use blocks
                tool_calls = []
                text_parts = []
                raw_content = msg.get('content', [])
                if isinstance(raw_content, list):
                    for block in raw_content:
                        if isinstance(block, dict):
                            if block.get('type') == 'tool_use':
                                tool_calls.append({
                                    "id": block.get('id', ''),
                                    "type": "function",
                                    "function": {
                                        "name": block.get('name', ''),
                                        "arguments": json.dumps(block.get('input', {}))
                                    }
                                })
                            elif block.get('type') in ('text', 'thinking'):
                                t = block.get('text', '')
                                if block.get('type') == 'thinking' and t:
                                    text_parts.append(f"<think>{t}</think>")
                                elif t:
                                    text_parts.append(t)
                
                msg_obj = {"role": "assistant", "content": redact('\n'.join(text_parts))}
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                messages.append(msg_obj)
            
            elif role == 'tool':
                tool_call_id = msg.get('tool_use_id', '') or event.get('id', '')
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": redact(content)
                })
    
    # Filter: need at least user + assistant
    turns = [m for m in messages if m['role'] != 'system']
    has_user = any(m['role'] == 'user' for m in turns)
    has_assistant = any(m['role'] == 'assistant' for m in turns)
    
    if not (has_user and has_assistant) or len(turns) < 3:
        return None
    
    return {"messages": messages, "source": "pi-mono"}


def load_pi_mono(directory: str, max_tokens: int = 24000) -> list[dict]:
    """Load and convert all Pi-mono raw JSONL files."""
    examples = []
    skipped = 0
    pi_dir = Path(directory)
    
    if not pi_dir.exists():
        print(f"  pi-mono: directory not found at {directory}")
        return []
    
    files = sorted(pi_dir.glob("*.jsonl"))
    for fpath in files:
        events = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        chat = convert_pi_mono_session(events)
        if chat is None:
            skipped += 1
            continue
        
        est = estimate_tokens(chat['messages'])
        if est > max_tokens:
            skipped += 1
            continue
        
        examples.append(chat)
    
    print(f"  pi-mono: {len(examples)} loaded from {len(files)} files, {skipped} skipped")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: sergiopaniego/pi-mono-chat (pre-formatted, no tools)
# ─────────────────────────────────────────────────────────────────────────────
def load_pi_mono_chat(path: str) -> list[dict]:
    """Load pre-formatted chat pairs. Add system prompt if missing."""
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get('messages', [])
            if len(msgs) < 2:
                skipped += 1
                continue
            
            # Ensure system prompt exists
            if msgs[0].get('role') != 'system':
                msgs.insert(0, {
                    "role": "system",
                    "content": "You are an expert coding assistant. Answer questions about code clearly and accurately."
                })
            
            for m in msgs:
                if m.get('content'):
                    m['content'] = redact(m['content'])
            
            examples.append({"messages": msgs, "source": "pi-mono-chat"})
    
    print(f"  pi-mono-chat: {len(examples)} loaded, {skipped} skipped")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: burtenshaw/sft-on-traces (Codex format - use preprocess.py output)
# ─────────────────────────────────────────────────────────────────────────────
def load_burtenshaw(path: str) -> list[dict]:
    """Load the already-preprocessed burtenshaw example."""
    examples = []
    if not os.path.exists(path):
        print(f"  burtenshaw: not found at {path}")
        return []
    
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get('messages', [])
            if len(msgs) >= 3:
                examples.append({"messages": msgs, "source": "burtenshaw"})
    
    print(f"  burtenshaw: {len(examples)} loaded")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5: my-agy-traces (Personal Antigravity sessions exported by export_my_traces.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_my_traces(path: str, max_tokens: int = 24000) -> list[dict]:
    """Load your personalized SFT traces."""
    examples = []
    skipped = 0
    if not os.path.exists(path):
        return []
    
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get('messages', [])
            if len(msgs) < 3:
                continue
                
            est = estimate_tokens(msgs)
            if est > max_tokens:
                skipped += 1
                continue
                
            examples.append({"messages": msgs, "source": "my-agy-traces"})
            
    print(f"  my-agy-traces: {len(examples)} loaded, {skipped} skipped (over {max_tokens} tokens)")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Merge all datasets into train/eval')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.9)
    parser.add_argument('--max-tokens', type=int, default=24000,
                        help='Max tokens per example (T4 safe default)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MERGING ALL DATASETS")
    print("=" * 60)
    
    all_examples = []
    
    # 1. vojtavlas2 (biggest, highest quality)
    voj_path = "data/raw/pi_sft_qwen3_24k.jsonl"
    if os.path.exists(voj_path):
        all_examples.extend(load_vojtavlas2(voj_path, args.max_tokens))
    
    # 2. Pi-mono raw sessions
    all_examples.extend(load_pi_mono("data/raw/pi-mono", args.max_tokens))
    
    # 3. Pi-mono-chat
    chat_path = "data/raw/pi-mono-chat.jsonl"
    if os.path.exists(chat_path):
        all_examples.extend(load_pi_mono_chat(chat_path))
    
    # 4. Burtenshaw (already preprocessed)
    all_examples.extend(load_burtenshaw("data/train.jsonl"))
    
    # 5. Personal traces
    personal_path = "data/raw/all_my_traces.jsonl"
    if os.path.exists(personal_path):
        all_examples.extend(load_my_traces(personal_path, args.max_tokens))
    
    # Deduplicate
    seen = set()
    unique = []
    for ex in all_examples:
        h = example_hash(ex)
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    
    dupes = len(all_examples) - len(unique)
    print(f"\nDeduplicated: removed {dupes} duplicates")
    print(f"Total unique examples: {len(unique)}")
    
    # Source breakdown
    source_counts = Counter(ex.get('source', 'unknown') for ex in unique)
    print("\nBy source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src:20s} {cnt:6d} examples")
    
    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(unique)
    
    split_idx = int(len(unique) * args.train_split)
    train = unique[:split_idx]
    eval_  = unique[split_idx:]
    
    # Save
    train_path = os.path.join(args.output_dir, 'train.jsonl')
    eval_path  = os.path.join(args.output_dir, 'eval.jsonl')
    
    with open(train_path, 'w') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')
    
    with open(eval_path, 'w') as f:
        for ex in eval_:
            f.write(json.dumps(ex) + '\n')
    
    # Token statistics
    train_tokens = [estimate_tokens(ex['messages']) for ex in train]
    eval_tokens  = [estimate_tokens(ex['messages']) for ex in eval_]
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"  Train: {len(train):,} examples → {train_path}")
    print(f"  Eval:  {len(eval_):,} examples → {eval_path}")
    print(f"\n  Train token stats:")
    print(f"    Mean:   {sum(train_tokens)/len(train_tokens):,.0f}")
    print(f"    Median: {sorted(train_tokens)[len(train_tokens)//2]:,.0f}")
    print(f"    Min:    {min(train_tokens):,.0f}")
    print(f"    Max:    {max(train_tokens):,.0f}")
    print(f"    Total:  {sum(train_tokens):,.0f}")
    
    # File sizes
    print(f"\n  File sizes:")
    print(f"    train.jsonl: {os.path.getsize(train_path)/1e6:.1f} MB")
    print(f"    eval.jsonl:  {os.path.getsize(eval_path)/1e6:.1f} MB")


if __name__ == '__main__':
    main()
