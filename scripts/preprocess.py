#!/usr/bin/env python3
"""
preprocess.py - Convert Pi-Mono/Codex agent traces to SFT training format.

WHAT THIS SCRIPT DOES (read this before running!):
==================================================
The raw trace file (example.jsonl) is an event stream from a real Codex/Pi 
coding session. Each line is one event. Events are grouped into "turns":
  - One user message (the task)
  - Many assistant reasoning + tool call steps
  - Tool outputs from the terminal environment

We convert this into the chat format SFTTrainer expects:
  {"messages": [
    {"role": "system", "content": "You are a coding agent..."},
    {"role": "user",  "content": "sft train gemma 4 2b model..."},
    {"role": "assistant", "content": "[thinking]\nI'll check the repo...",
     "tool_calls": [{"id": "call_nHZaci3", "type": "function",
                     "function": {"name": "exec_command", "arguments": "..."}}]},
    {"role": "tool", "tool_call_id": "call_nHZaci3", "content": "...stdout..."},
    {"role": "assistant", "content": "Done! The model has been trained."},
  ]}

CRITICAL LEARNING POINT - Loss Masking:
  During training, SFTTrainer will apply DataCollatorForCompletionOnlyLM.
  This masks ALL tokens except assistant turns.
  Tokens with label=-100 contribute ZERO gradient.
  
  Masked (label=-100):  system, user, tool (outputs)
  Trained (label=token): assistant content + tool_calls
  
  Why? Tool outputs ("Chunk ID: 3eae07\nWall time...") are unpredictable 
  external world outputs. Training the model to predict them causes gradient 
  noise and hallucination. We only want it to learn WHEN and HOW to act.

HOW THE TRACE MAPS TO CHAT FORMAT:
  Raw event type           → Chat role
  ─────────────────────────────────────────────────
  session_meta             → (skip - metadata only)
  event_msg/task_started   → (skip - bookkeeping)
  event_msg/user_message   → user
  response_item/message    → assistant (phase=commentary)
  response_item/function_call → assistant tool_calls
  response_item/function_call_output → tool
  response_item/reasoning  → (skip - encrypted, not trainable)
  turn_context             → (skip - retrieval context)
"""

import json
import os
import random
import re
import hashlib
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT - This defines the agent's identity and tool schemas.
# The model sees this at position 0 of every conversation.
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

Before taking action, reason about the task. Then call the appropriate tool.
After receiving tool results, continue reasoning and acting until the task is complete."""

# ─────────────────────────────────────────────────────────────────────────────
# PRIVACY PATTERNS - Redact anything that looks like a secret.
# pi-share-hf should already handle this, but we double-check.
# ─────────────────────────────────────────────────────────────────────────────
SECRET_PATTERNS = [
    r'hf_[a-zA-Z0-9]{30,}',          # HuggingFace tokens
    r'sk-[a-zA-Z0-9]{30,}',           # OpenAI keys
    r'ghp_[a-zA-Z0-9]{36}',           # GitHub personal access tokens
    r'AIza[0-9A-Za-z\-_]{35}',        # Google API keys
    r'[a-zA-Z0-9+/]{40,}={0,2}',      # Generic base64 secrets (long)
]

def redact_secrets(text: str) -> str:
    """Replace any detected secrets with a placeholder."""
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text)
    return text


def extract_text(content) -> str:
    """Extract plain text from a content field.
    
    Content can be a plain string OR a list of typed blocks:
      [{"type": "output_text", "text": "..."}, ...]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get('text', '') or block.get('content', ''))
        return '\n'.join(p for p in parts if p)
    return ''


def events_to_chat(events: list[dict]) -> dict | None:
    """
    Convert a list of raw trace events into one training example.
    
    Returns a dict with {"messages": [...]} or None if the trace
    should be filtered out (too short, no user task, etc.)
    
    LEARNING NOTE: Walk through this function line-by-line.
    The structure of the output is exactly what SFTTrainer consumes.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # We need to match tool_calls with their outputs using call_id.
    # Build a map of call_id → index in messages so we can append
    # tool outputs in the right order.
    pending_tool_calls = {}  # call_id → message index where it lives
    
    for event in events:
        etype = event.get('type')
        payload = event.get('payload', {})
        
        if not isinstance(payload, dict):
            continue
        
        ptype = payload.get('type')
        role  = payload.get('role', '')
        
        # ── User message ──────────────────────────────────────────────────
        if etype == 'event_msg' and ptype == 'user_message':
            raw_msg = payload.get('message', '')
            text = raw_msg if isinstance(raw_msg, str) else extract_text(raw_msg)
            text = redact_secrets(text.strip())
            if text:
                messages.append({"role": "user", "content": text})
        
        # ── User message (alternate: response_item with role=user) ────────
        elif etype == 'response_item' and ptype == 'message' and role == 'user':
            text = redact_secrets(extract_text(payload.get('content', '')).strip())
            if text:
                # Don't add if we already have this user turn
                if not messages or messages[-1]['role'] != 'user':
                    messages.append({"role": "user", "content": text})
        
        # ── Assistant message (commentary/final response) ─────────────────
        elif etype == 'response_item' and ptype == 'message' and role == 'assistant':
            text = redact_secrets(extract_text(payload.get('content', '')).strip())
            phase = payload.get('phase', '')
            if text:
                # Merge into existing assistant turn if last message is already assistant
                # (multiple commentary blocks = one conceptual assistant turn)
                if messages and messages[-1]['role'] == 'assistant' and 'tool_calls' not in messages[-1]:
                    messages[-1]['content'] += '\n' + text
                else:
                    messages.append({"role": "assistant", "content": text})
        
        # ── Tool call emitted by the model ────────────────────────────────
        # LEARNING NOTE: This is a TARGET TOKEN - the model must learn to 
        # generate these. It will receive gradient updates here.
        elif etype == 'response_item' and ptype == 'function_call':
            call_id  = payload.get('call_id', '')
            name     = payload.get('name', '')
            args     = payload.get('arguments', '{}')
            
            tool_call_obj = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args}
            }
            
            # Attach to existing assistant turn, or create a new one
            if messages and messages[-1]['role'] == 'assistant':
                if 'tool_calls' not in messages[-1]:
                    messages[-1]['tool_calls'] = []
                messages[-1]['tool_calls'].append(tool_call_obj)
            else:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call_obj]
                })
            
            # Remember where this call_id lives so the output can find it
            pending_tool_calls[call_id] = len(messages) - 1
        
        # ── Tool output (environment response) ────────────────────────────
        # LEARNING NOTE: This is a MASKED token - the model should NOT try to
        # predict bash stdout. label=-100 means ZERO gradient here.
        elif etype == 'response_item' and ptype == 'function_call_output':
            call_id = payload.get('call_id', '')
            output  = redact_secrets(payload.get('output', '').strip())
            if output:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output
                })
        
        # ── Skip: reasoning (encrypted), session_meta, bookkeeping ────────
        # Encrypted reasoning is not trainable; session_meta is metadata.
    
    # ── Quality filters ───────────────────────────────────────────────────
    # Count meaningful turns (exclude system prompt)
    turns = [m for m in messages if m['role'] != 'system']
    
    if len(turns) < 3:
        return None  # Too short to learn from
    
    has_user = any(m['role'] == 'user' for m in turns)
    has_assistant = any(m['role'] == 'assistant' for m in turns)
    has_tool = any(m['role'] in ('assistant',) and 'tool_calls' in m for m in turns)
    
    if not (has_user and has_assistant):
        return None  # Missing basic agent structure
    
    return {"messages": messages}


def split_into_sessions(all_events: list[dict]) -> list[list[dict]]:
    """
    A single JSONL file may contain multiple sessions.
    Split on 'session_meta' events to get individual sessions.
    Each session is one training example.
    """
    sessions = []
    current = []
    
    for event in all_events:
        if event.get('type') == 'session_meta' and current:
            sessions.append(current)
            current = []
        current.append(event)
    
    if current:
        sessions.append(current)
    
    return sessions


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: 4 characters ≈ 1 token."""
    total_chars = sum(
        len(m.get('content', '') or '') +
        len(json.dumps(m.get('tool_calls', [])))
        for m in messages
    )
    return total_chars // 4


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Pi-Mono traces into SFT chat format'
    )
    parser.add_argument('--input',  default='data/raw/example.jsonl', help='Raw trace JSONL')
    parser.add_argument('--output-dir', default='data', help='Where to save train.jsonl + eval.jsonl')
    parser.add_argument('--train-split', type=float, default=0.9, help='Fraction for training (rest = eval)')
    parser.add_argument('--min-turns', type=int, default=3, help='Minimum non-system turns to keep')
    parser.add_argument('--max-tokens', type=int, default=32768, help='Max token estimate before filtering')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Reading events from: {args.input}')
    all_events = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f'Total raw events loaded: {len(all_events)}')
    
    # Split into sessions
    sessions = split_into_sessions(all_events)
    print(f'Sessions found: {len(sessions)}')
    
    # Convert each session to chat format
    stats = {
        'total': len(sessions),
        'converted': 0,
        'filtered_too_short': 0,
        'filtered_too_long': 0,
        'filtered_no_structure': 0,
    }
    
    examples = []
    for i, session_events in enumerate(sessions):
        chat = events_to_chat(session_events)
        
        if chat is None:
            stats['filtered_no_structure'] += 1
            continue
        
        turns = [m for m in chat['messages'] if m['role'] != 'system']
        if len(turns) < args.min_turns:
            stats['filtered_too_short'] += 1
            continue
        
        est_tokens = estimate_tokens(chat['messages'])
        if est_tokens > args.max_tokens:
            stats['filtered_too_long'] += 1
            continue
        
        examples.append(chat)
        stats['converted'] += 1
    
    if not examples:
        print('\n  No examples produced. The file may contain only one large session.')
        print('Writing it as a single training example...')
        # Try treating the whole file as one session
        chat = events_to_chat(all_events)
        if chat:
            examples = [chat]
            stats['converted'] = 1
    
    # Shuffle & split
    random.seed(42)
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * args.train_split))
    train_data = examples[:split_idx]
    eval_data  = examples[split_idx:] if len(examples) > 1 else examples  # eval = copy if only 1 example
    
    # Save
    train_path = os.path.join(args.output_dir, 'train.jsonl')
    eval_path  = os.path.join(args.output_dir, 'eval.jsonl')
    
    with open(train_path, 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex) + '\n')
    
    with open(eval_path, 'w') as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + '\n')
    
    # Report
    print('\n' + '='*50)
    print('PREPROCESSING COMPLETE')
    print('='*50)
    print(f"  Raw events:          {stats['total']} sessions")
    print(f"   Converted:         {stats['converted']}")
    print(f"  ❌ Too short:         {stats['filtered_too_short']}")
    print(f"  ❌ Too long:          {stats['filtered_too_long']}")
    print(f"  ❌ No structure:      {stats['filtered_no_structure']}")
    print(f"  Train examples:      {len(train_data)}  → {train_path}")
    print(f"  Eval examples:       {len(eval_data)}  → {eval_path}")
    
    # Show one converted example
    if examples:
        ex = examples[0]
        msgs = ex['messages']
        print(f'\n--- SAMPLE CONVERTED EXAMPLE ---')
        print(f'  Total messages: {len(msgs)}')
        for m in msgs:
            role = m['role']
            has_calls = 'tool_calls' in m
            text_preview = (m.get('content') or '')[:80].replace('\n', ' ')
            if has_calls:
                call_names = [tc['function']['name'] for tc in m['tool_calls']]
                print(f'  [{role:9}] + tool_calls={call_names}  "{text_preview}"')
            else:
                print(f'  [{role:9}] "{text_preview}"')
        print()
        est = estimate_tokens(msgs)
        print(f'  Estimated tokens: ~{est:,}')
        print(f'\n   Token masking will apply to all [tool] and [user] and [system] tokens.')
        print(f'   Gradient will flow ONLY through [assistant] tokens.')


if __name__ == '__main__':
    main()
