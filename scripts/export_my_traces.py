#!/usr/bin/env python3
"""
export_my_traces.py — Export your Antigravity (AGY) coding agent session transcripts
                      as SFT-ready training data for fine-tuning coding SLMs.

WHY THIS MATTERS FOR CONTINUAL LEARNING:
=========================================
This is the same concept as pi-share-hf, but for YOUR sessions.
Every time you work with an AI coding agent, you generate valuable training data:
  - Your prompts show what real developers ask
  - The agent's thinking reveals planning and reasoning strategies
  - Tool calls show which commands to run and when
  - Tool results + subsequent actions show error recovery

By capturing YOUR sessions, you create a personalized dataset that teaches
future models YOUR coding style, YOUR tools, YOUR project structures.

This is the continual learning loop:
  [You code with agent] → [Export traces] → [Fine-tune SLM] → [Better agent] → repeat

USAGE:
  # Export current session
  python3 scripts/export_my_traces.py

  # Export a specific conversation
  python3 scripts/export_my_traces.py --conversation-id <id>

  # Export all sessions in the AGY brain directory
  python3 scripts/export_my_traces.py --all
"""

import json
import os
import re
import hashlib
import glob
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PRIVACY — Comprehensive PII & Secret Redaction
# Run this on EVERY text field before saving. Better to over-redact than leak.
# ─────────────────────────────────────────────────────────────────────────────

# API Keys & Tokens
SECRET_PATTERNS = [
    re.compile(r'hf_[a-zA-Z0-9]{20,}'),                     # HuggingFace tokens
    re.compile(r'sk-[a-zA-Z0-9\-]{20,}'),                    # OpenAI / Anthropic keys
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),                      # GitHub PATs
    re.compile(r'gho_[a-zA-Z0-9]{30,}'),                     # GitHub OAuth tokens
    re.compile(r'github_pat_[a-zA-Z0-9_]{30,}'),             # GitHub fine-grained PATs
    re.compile(r'ghu_[a-zA-Z0-9]{36}'),                      # GitHub user-to-server tokens
    re.compile(r'AIza[0-9A-Za-z\-_]{35}'),                   # Google API keys
    re.compile(r'ya29\.[0-9A-Za-z\-_]+'),                    # Google OAuth access tokens
    re.compile(r'AKIA[0-9A-Z]{16}'),                         # AWS access key IDs
    re.compile(r'aws_secret_access_key\s*[=:]\s*\S+', re.I), # AWS secret keys
    re.compile(r'Bearer\s+[a-zA-Z0-9._\-]{20,}'),           # Bearer auth headers
    re.compile(r'token[\"\s:=]+[\"\']?[a-zA-Z0-9._\-]{20,}', re.IGNORECASE),
    re.compile(r'password[\"\s:=]+[\"\']?\S{8,}', re.IGNORECASE),
    re.compile(r'ssh-(?:rsa|ed25519|ecdsa)\s+\S{40,}'),      # SSH public keys
    re.compile(r'-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----[\s\S]*?-----END'),  # SSH private keys
]

# Personal Identifiable Information
PII_PATTERNS = [
    re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),  # Email addresses
    re.compile(r'(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)'),  # US phone numbers
    re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),             # IPv4 addresses
    re.compile(r'(?:https?://)?(?:[a-z0-9]+@)?github\.com[:/]\S+\.git'), # Git remote URLs with usernames
]

# Path patterns — anonymize home directory usernames
PATH_PATTERNS = [
    (re.compile(r'/Users/[a-zA-Z0-9._-]+/'), r'/Users/user/'),      # macOS
    (re.compile(r'/home/[a-zA-Z0-9._-]+/'), r'/home/user/'),         # Linux
    (re.compile(r'C:\\Users\\[a-zA-Z0-9._-]+\\'), r'C:\\Users\\user\\'),  # Windows
]

def redact_secrets(text: str) -> str:
    """Redact API keys, tokens, and credentials."""
    if not text:
        return text
    for p in SECRET_PATTERNS:
        text = p.sub('[REDACTED_SECRET]', text)
    return text

def redact_pii(text: str) -> str:
    """Redact emails, phone numbers, IP addresses."""
    if not text:
        return text
    for p in PII_PATTERNS:
        text = p.sub('[REDACTED_PII]', text)
    return text

def redact_paths(text: str) -> str:
    """Anonymize home directory usernames in file paths."""
    if not text:
        return text
    for pattern, replacement in PATH_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def full_redact(text: str) -> str:
    """Apply ALL redaction layers. Call this on every text field."""
    text = redact_secrets(text)
    text = redact_pii(text)
    text = redact_paths(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT for exported traces
# ─────────────────────────────────────────────────────────────────────────────
EXPORT_SYSTEM_PROMPT = """You are an expert AI coding agent with access to the following tools:

- **run_command**: Execute shell commands. Returns stdout/stderr and exit code.
- **view_file**: Read file contents with optional line range.
- **write_to_file**: Create a new file with specified content.
- **replace_file_content**: Edit specific lines in an existing file.
- **grep_search**: Search for patterns across files using ripgrep.
- **find_by_name**: Search for files/directories by name pattern.
- **list_dir**: List directory contents.
- **search_web**: Search the web for information.
- **read_url_content**: Fetch and read a URL's content.

Before acting, think through the problem step-by-step. Then use the appropriate tool.
After receiving results, continue reasoning and acting until the task is complete.
When you encounter errors, analyze them and try alternative approaches."""


def convert_agy_transcript(transcript_path: str, redact: bool = True) -> list[dict]:
    """
    Convert an Antigravity transcript.jsonl into SFT training examples.
    
    AGY transcript format:
      Each line is a JSON step with:
        - source: USER_EXPLICIT, MODEL, SYSTEM
        - type: USER_INPUT, PLANNER_RESPONSE, GENERIC, SYSTEM_MESSAGE
        - content: text content
        - thinking: model's chain-of-thought (when available)
        - tool_calls: list of tool invocations with args
    
    The pattern is:
      USER_INPUT → [PLANNER_RESPONSE with tool_calls → GENERIC (tool result)]* → PLANNER_RESPONSE (final answer)
    
    We convert this to:
      [system, user, assistant (with <think> + tool_calls), tool (result), assistant, ...]
    """
    
    steps = []
    with open(transcript_path) as f:
        for line in f:
            if line.strip():
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    if not steps:
        return []
    
    # Group steps into conversation turns
    # A "turn" starts with USER_INPUT and includes all subsequent MODEL/SYSTEM steps
    # until the next USER_INPUT
    turns = []
    current_turn = []
    
    for step in steps:
        if step.get('type') == 'USER_INPUT' and current_turn:
            turns.append(current_turn)
            current_turn = []
        current_turn.append(step)
    if current_turn:
        turns.append(current_turn)
    
    # Convert each turn group into training examples
    # Each user message + agent response cycle = one conversation
    examples = []
    conversation = [{"role": "system", "content": EXPORT_SYSTEM_PROMPT}]
    
    for turn_steps in turns:
        for step in turn_steps:
            source = step.get('source', '')
            stype = step.get('type', '')
            content = step.get('content', '') or ''
            thinking = step.get('thinking', '') or ''
            tool_calls = step.get('tool_calls', [])
            
            if redact:
                content = full_redact(content)
                thinking = full_redact(thinking)
            
            # ── User message ──────────────────────────────────────────
            if stype == 'USER_INPUT':
                # Clean up the content (remove system metadata wrappers)
                user_text = content.strip()
                if user_text:
                    conversation.append({"role": "user", "content": user_text})
            
            # ── Agent response (with optional thinking + tool calls) ──
            elif stype == 'PLANNER_RESPONSE' and source == 'MODEL':
                assistant_parts = []
                
                # Add thinking as <think> block
                if thinking.strip():
                    assistant_parts.append(f"<think>{thinking.strip()}</think>")
                
                # Add response content
                if content.strip():
                    assistant_parts.append(content.strip())
                
                assistant_content = '\n'.join(assistant_parts)
                
                msg = {"role": "assistant", "content": assistant_content}
                
                # Convert tool calls to standard format
                if tool_calls:
                    formatted_calls = []
                    for tc in tool_calls:
                        name = tc.get('name', '')
                        args = tc.get('args', {})
                        
                        # Skip internal/meta tool calls
                        if name in ('invoke_subagent', 'manage_subagents', 'define_subagent',
                                    'send_message', 'manage_task', 'schedule',
                                    'ask_question', 'generate_image'):
                            continue
                        
                        # Clean args — remove toolAction/toolSummary metadata
                        clean_args = {
                            k: v for k, v in args.items()
                            if k not in ('toolAction', 'toolSummary')
                        }
                        
                        if redact:
                            clean_args = {k: full_redact(str(v)) if isinstance(v, str) else v
                                         for k, v in clean_args.items()}
                        
                        call_id = hashlib.md5(
                            f"{name}:{json.dumps(clean_args)[:200]}".encode()
                        ).hexdigest()[:12]
                        
                        formatted_calls.append({
                            "id": f"call_{call_id}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(clean_args)
                            }
                        })
                    
                    if formatted_calls:
                        msg["tool_calls"] = formatted_calls
                
                if assistant_content or msg.get("tool_calls"):
                    conversation.append(msg)
            
            # ── Tool result (GENERIC step after a tool call) ──────────
            elif stype == 'GENERIC' and source == 'MODEL':
                if content.strip():
                    tool_content = content.strip()
                    if redact:
                        tool_content = full_redact(tool_content)
                    
                    # Only add as tool result if the previous message had tool_calls
                    if (conversation and 
                        conversation[-1].get('role') == 'assistant' and 
                        conversation[-1].get('tool_calls')):
                        
                        # Match to the last tool call
                        last_calls = conversation[-1].get('tool_calls', [])
                        call_id = last_calls[-1]['id'] if last_calls else 'unknown'
                        
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_content
                        })
            
            # ── System messages (skip most, keep important ones) ──────
            elif stype == 'SYSTEM_MESSAGE':
                pass  # Skip system bookkeeping messages
    
    # Filter: need meaningful content
    non_system = [m for m in conversation if m['role'] != 'system']
    if len(non_system) < 3:
        return []
    
    has_user = any(m['role'] == 'user' for m in non_system)
    has_assistant = any(m['role'] == 'assistant' for m in non_system)
    
    if not (has_user and has_assistant):
        return []
    
    # Split long conversations into chunks at natural user turn boundaries
    # to keep examples under token limits
    MAX_TOKENS_PER_EXAMPLE = 24000
    
    results = []
    current_example = [{"role": "system", "content": EXPORT_SYSTEM_PROMPT}]
    current_tokens = len(EXPORT_SYSTEM_PROMPT) // 4
    
    for msg in conversation[1:]:  # skip system prompt (already added)
        msg_tokens = len(json.dumps(msg)) // 4
        
        if current_tokens + msg_tokens > MAX_TOKENS_PER_EXAMPLE and msg['role'] == 'user':
            # Save current chunk if meaningful
            non_sys = [m for m in current_example if m['role'] != 'system']
            if len(non_sys) >= 3:
                results.append({"messages": current_example, "source": "my-agy-traces"})
            # Start new chunk
            current_example = [{"role": "system", "content": EXPORT_SYSTEM_PROMPT}]
            current_tokens = len(EXPORT_SYSTEM_PROMPT) // 4
        
        current_example.append(msg)
        current_tokens += msg_tokens
    
    # Save final chunk
    non_sys = [m for m in current_example if m['role'] != 'system']
    if len(non_sys) >= 3:
        results.append({"messages": current_example, "source": "my-agy-traces"})
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Export your Antigravity agent traces as SFT training data'
    )
    parser.add_argument('--conversation-id', type=str,
                        default='d3872ad5-fb65-41e6-8816-843fa80715d9',
                        help='Conversation ID to export')
    parser.add_argument('--all', action='store_true',
                        help='Export ALL conversations from the AGY brain directory')
    parser.add_argument('--output', type=str, default='data/raw/my_traces.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--brain-dir', type=str,
                        default=os.path.expanduser('~/.gemini/antigravity/brain'),
                        help='AGY brain directory')
    parser.add_argument('--no-redact', action='store_true',
                        help='Skip secret redaction (NOT recommended)')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 60)
    print("EXPORTING ANTIGRAVITY AGENT TRACES")
    print("=" * 60)
    
    all_examples = []
    
    if args.all:
        # Find all conversation transcripts
        pattern = os.path.join(args.brain_dir, '*', '.system_generated', 'logs', 'transcript_full.jsonl')
        transcripts = glob.glob(pattern)
        if not transcripts:
            # Fallback to compact
            pattern = os.path.join(args.brain_dir, '*', '.system_generated', 'logs', 'transcript.jsonl')
            transcripts = glob.glob(pattern)
        print(f"Found {len(transcripts)} conversation transcripts")
    else:
        # Single conversation
        base = os.path.join(args.brain_dir, args.conversation_id, '.system_generated', 'logs')
        full = os.path.join(base, 'transcript_full.jsonl')
        compact = os.path.join(base, 'transcript.jsonl')
        transcripts = [full if os.path.exists(full) else compact]
        print(f"Exporting conversation: {args.conversation_id}")
    
    for t_path in transcripts:
        if not os.path.exists(t_path):
            continue
        
        conv_id = Path(t_path).parent.parent.parent.name
        print(f"\n  Processing: {conv_id[:12]}...")
        
        examples = convert_agy_transcript(t_path, redact=not args.no_redact)
        print(f"    → {len(examples)} training examples extracted")
        
        for ex in examples:
            ex['conversation_id'] = conv_id
            ex['exported_at'] = datetime.now().isoformat()
        
        all_examples.extend(examples)
    
    # Save
    with open(args.output, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    
    # Stats
    total_msgs = sum(len(ex['messages']) for ex in all_examples)
    total_tokens = sum(
        sum(len(m.get('content', '') or '') for m in ex['messages']) // 4
        for ex in all_examples
    )
    
    tool_call_count = sum(
        sum(len(m.get('tool_calls', [])) for m in ex['messages'])
        for ex in all_examples
    )
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"  Examples:      {len(all_examples)}")
    print(f"  Total messages: {total_msgs}")
    print(f"  Total tokens:  ~{total_tokens:,}")
    print(f"  Tool calls:    {tool_call_count}")
    print(f"  Output:        {args.output}")
    print(f"  Secrets:       {'NOT redacted ⚠️' if args.no_redact else 'Redacted ✅'}")
    print(f"\n  To include in training, re-run:")
    print(f"    python3 scripts/merge_all_data.py")


if __name__ == '__main__':
    main()
