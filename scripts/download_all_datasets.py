#!/usr/bin/env python3
"""
download_all_datasets.py - Download and merge all coding agent datasets for SFT training.

DATASETS WE USE AND WHY EACH ONE MATTERS:
==========================================

1. burtenshaw/sft-on-traces (HF Bucket)
   - What: Ben Burtenshaw's own curated SFT example from his livestream
   - Format: Codex/OpenAI-style event stream (response_item, function_call, etc.)
   - Size: 1 example (~150K tokens) - Ben's complete agent session
   - Why use it: Gold-standard example showing professional agent workflow end-to-end
   -  Already downloaded as data/raw/example.jsonl

2. badlogicgames/pi-mono (HF Dataset)
   - What: 627 real coding sessions from Mario Zechner (creator of the Pi coding agent)
   - Format: Pi native format (session, message, compaction, branch_summary events)
   - Size: 224.8 MB across 627 JSONL files
   - Why use it: Raw, real-world coding traces covering bugs, features, refactors
   -  Different format from burtenshaw - needs its own parser branch

3. sergiopaniego/pi-mono-chat (HF Dataset)
   - What: 797 train + 89 test pre-formatted (user, assistant) pairs from Pi sessions
   - Format: Standard {"messages": [{"role": ..., "content": ...}]} - NO tool calls
   - Size: ~940 KB - small but clean
   - Why use it: Easy to merge, provides conversational (non-tool) coding knowledge
   -  No tool calls - useful for response quality, not tool-use learning

LEARNING INSIGHT - Why use all three?
   Diverse data → robust generalization. The model sees:
   - Professional OpenAI Codex-style agentic sessions (burtenshaw)
   - Native Pi agent sessions with rich tool use (pi-mono)  
   - Clean Q&A coding knowledge (pi-mono-chat)
   Combining these reduces overfitting to any single session style.

RUN:
   python3 scripts/download_all_datasets.py
   
   Then run preprocessing:
   python3 scripts/preprocess.py (handles both formats automatically)
"""

import json
import os
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Load token from .env file if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ.setdefault("HF_TOKEN", line.split("=", 1)[1].strip())

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment or .env file")
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"}

RAW_DIR   = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hf_get(url: str) -> bytes:
    """GET a HuggingFace URL with auth, with simple retry."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limit
                wait = 2 ** attempt * 5
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after 3 attempts: {url}")


def hf_get_json(url: str) -> dict | list:
    return json.loads(hf_get(url).decode())


def download_file(url: str, dest: Path, label: str = "") -> int:
    """Download a file and return bytes written. Skips if already exists."""
    if dest.exists() and dest.stat().st_size > 100:
        return dest.stat().st_size  # already downloaded
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return len(data)
        except Exception as e:
            if attempt == 2:
                print(f"    Failed to download {label}: {e}")
                return 0
            time.sleep(2 ** attempt)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 - burtenshaw/sft-on-traces (already downloaded)
# ─────────────────────────────────────────────────────────────────────────────
def check_burtenshaw_traces():
    dest = RAW_DIR / "example.jsonl"
    if dest.exists():
        size = dest.stat().st_size
        print(f" burtenshaw/sft-on-traces: already downloaded ({size/1e6:.2f} MB)")
        return True
    
    print(" Downloading burtenshaw/sft-on-traces/example.jsonl ...")
    url = "https://huggingface.co/buckets/burtenshaw/sft-on-traces/resolve/example.jsonl"
    n = download_file(url, dest, "sft-on-traces")
    print(f"   → {n/1e6:.2f} MB downloaded")
    return n > 0


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 - badlogicgames/pi-mono (627 raw JSONL files)
# ─────────────────────────────────────────────────────────────────────────────
def download_pi_mono(max_files: int = None, min_size_bytes: int = 5000):
    """
    Download JSONL session files from badlogicgames/pi-mono.
    
    Args:
        max_files: Cap on number of files to download (None = all 627)
        min_size_bytes: Skip tiny sessions (default 5 KB minimum)
    
    The pi-mono format is the NATIVE Pi agent format:
      {"type": "session", "id": ..., "cwd": ...}
      {"type": "message", "message": {"role": "user", "content": [...]}}
      {"type": "message", "message": {"role": "assistant", "content": [...]}}
      etc.
    This is DIFFERENT from the burtenshaw/sft-on-traces Codex format.
    Our preprocess.py handles both.
    """
    pi_mono_dir = RAW_DIR / "pi-mono"
    pi_mono_dir.mkdir(exist_ok=True)
    
    print("\n Downloading badlogicgames/pi-mono ...")
    print("   Getting file list...")
    
    files_data = hf_get_json(
        "https://huggingface.co/api/datasets/badlogicgames/pi-mono/tree/main?limit=1000"
    )
    
    jsonl_files = [
        f for f in files_data
        if f.get("path", "").endswith(".jsonl") and f.get("size", 0) >= min_size_bytes
    ]
    
    print(f"   Found {len(jsonl_files)} usable sessions (≥{min_size_bytes/1e3:.0f} KB)")
    
    if max_files:
        # Prioritize larger files (longer = richer sessions)
        jsonl_files = sorted(jsonl_files, key=lambda f: -f["size"])[:max_files]
        print(f"   Limiting to top {max_files} largest sessions")
    
    total_bytes = sum(f["size"] for f in jsonl_files)
    print(f"   Total to download: {total_bytes/1e6:.1f} MB")
    
    downloaded = 0
    skipped = 0
    for i, file_info in enumerate(jsonl_files):
        path = file_info["path"]
        fname = Path(path).name
        dest = pi_mono_dir / fname
        
        url = f"https://huggingface.co/datasets/badlogicgames/pi-mono/resolve/main/{path}"
        n = download_file(url, dest, fname)
        
        if n > 0:
            downloaded += 1
        else:
            skipped += 1
        
        if (i + 1) % 50 == 0 or (i + 1) == len(jsonl_files):
            print(f"   Progress: {i+1}/{len(jsonl_files)} files ({downloaded} downloaded, {skipped} skipped)")
        
        # Small polite delay to avoid hammering HF servers
        if i % 10 == 9:
            time.sleep(0.5)
    
    actual_files = list(pi_mono_dir.glob("*.jsonl"))
    actual_size = sum(f.stat().st_size for f in actual_files)
    print(f" badlogicgames/pi-mono: {len(actual_files)} files, {actual_size/1e6:.1f} MB total")
    return len(actual_files)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3 - sergiopaniego/pi-mono-chat (pre-formatted chat, via parquet)
# ─────────────────────────────────────────────────────────────────────────────
def download_pi_mono_chat():
    """
    Download sergiopaniego/pi-mono-chat and convert to JSONL.
    
    This dataset is already in {"messages": [{"role":..., "content":...}]} format.
    No tool calls - pure user/assistant coding conversations derived from Pi sessions.
    
    LEARNING NOTE: This is what pi-mono sessions look like AFTER distillation - 
    the long multi-turn tool-heavy traces have been compressed into clean 
    user→assistant pairs. Great for teaching conversational style, less so for 
    tool-use mechanics.
    """
    dest = RAW_DIR / "pi-mono-chat.jsonl"
    if dest.exists() and dest.stat().st_size > 10000:
        size = dest.stat().st_size
        print(f" sergiopaniego/pi-mono-chat: already downloaded ({size/1e3:.0f} KB)")
        return True
    
    print("\n Downloading sergiopaniego/pi-mono-chat ...")
    
    examples = []
    for split in ["train", "test"]:
        offset = 0
        limit = 100
        while True:
            url = (
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset=sergiopaniego/pi-mono-chat"
                f"&config=default&split={split}"
                f"&offset={offset}&limit={limit}"
            )
            data = hf_get_json(url)
            rows = data.get("rows", [])
            if not rows:
                break
            
            for r in rows:
                row = r["row"]
                msgs = row.get("messages", [])
                if len(msgs) >= 2:
                    # Add system prompt to give context
                    full_msgs = [{
                        "role": "system",
                        "content": (
                            "You are an expert coding assistant. "
                            "Answer questions about code clearly and accurately."
                        )
                    }] + msgs
                    examples.append({"messages": full_msgs, "source": "pi-mono-chat"})
            
            offset += len(rows)
            total = data.get("num_rows_total", 0)
            if offset >= total:
                break
        
        print(f"   {split}: collected {offset} examples")
    
    with open(dest, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f" sergiopaniego/pi-mono-chat: {len(examples)} examples → {dest}")
    return len(examples)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download all SFT coding agent datasets")
    parser.add_argument("--pi-mono-limit", type=int, default=None,
                        help="Max number of pi-mono files to download (default: all 627)")
    parser.add_argument("--skip-pi-mono", action="store_true",
                        help="Skip the large badlogicgames/pi-mono download")
    args = parser.parse_args()

    print("=" * 60)
    print("SFT CODING AGENT - Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {RAW_DIR.absolute()}")
    print()

    # 1. burtenshaw/sft-on-traces
    check_burtenshaw_traces()

    # 2. badlogicgames/pi-mono
    if not args.skip_pi_mono:
        download_pi_mono(max_files=args.pi_mono_limit, min_size_bytes=5000)
    else:
        print("⏭️  Skipping badlogicgames/pi-mono (--skip-pi-mono)")

    # 3. sergiopaniego/pi-mono-chat
    download_pi_mono_chat()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    all_files = list(RAW_DIR.rglob("*.jsonl"))
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"Total raw files:  {len(all_files)}")
    print(f"Total raw size:   {total_size/1e6:.1f} MB")
    print()
    print("Next step: run preprocessing to convert all formats to SFT chat format:")
    print("  python3 scripts/preprocess.py --input data/raw/ --output-dir data/")


if __name__ == "__main__":
    main()
