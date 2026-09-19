# SFT Coding Agent — Continual Learning on Pi-Mono Traces

## Overview
This project implements a Supervised Fine-Tuning (SFT) pipeline for coding agents, following the methodology from Ben Burtenshaw's Training Agents. We fine-tune open large language models (like Gemma 4 and Qwen) on coding agent session traces from the `badlogicgames/pi-mono` dataset. The key technique used is completion-only loss masking, ensuring the model focuses on generating correct actions and code without being penalized for user inputs or system prompts.

## Architecture Pipeline
```text
[Pi-Mono Traces] -> [Dataset Prep (Chat format)] -> [DataCollatorForCompletionOnlyLM]
                                                              |
[Pre-trained Model (e.g. Gemma 4)] --> [QLoRA (4-bit)] --> [SFT Trainer] --> [Fine-tuned LoRA Adapter]
                                                              |
                                                       [trackio Logging]
```

## Quick Start
### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or via pip install .
```

### 2. Download Data
Traces are pulled from Hugging Face datasets (`badlogicgames/pi-mono`) or buckets.
```bash
# Add your data downloading script command here
```

### 3. Run Training
```bash
python train.py --config configs/default.yaml
```

## Free GPU Setup (Colab / Kaggle)
1. Ensure your environment has a T4 GPU (16GB VRAM) available.
2. Install dependencies: `pip install -r requirements.txt`
3. If using `bitsandbytes` on Colab, `accelerate` and `peft` will automatically handle 4-bit quantization as configured in `configs/default.yaml`.
4. Run the training script directly. The default batch size and gradient accumulation steps are tuned for 16GB GPUs.

## References
- Training Agents Repo: [burtenshaw/training-agents](https://github.com/burtenshaw/training-agents)
- Pi-Mono Dataset: [badlogicgames/pi-mono](https://huggingface.co/datasets/badlogicgames/pi-mono)
- Training Traces: `hf://buckets/burtenshaw/sft-on-traces`
- Trackio: `trackio` (W&B API-compatible tracking)

## License
MIT
