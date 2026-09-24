# SFT Coding Agent - eigentiki

I trained a large language model to work as an autonomous coding agent. I named it eigentiki. I used a method called Supervised Fine-Tuning.

## What this project does

I took a base model (Google Gemma 2 9B). I taught it how to think, use tools, and fix errors. I trained it on 6,625 real coding sessions. 

The most important part of this training is "completion-only loss masking". I forced the model to only learn from the agent actions. I hid the system prompts and the tool outputs during training. This stops the model from trying to guess what the computer will say back to it.

## Links

* **Model on Hugging Face:** [focustiki/eigentiki](https://huggingface.co/focustiki/eigentiki)
* **Dataset on Hugging Face:** [focustiki/sft-coding-agent-traces](https://huggingface.co/focustiki/sft-coding-agent-traces)
* **Training Code:** [notebooks/sft_training.ipynb](notebooks/sft_training.ipynb)

## How I did the training

I did not guess the best settings. I ran a sweep of three different setups. I set a rule to pick the winner based on the lowest evaluation loss on held-out data. I trained the model using 4-bit QLoRA and Unsloth on an A100 GPU in Google Colab.

### Sweep Results

| Run Name | Learning Rate | LoRA Rank | Sequence Length | Eval Loss | Train Loss |
|---|---|---|---|---|---|
| **run_1 (Winner)** | 0.0002 | 16 | 4096 | **4.5200** | 5.6922 |
| run_2 | 0.0001 | 32 | 8192 | 4.6942 | 5.9130 |
| run_3 | 0.00005 | 64 | 8192 | 4.8814 | 6.1487 |

Run 1 won because it had the lowest evaluation loss (4.5200). I uploaded this winning version to Hugging Face.

## Sanity Check

I tested the winning model on 10 HumanEval Python problems. The model successfully generated code for all 10 problems. This proved the training did not break the base coding skills of the model. A full evaluation of agent skills requires Inspect AI and vLLM.

## Folder Structure

* `scripts/`: Code to clean private data, merge trace files, and push the dataset to Hugging Face.
* `notebooks/`: The Google Colab notebook used for the training sweep.
* `configs/`: Files that hold settings for the project.
