# Evaluation Guide for eigentiki

This document explains how to test your AI agent. I wrote this guide in simple English. Anyone can understand this high-level framework.

## 1. What is an Agent Evaluation?
An evaluation is a test. We do not just ask the model a question. We give the model a goal. We put the model in a safe box called a sandbox. We give it tools, like a terminal and a Python runner. We watch how it uses the tools to achieve the goal. Finally, we score the result.

## 2. The Framework
We use two main tools for this test:
* **Inspect AI:** This is the test manager. It creates the sandbox. It loads the test questions. It records the agent actions. It scores the final answer.
* **vLLM:** This is the engine. It runs your model very fast. It requires a Graphics Processing Unit (GPU) to work.

## 3. The Test Data
We use a dataset called HumanEval. This dataset contains basic Python coding problems. The agent must read the problem, write the code, test it, and submit the correct code.

## 4. How the Hugging Face Space Works
A Hugging Face Space is a computer in the cloud. We use a "Docker" space. Docker is a system that packages all the code and tools into one container.

When the Space starts, it does two things:
1. It runs the evaluation script. The agent attempts to solve 5 problems.
2. It starts a web server on port 7860. This web server shows a beautiful dashboard with the test results.

## 5. Important Warning About Hardware
By default, Hugging Face gives you a free CPU (a standard processor). Your model is too large for a CPU. The test will fail on a free CPU. You must go to your Space settings and select a GPU (like an L4 or A10G). This will cost a small amount of money per hour.

## 6. Option A: Run the Test in Google Colab (Recommended)
I made a notebook for this: `notebooks/eval_agent.ipynb`.

What it tests:
* **Code test:** The model solves 164 HumanEval problems. I run the official unit tests. The score is pass@1.
* **Format test:** The model gets 5 agent tasks. I check if it writes a valid tool call.

How to run it:
1. Open the notebook in Colab from GitHub.
2. Select the A100 GPU.
3. Add the secret `HF_TOKEN` in Colab Secrets.
4. Click Run all.

How to get a fair comparison:
1. Run the notebook one time as it is. This tests `focustiki/eigentiki`.
2. Change `MODEL_NAME` to `unsloth/gemma-2-9b-it-bnb-4bit`. Run it again. This gives the base score.
3. Compare the two scores. A similar score means the training kept the coding skill.

Cost: Colab uses compute units. An A100 uses about 7 units each hour. One run takes about 1 hour or less.

## 7. Option B: Hugging Face Space (Paid)
The files are in `hf_space_files/`. Hugging Face charges per hour for a GPU. The price is about 0.40 to 2.50 US dollars each hour, based on the GPU. Check the current price on the Space settings page before you start.

## 8. Limits of These Tests
* HumanEval tests single functions. It does not test long agent work with many tool steps.
* The format test uses only 5 tasks. It shows if the format is correct. It does not show if the agent is smart.
* The Inspect AI script in `scripts/evaluate_agent.py` now runs the official unit tests. I checked the scorer with two fake answers. The perfect answers scored 100 percent. The bad answers scored 0 percent. This shows the scorer works.
* I have not run the script on the real model yet. I must do this on a GPU machine.

## 9. How to Run the Inspect AI Script
My model on the Hub is a LoRA adapter. It sits on top of the base model `google/gemma-2-9b-it`. This base model needs access. I must accept the license on the Hugging Face page and use my token.

Steps on a GPU machine:
1. Install the tools: `pip install inspect-ai transformers peft accelerate torch`
2. Set my token: `export HF_TOKEN=<my token>`
3. Run a small test first: `inspect eval scripts/evaluate_agent.py --model hf/focustiki/eigentiki --limit 10`
4. Run the full test: `inspect eval scripts/evaluate_agent.py --model hf/focustiki/eigentiki`
5. Open the result viewer: `inspect view`

To use Docker for the sandbox, add `-T sandbox=docker`. Docker is safer, because the model code runs in a closed box.

Note on vLLM: vLLM is faster. But it needs a merged model, not an adapter. I do not need it for 164 problems.
