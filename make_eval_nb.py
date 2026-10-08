import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(True)}

cells = []

cells.append(md("""
# Test eigentiki (Evaluation Notebook)

I use this notebook to test my model `focustiki/eigentiki`. I run it on a Colab A100 GPU.

## What this test does
1. **Code test (HumanEval):** The model gets 164 Python problems. It writes a function for each one. I then run the official unit tests on each function. The score is the share of problems that pass. This is called pass@1.
2. **Format test:** I give the model 5 agent tasks. I check if the model writes a valid tool call in the format I trained it on.

## Before I start
1. Pick the **A100 GPU** in Runtime > Change runtime type.
2. Add my Hugging Face token to Colab Secrets. The name is `HF_TOKEN`. Notebook access must be on.
3. Click Runtime > Run all.

## How to read the result
- A score is only useful when I compare it to something. I compare it to the base model `google/gemma-2-9b-it`.
- A big drop from the base score means the training hurt the coding skill.
- A similar score means the training kept the coding skill and added agent skill.
"""))

cells.append(md("## 1. Install the tools"))
cells.append(code("""
!pip install unsloth
!pip install --upgrade datasets
"""))

cells.append(md("## 2. Log in to Hugging Face"))
cells.append(code("""
from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get("HF_TOKEN")
login(token=hf_token)
print("I logged in to Hugging Face.")
"""))

cells.append(md("""
## 3. Load the model
I load my trained model from the Hub. I can change `MODEL_NAME` to `unsloth/gemma-2-9b-it-bnb-4bit` to test the base model. I do this to get the base score.
"""))
cells.append(code("""
from unsloth import FastLanguageModel

MODEL_NAME = "focustiki/eigentiki"   # change to "unsloth/gemma-2-9b-it-bnb-4bit" for the base score

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
print("I loaded:", MODEL_NAME)
"""))

cells.append(md("""
## 4. Helper code
This code does three jobs:
- It asks the model to write a function.
- It pulls the code out of the answer.
- It runs the code with the official tests in a separate process. The process stops after 10 seconds, so a bad loop cannot freeze my notebook.
"""))
cells.append(code("""
import re, json, subprocess, tempfile, os
from datasets import load_dataset

problems = load_dataset("openai/openai_humaneval", split="test")
print("HumanEval problems:", len(problems))

def ask_model(user_text, max_new_tokens=768):
    messages = [{"role": "user", "content": user_text}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = max_new_tokens,
        use_cache = True,
        do_sample = False,   # same answer each time, so the test is fair
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

def extract_code(answer):
    blocks = re.findall(r"```(?:python)?\\n(.*?)```", answer, re.DOTALL)
    return blocks[0] if blocks else answer

def run_tests(program, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.remove(path)
"""))

cells.append(md("""
## 5. Run the code test (HumanEval)
This step takes about 20 to 40 minutes. I get one result per problem.
"""))
cells.append(code("""
results = []
passed = 0

for i, p in enumerate(problems):
    prompt_text = (
        "Complete this Python function. Return the full function in one python code block.\\n\\n"
        + p["prompt"]
    )
    answer = ask_model(prompt_text)
    body = extract_code(answer)

    # If the model wrote the full function, use it. If not, add my prompt in front of it.
    if f"def {p['entry_point']}" in body:
        program = body
    else:
        program = p["prompt"] + body

    program += "\\n\\n" + p["test"] + f"\\n\\ncheck({p['entry_point']})\\n"
    ok = run_tests(program)
    passed += int(ok)
    results.append({"task_id": p["task_id"], "passed": ok})

    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{len(problems)} done. Pass so far: {passed}/{i + 1}")

pass_at_1 = passed / len(problems)
print(f"\\nFinal pass@1: {pass_at_1:.3f} ({passed}/{len(problems)})")
"""))

cells.append(md("""
## 6. Run the format test
I trained the model to write tool calls inside `<tool_call>` tags. In this test I check if the model does it. I also check if the text inside the tags is valid JSON. This is a simple format check. It does not prove the model solves tasks well.
"""))
cells.append(code("""
SYSTEM = (
    "You are eigentiki, an expert coding assistant. You have these tools: "
    "bash, read, write, edit. To use a tool, write a JSON list inside <tool_call> tags."
)
agent_tasks = [
    "List all Python files in the current folder.",
    "Show the first 10 lines of README.md.",
    "Create a file named hello.py that prints hello.",
    "Find the line that contains the word TODO in main.py.",
    "Run the tests in the tests folder.",
]

format_results = []
for task in agent_tasks:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=400, use_cache=True, do_sample=False)
    answer = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    match = re.search(r"<tool_call>\\s*(.*?)\\s*</tool_call>", answer, re.DOTALL)
    has_tag = match is not None
    valid_json = False
    if has_tag:
        try:
            json.loads(match.group(1))
            valid_json = True
        except Exception:
            valid_json = False
    format_results.append({"task": task, "has_tag": has_tag, "valid_json": valid_json, "answer": answer})
    print(f"Task: {task}\\n  tag: {has_tag}, valid JSON: {valid_json}\\n")

tag_rate = sum(r["has_tag"] for r in format_results) / len(format_results)
json_rate = sum(r["valid_json"] for r in format_results) / len(format_results)
print(f"Tool tag rate: {tag_rate:.2f}")
print(f"Valid JSON rate: {json_rate:.2f}")
"""))

cells.append(md("""
## 7. Save the results
I save the numbers in a file. I push the file to the Hub, next to the model. The file name includes the model name, so I do not mix up the base run and the trained run.
"""))
cells.append(code("""
from datetime import datetime
from huggingface_hub import HfApi

report = {
    "date": datetime.now().isoformat(),
    "model_tested": MODEL_NAME,
    "humaneval_pass_at_1": pass_at_1,
    "humaneval_passed": passed,
    "humaneval_total": len(problems),
    "format_tag_rate": tag_rate,
    "format_valid_json_rate": json_rate,
    "decoding": "greedy, max_new_tokens=768",
    "note": "This is a single-shot code test plus a small format check. It is not a full agent test.",
}

file_name = "eval_" + MODEL_NAME.replace("/", "_") + ".json"
with open(file_name, "w") as f:
    json.dump(report, f, indent=2)

HfApi(token=hf_token).upload_file(
    path_or_fileobj = file_name,
    path_in_repo = file_name,
    repo_id = "focustiki/eigentiki",
    repo_type = "model",
)
print(json.dumps(report, indent=2))
print("I pushed the report to https://huggingface.co/focustiki/eigentiki")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
with open("notebooks/eval_agent.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote notebooks/eval_agent.ipynb")
