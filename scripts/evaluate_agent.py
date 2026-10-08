"""
Inspect AI Evaluation Script for eigentiki

This script sets up a professional agentic evaluation using Inspect AI.
It gives the agent access to a bash terminal and Python execution,
then asks it to solve coding problems from the HumanEval dataset.

To run this locally or on a Hugging Face Job (requires GPU):
1. pip install inspect-ai vllm
2. inspect eval scripts/evaluate_agent.py --model vllm/focustiki/eigentiki --limit 10
"""

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import bash, python

# Map the HuggingFace dataset to Inspect's format
def record_to_sample(record):
    return {
        "input": record["prompt"],
        "target": record["canonical_solution"],
        "metadata": {"task_id": record["task_id"]}
    }

@task
def eigentiki_humaneval():
    return Task(
        dataset=hf_dataset(
            path="openai_humaneval",
            split="test",
            sample_fields=record_to_sample
        ),
        plan=[
            system_message(
                "You are eigentiki, an expert coding assistant. "
                "You have access to a bash terminal and a Python interpreter. "
                "Read the coding problem, use your tools to write and test the code, "
                "and then return the final correct Python function."
            ),
            use_tools([bash(), python()]),
            generate()
        ],
        scorer=match(),
    )
