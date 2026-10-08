"""
Inspect AI test for eigentiki (HumanEval, with real unit tests).

What this file does:
1. It loads the 164 HumanEval problems.
2. It asks the model to write each function.
3. It pulls the code out of the answer.
4. It runs the official unit tests in a sandbox.
5. The score is the share of problems that pass (pass@1).

How to run (Hugging Face model, needs a GPU and the peft package):
    pip install inspect-ai transformers peft accelerate torch
    inspect eval scripts/evaluate_agent.py --model hf/focustiki/eigentiki --limit 10

To use Docker for the sandbox (safer for unknown code):
    inspect eval scripts/evaluate_agent.py -T sandbox=docker --model hf/focustiki/eigentiki

Note on what this test does not cover:
This test checks single functions. It does not test long agent work with many tool steps.
"""

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import ExecResult, sandbox as get_sandbox

# Seconds I allow for the tests of one problem. This stops endless loops.
VERIFY_TIMEOUT = 30

SYSTEM_PROMPT = (
    "You are eigentiki, an expert coding assistant. "
    "Write correct Python code. Return the full function in one python code block."
)

INSTRUCTION = (
    "Complete the Python function below. "
    "Return the full function in one python code block.\n\n"
)


def record_to_sample(record: dict) -> Sample:
    """Change one HumanEval row into one Inspect sample."""
    return Sample(
        id=record["task_id"],
        input=INSTRUCTION + record["prompt"],
        target=record["canonical_solution"],
        metadata={
            "prompt": record["prompt"],
            "test": record["test"],
            "entry_point": record["entry_point"],
        },
    )


def find_code(completion: str) -> str:
    """Pull the code out of the model answer.

    I use the first fenced code block. If there is no block, I use the full answer.
    """
    blocks = re.findall(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    return blocks[0] if blocks else completion


@scorer(metrics=[accuracy(), stderr()])
def verify():
    """Run the official unit tests on the code from the model."""

    async def score(state: TaskState, target: Target) -> Score:
        answer = find_code(state.output.completion)
        meta = state.metadata

        # If the model wrote the full function, I use it as it is.
        # If not, I put the original prompt in front of the answer.
        if f"def {meta['entry_point']}" in answer:
            body = answer
        else:
            body = meta["prompt"] + answer

        program = "".join(
            [
                body,
                "\n\n",
                meta["test"],
                "\n\n",
                f"check({meta['entry_point']})\n",
            ]
        )

        try:
            result: ExecResult[str] = await get_sandbox().exec(
                cmd=["python", "-c", program],
                timeout=VERIFY_TIMEOUT,
            )
            passed = result.success
            explanation = (
                "All tests passed." if passed else f"Tests failed:\n{result.stderr}"
            )
        except TimeoutError:
            passed = False
            explanation = "The tests took too long."

        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=answer,
            explanation=explanation,
        )

    return score


@task
def eigentiki_humaneval(sandbox: str = "local") -> Task:
    """The HumanEval test for eigentiki.

    sandbox: "local" runs the tests on this computer.
             "docker" runs the tests in a Docker container.
    """
    return Task(
        dataset=hf_dataset(
            path="openai/openai_humaneval",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=verify(),
        sandbox=sandbox,
    )
