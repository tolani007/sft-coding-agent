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
