#!/bin/bash
echo "Starting evaluation..."
# Run 5 samples to keep it fast
inspect eval evaluate_agent.py --model vllm/focustiki/eigentiki --limit 5
echo "Evaluation finished. Starting viewer..."
# Start the Inspect View web interface
inspect view --port 7860 --host 0.0.0.0
