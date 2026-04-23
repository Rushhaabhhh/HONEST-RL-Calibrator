---
title: HONEST Env
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# HONEST-RL-Calibrator

HONEST, short for Honesty-Optimised and Normalized Environment for Self-Triage, is an OpenEnv-compliant reinforcement learning and evaluation environment designed to test and improve the **honesty and confidence calibration** of AI agents (LLMs).

Instead of just checking if an agent gets the answer right, HONEST-RL-Calibrator forces the agent to report its confidence alongside its answer, or explicitly abstain from answering. It then scores the agent using a proper scoring rule (Brier Score) combined with adaptive curriculum difficulty.

## How It Works

### 1. The Generators (Domains)
The environment procedurally generates questions across three domains, each with 5 difficulty levels:
- **Math (`math_gen.py`)**: Ranging from simple arithmetic addition to compound interest and quadratics.
- **Python Code (`code_gen.py`)**: Ranging from simple variable execution to nested loops and bounded recursion. Answers are dynamically evaluated using an isolated `exec()` context.
- **Logic Puzzles (`logic_gen.py`)**: Setup using `python-constraint`. Ranging from simple transitive relations (A > B > C) to 4x4 Zebra puzzles, guaranteed to have exactly one unique solution.

### 2. Action Interface
The agent interacts with the environment by supplying an XML format specifying its answer and a confidence score between 0.0 and 1.0 (e.g., `<answer>42</answer><confidence>0.9</confidence>`), or it can explicitly abstain (`<abstain/>`).

### 3. The Reward Scheme (Brier Score)
The evaluation is handled by `verifier.py` and `reward.py`. 
- **Correct + High Confidence**: Slight positive reward (e.g., `~ +0.02`).
- **Wrong + High Confidence**: Heavy penalty (e.g., `~ -0.8`). The model is punished for being overconfident.
- **Wrong + Low Confidence**: Treated equivalently to correct + high confidence because the model accurately calibrated its own uncertainty.
- **Abstain**: Gives a slight penalty on easier questions, but zero penalty on extremely hard questions.
- **Malformed Action**: Returns a fixed penalty.

### 4. Adaptive Difficulty Engine
Controlled by `server/difficulty.py`, the environment manages a sliding window of the past 20 episodes for each domain. 
- **Increase**: If the agent's rolling accuracy on a domain exceeds 70%, the difficulty increments (up to 5).
- **Decrease**: If the accuracy falls below 30%, the difficulty decrements (down to 1).
- **Hysteresis**: Ensures that difficulty does not rapidly oscillate by enforcing a cooldown of 10 episodes between changes.

### 5. OpenEnv Server & Client Integration
- **Server (`server/app.py`, `server/environment.py`)**: The `HonestEnvironment` scales to parallel instances, managing randomized episodes of 5 steps each. It is served using FastAPI, making it accessible via HTTP/WebSockets.
- **Client (`client/client.py`)**: An asynchronous `EnvClient` wrapper (`HonestEnv`) which enables remote test runners and evaluation pipelines to interface seamlessly with the backend using standard OpenEnv APIs.

## Running

You can run the environment natively using:

```bash
# Setup your virtual environment
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Run the local server using our run script
./run_server.sh
```

Or using Docker (ready for Hugging Face Spaces):

```bash
# Build the Docker image
docker build -t honest-rl-calibrator:latest .

# Run the container locally mapped to port 8000
docker run -p 8000:8000 -d --name honest-test honest-rl-calibrator:latest
```

## Testing
We have achieved 100% test coverage across generators, logic uniqueness verification, rewards, difficulty engines, state machines, and the server-client WebSocket integration testing.

```bash
./venv/bin/pytest tests/ -v
```
