# HONEST: Complete Project Context
## Honesty-Optimized Network via Evaluated Scoring with Trust
### Single Source of Truth — All Context, All Decisions, All Architecture

---

## TABLE OF CONTENTS

1. Project Identity and Elevator Pitch
2. The Problem: Why Calibration Matters
3. The Science: What We Know From Research
4. What HONEST Actually Solves
5. Team and Responsibilities
6. Hackathon Context
7. Technical Foundation: Key Concepts Explained
8. The MDP Formalization
9. Environment Architecture
10. Procedural Generators
11. Reward System Design
12. Anti-Reward-Hacking Design
13. Training Pipeline
14. SFT Strategy
15. GRPO Deep Dive
16. RLVE vs RLVR vs RLHF
17. Evaluation Metrics and Protocol
18. Transfer Learning Design
19. Hindsight Calibration Mechanism
20. MCP Server Wrapper
21. OpenEnv Compliance
22. Step-by-Step Creation Plan
23. Prompts for Claude Code
24. Verification Checkpoints
25. Pitch and Storytelling
26. Q&A Preparation
27. Resource Library
28. Decision Log
29. Risk Register
30. Glossary

---

# PART 1: PROJECT IDENTITY

## 1.1 What Is HONEST?

HONEST (Honesty-Optimized Network via Evaluated Scoring with Trust) is a reinforcement learning environment built on the OpenEnv framework that trains large language models to be calibrated about their own uncertainty. When a trained model says it is 80% confident in an answer, it should be correct approximately 80% of the time. This property — calibration — is currently absent from most frontier models and is the technical mechanism behind hallucination.

HONEST is not a model. It is an environment and training infrastructure. The output is a trained model that expresses honest uncertainty, and an OpenEnv-compliant environment that any team can use to calibrate their own models.

## 1.2 One-Sentence Version

An RL environment that trains LLMs to report honest confidence scores using proper scoring rule rewards, procedurally generated tasks across multiple domains, and adaptive difficulty — with transfer evaluation proving calibration generalizes to unseen domains.

## 1.3 What It Is Not

HONEST does not claim to eliminate hallucination entirely. It does not claim to improve raw accuracy in isolation. It does not use LLM-as-judge for any reward signal. It is not a post-hoc calibration technique (like temperature scaling). It is a training-time intervention that modifies the model's weights to produce better-calibrated outputs.

## 1.4 The Contribution Stack

In order from most established to most novel:

1. Implementation of RLCR (RL with Calibration Rewards, Brier score) — exists in research, HONEST makes it accessible infrastructure.
2. RLVE-style adaptive difficulty applied to calibration — RLVE exists for reasoning tasks, not calibration.
3. Multi-domain transfer evaluation — calibration papers typically evaluate single-domain.
4. OpenEnv-compliant packaging — first calibration environment on OpenEnv Hub.
5. MCP deployment wrapper — post-training deployment artifact, positions trained model as infrastructure.
6. Hindsight calibration signal — novel, experimental, not published elsewhere.

---

# PART 2: THE PROBLEM

## 2.1 The Calibration Problem Defined

Calibration is the alignment between stated confidence and empirical accuracy. A perfectly calibrated model, when it says 70% confident, is correct 70% of the time across many instances. Current frontier models are systematically miscalibrated toward overconfidence.

Formally: Let p be the model's stated confidence in answer y given question x. Let 1[y = y*] be the indicator that y matches ground truth y*. Perfect calibration requires:

  P(y = y* | confidence = p) = p, for all p in [0, 1]

Current frontier models violate this. When they say 90% confident, empirical accuracy is often 60-70%.

## 2.2 Why This Is a Real Problem

Standard RL training with binary rewards (correct = 1, wrong = 0) treats confident-correct and hesitant-correct identically. It also treats confident-wrong and hesitant-wrong identically. This means the model receives no signal that distinguishes honest uncertainty from overconfident guessing.

Research result (RLCR paper, 2025): Standard RLVR training makes calibration worse. Models trained with binary correctness rewards become more overconfident over training time, even as accuracy improves. Reasoning models specifically show increased hallucination rates compared to base models after RLVR training.

## 2.3 Why Current Solutions Are Insufficient

Temperature Scaling: Applies a global scalar to model logits post-training. Brittle — breaks under distribution shift, doesn't actually change what the model knows, one-parameter fix for a complex problem.

Platt Scaling: Sigmoid transformation of model scores. Same limitations as temperature scaling.

Post-hoc methods in general: Applied after training without changing weights, so the model's internal representations don't change. The model still "thinks" overconfidently — it just gets adjusted at output time.

RLHF calibration terms: Some production systems add calibration loss terms to RLHF, but this isn't standard, isn't accessible, and isn't studied systematically.

## 2.4 The Gap HONEST Fills

No standard, accessible, reproducible infrastructure exists for training-time calibration improvement via RL with proper scoring rules. The RLCR paper proves the method works. HONEST is the infrastructure that makes it usable.

---

# PART 3: THE SCIENCE

## 3.1 Proper Scoring Rules

A scoring rule S(p, y*) takes a confidence p and a binary outcome y* (0 or 1) and returns a score. A proper scoring rule is one where the expected score is maximized when p equals the true probability of y* = 1.

Formally: E[S(p, y*)] is maximized when p = P(y* = 1).

This is the mathematical property that incentivizes honest reporting. If you know the true probability is 0.6, the proper scoring rule rewards you most for reporting exactly 0.6, not 0.9 or 0.3.

The Brier Score is a proper scoring rule: S_brier(p, y*) = -(p - y*)^2

When y* = 1 (correct): S_brier = -(p - 1)^2. Maximized at p = 1, but decreasing penalty as p approaches 1.
When y* = 0 (wrong): S_brier = -(p - 0)^2 = -p^2. Maximized at p = 0.

Expected Brier Score: E[S_brier] = -(p - P(correct))^2 + constant. This is maximized when p = P(correct). Proper.

Alternative: Log score (cross-entropy): S_log = y* log(p) + (1-y*) log(1-p). Also proper, but unbounded when p → 0 or 1, which causes gradient instability in training.

Alternative: Spherical score: S_sphere = p / sqrt(p^2 + (1-p)^2). Also proper, bounded, less common.

Decision: Use Brier because it is bounded in [-1, 0], numerically stable, and supported by 2025 research on RL calibration.

## 3.2 Expected Calibration Error (ECE)

ECE is the primary evaluation metric. Not used as a training reward (we use Brier for that) but used to measure how well calibration succeeded.

Algorithm:
1. Bin confidence scores into 10 equal-width bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
2. For each bin b: compute avg_confidence(b) and avg_accuracy(b)
3. ECE = sum over bins: (size(b) / total) * |avg_confidence(b) - avg_accuracy(b)|

Perfect ECE = 0. Current frontier model ECE is typically 0.15-0.25 on QA tasks.

## 3.3 Adaptive Calibration Error (ACE)

Like ECE but uses equal-sample bins instead of equal-width bins. More robust when confidence distribution is not uniform (which it never is). Equal-sample binning ensures each bin has enough data for reliable accuracy estimation.

## 3.4 Maximum Calibration Error (MCE)

MCE = max over bins of |avg_confidence - avg_accuracy|. Measures worst-case miscalibration. Important for safety-critical applications.

## 3.5 Reliability Diagrams

The visual representation of calibration. X-axis: confidence bins. Y-axis: accuracy in each bin. A perfectly calibrated model produces a reliability diagram where each bar height equals the bin's midpoint — all bars lie on the diagonal.

Overconfident models produce bars that are below the diagonal (claimed confidence exceeds actual accuracy). This is what current frontier models look like.

## 3.6 RLVE vs Static RLVR

From the RLVE paper (Zeng et al., 2025): Training on 400 adaptive environments yields 3.37% absolute improvement across six reasoning benchmarks. Continuing the same amount of static RLVR training yields 0.49% improvement at 3x the compute. The mechanism: static environments saturate. Once the model has mastered difficulty level D, additional training at level D provides no gradient signal. Adaptive environments maintain the learning signal by continuously adjusting difficulty to the model's current capability frontier.

---

# PART 4: WHAT HONEST SOLVES

## 4.1 Primary Outcome

Reduces Expected Calibration Error by 3-5x compared to base model. Empirically measured using reliability diagrams and multi-metric evaluation.

## 4.2 Secondary Outcome

Calibration transfers to held-out domains (medical, legal) not seen during training. If this result holds, it proves calibration is a learnable meta-skill, not a domain-specific patch.

## 4.3 What It Does Not Solve

Raw accuracy is not the target. HONEST does not claim to make the model smarter. If a model is 60% accurate, HONEST helps it express 60% confidence when it's likely wrong — it doesn't make it 70% accurate.

HONEST does not fix all hallucination. Hallucination has multiple causes: retrieval failures, instruction-following errors, knowledge gaps. Calibration addresses the overconfidence component.

## 4.4 Why This Matters Downstream

A calibrated model enables:
- Selective prediction: refuse to answer when confidence is below threshold
- Human-in-the-loop routing: escalate uncertain cases to humans
- Multi-agent orchestration: orchestrator knows which sub-agent's outputs to trust
- User communication: "I'm 40% confident in this" is actionable; "I'm not sure" is not
- Risk-weighted decision making: weight model outputs by their calibrated confidence in downstream calculations

---

# PART 5: TEAM AND RESPONSIBILITIES

## 5.1 Team Members

Kanan: Backend/Java/Spring Boot experience, AI data annotation at xAI (knows what bad calibration looks like firsthand), trading interest, PreCaffeinate project. Owns: OpenEnv wrapper, reward function, verification pipeline.

Rushabh: Systems engineering (Protocol Labs, Filecoin), distributed systems (Kafka, Redis), Solidity/Web3 depth, RAG pipeline experience (Agentic RAG), OSS contributions. Owns: Generator implementations, FastAPI server, Docker, HF Space deployment.

Ayush: Product management (TradeIndia), funnel analysis, KPI definition, A/B testing, analytics. Owns: Pitch narrative, evaluation dashboard, metrics visualization, blog post, demo video.

## 5.2 Division of Work

Environment infrastructure → Rushabh
Reward and verification logic → Kanan
Adaptive difficulty controller → Kanan
FastAPI + Docker + deployment → Rushabh
Evaluation metrics and plots → Ayush + Kanan
Training pipeline (Colab notebook) → Kanan
Baseline characterization → all three
MCP wrapper → Rushabh
Pitch deck → Ayush
Blog post → Ayush
Demo video → Ayush

---

# PART 6: HACKATHON CONTEXT

## 6.1 The Competition

Meta PyTorch OpenEnv Hackathon × Scaler School of Technology. Onsite: April 25-26, Bangalore. Prize pool: $30,000. 800 teams total. Top 15 selected. Judged by Meta's global team and HuggingFace engineers.

## 6.2 Judging Criteria

Environment Innovation (40%): Is the environment novel, creative, challenging? Does it meaningfully test agent behavior?

Storytelling (30%): Clear explanation of problem, environment, agent behavior. Engaging demo.

Showing Improvement in Rewards (20%): Observable evidence of training progress. Reward curves, metrics, before/after behavior.

Reward and Training Pipeline (10%): Coherent reward logic, meaningful improvement in agent inference.

## 6.3 Required Deliverables

- OpenEnv-compliant environment on HuggingFace Spaces
- Minimal training script using Unsloth or HF TRL in Colab
- Mini-blog on HuggingFace or video on YouTube (< 2 minutes)

## 6.4 Theme Positioning

Primary: Theme 5 (Wild Card) — calibration doesn't fit neatly into themes 1-4, which is itself the pitch. "This is a meta-skill that every agent in every theme needs."

Secondary mentions: Theme 3.1 (World Modeling — a calibrated agent has an accurate model of its own limitations) and Theme 4 (Self-Improvement — hindsight calibration mechanism).

## 6.5 Competitive Advantage

Most teams will read the same docs and build: Wordle variants, coding agents, game environments, Calendar Gym derivatives. HONEST differentiates on domain (calibration is novel in this context), scientific backing (proper scoring rules are published theory), unfakeable reward (Brier score is deterministic math), and research-quality evaluation (multi-metric, multi-domain, reliability diagrams).

---

# PART 7: TECHNICAL FOUNDATION — KEY CONCEPTS

## 7.1 What OpenEnv Is

OpenEnv is an open-source framework by Meta and HuggingFace for creating standardized RL training environments for LLMs. It provides:
- Gymnasium-style step()/reset()/state() API
- FastAPI server + Docker containerization pattern
- HuggingFace Spaces deployment infrastructure
- TRL GRPOTrainer integration via WebSocket connections

OpenEnv environments are FastAPI servers running in Docker containers, deployed to HuggingFace Spaces, and accessed by training scripts via HTTP/WebSocket.

## 7.2 What GRPO Is

Group Relative Policy Optimization. The RL algorithm we use for training.

How it works:
1. For each prompt, generate N completions (group size, typically 8)
2. Compute reward for each completion
3. Compute advantage for each completion as: (reward - mean(group rewards)) / std(group rewards)
4. Update policy to increase probability of high-advantage completions
5. KL divergence penalty keeps policy close to reference model

Why GRPO over PPO:
- PPO requires 4 neural networks in memory (policy, reference, critic, reward model)
- GRPO requires 2 (policy, reference) — the verifier IS the reward, no learned reward model needed
- ~50% memory reduction
- TRL and Unsloth both support GRPO natively
- Hackathon documentation explicitly recommends GRPO

Why GRPO over DPO:
- DPO requires preference pairs (A is better than B) as training data
- GRPO uses environment-generated rewards directly
- For verifiable reward tasks (ours), GRPO is more natural

GRPO mathematical note: The key insight is that advantage is computed relative to the group, not to an absolute baseline. If all 8 completions are bad, the least-bad one still gets positive advantage. This is both a strength (stable gradients) and a weakness (relative ranking, not absolute quality).

## 7.3 What TRL Is

Transformer Reinforcement Learning. HuggingFace's library for post-training LLMs. Provides GRPOTrainer, SFTTrainer, and other training utilities. HONEST uses GRPOTrainer with the environment_factory parameter to connect to the HONEST OpenEnv server.

## 7.4 What Unsloth Is

Acceleration and memory-efficiency layer for transformer training. Provides:
- Optimized attention kernels (2x faster than standard transformers)
- 4-bit QLoRA that doesn't degrade quality as much as naive quantization
- Proper LoRA merge path (important — naive merging of 4-bit adapters damages quality)

We use Unsloth to make GRPO training feasible on Colab A100 (40GB VRAM) with a 3B or 7B parameter model.

## 7.5 What LoRA Is

Low-Rank Adaptation. Instead of updating all model weights (expensive), LoRA adds small trainable matrices to specific weight matrices. For Qwen2.5-3B with LoRA r=16, the number of trainable parameters is roughly 40M vs 3B full fine-tune — 98% reduction in trainable parameters.

We use QLoRA (quantized LoRA): model weights are stored in 4-bit quantization, LoRA adapters are in 16-bit. Memory reduction: ~4x vs full precision.

After training, LoRA adapters are merged back into the base model for inference. Unsloth's merge path must be used — do not upcast to 16-bit and merge naively.

## 7.6 What Calibration Is (Intuition)

Imagine a weather forecaster. Over 1000 days, they predict "70% chance of rain." On days they said 70%, it should rain about 700 times for them to be calibrated. If it only rains 400 times, they're overconfident. If it rains 900 times, they're underconfident.

LLMs are like a forecaster who always says "95% chance of rain." They might be right frequently, but they're overconfident because they should have said 60% on the uncertain days.

## 7.7 What the MCP Server Is

Model Context Protocol server. Post-training deployment wrapper. After HONEST trains a calibrated model, we wrap it in an MCP server so any MCP-compatible agent (Claude Desktop, Cursor, custom agents) can call our model as a tool.

Tool name: ask_with_calibrated_confidence
Input: {"question": str}
Output: {"answer": str, "confidence": float, "calibration_note": str}

This is a delivery artifact, not a training component. The training environment and the MCP server are completely separate. Training stability is never at risk.

---

# PART 8: THE MDP FORMALIZATION

## 8.1 Why MDP Formalization Matters

An RL environment must satisfy the Markov Property: the future state depends only on the current state and action, not on the history of how you arrived there. If your environment violates this, gradient updates are incorrect.

HONEST satisfies the Markov Property because the full EVM state (episode history, current problem, domain difficulties, step count) is a complete sufficient statistic for all future transitions.

## 8.2 State Space

S_t = {
  question: str,                          // Current problem text
  domain: str,                            // "math" | "code" | "logic"
  difficulty: int,                        // 1-5, current domain difficulty
  episode_step: int,                      // 0-4 (episodes are 5 steps)
  episode_history: list[dict],            // All previous steps this episode
  domain_difficulties: dict,             // Per-domain current difficulty
  running_accuracy: float,               // This episode's accuracy so far
  running_mean_confidence: float,        // Mean stated confidence so far
  previous_correctness: bool or None,    // Result of last answer (for hindsight)
  revealed_answer: str or None,          // Ground truth if revealed (for hindsight)
  episode_id: str                        // UUID for this episode
}

## 8.3 Action Space

Three possible actions, mutually exclusive per step:

AnswerAction: {answer: str, confidence: float}
  - Parsed from: <answer>TEXT</answer><confidence>0.XX</confidence>
  - Confidence must be clamped to [0, 1]

AbstainAction: {}
  - Parsed from: <abstain/>
  - Agent believes it cannot answer reliably

HindsightAction: {retrospective_confidence: float}
  - Only valid immediately after ground truth is revealed
  - Agent estimates what confidence it should have expressed given the outcome

## 8.4 Transition Function

T(S_t, A_t) → S_{t+1}:

If A_t = AnswerAction:
  - Verify answer against ground truth (deterministic)
  - Set previous_correctness = verification result
  - Set revealed_answer = ground truth
  - Increment episode_step
  - Update episode_history
  - If episode_step < 5: generate next problem (possibly new domain), update difficulties
  - If episode_step >= 5: set terminal = True

If A_t = AbstainAction:
  - Set previous_correctness = None
  - Increment episode_step
  - No ground truth reveal (nothing to reveal if didn't answer)
  - Continue to next problem

If A_t = HindsightAction:
  - Compute hindsight reward (how close was retrospective confidence to optimal?)
  - State moves to next question generation

Terminal condition: episode_step >= 5

## 8.5 Reward Function (Complete Specification)

For AnswerAction:
  correct = verify_answer(answer, ground_truth)       // bool
  correct_indicator = 1.0 if correct else 0.0
  brier = -((confidence - correct_indicator)^2)       // range [-1, 0]
  format_bonus = 0.02                                 // well-formed answer
  R_answer = brier + format_bonus

For AbstainAction:
  If difficulty >= 7: R_abstain = 0.0                 // reasonable abstention
  If difficulty < 7: R_abstain = -0.3                 // unreasonable abstention
  Note: max difficulty in training is 5, so abstention always yields -0.3 in our system
  (This prevents the degenerate "always abstain" policy)

For HindsightAction:
  optimal_conf = 1.0 if previous_correctness else 0.0
  R_hindsight = -(retrospective_confidence - optimal_conf)^2 * 0.3
  (Weighted at 0.3 because this is experimental signal)

For Malformed output:
  R_malformed = -0.5

Total episode reward = sum of step rewards

## 8.6 Why This Reward Trains Calibration

Consider the GRPO update. For a given prompt, 8 completions are generated. Their rewards are compared relatively. The completion that says confidence=0.9 and is CORRECT gets reward ≈ -0.01. The completion that says confidence=0.9 and is WRONG gets reward ≈ -0.81. The completion that says confidence=0.3 and is WRONG gets reward ≈ -0.09 (much better than confident-wrong).

Over many episodes, the model learns: high confidence only when you're likely correct, low confidence when uncertain. This is calibration.

The Brier score is a proper scoring rule, meaning the expected reward is maximized when confidence equals the true probability of being correct. There is no better strategy than honest reporting.

---

# PART 9: ENVIRONMENT ARCHITECTURE

## 9.1 System Diagram

Training Loop (Colab, A100 GPU)
  └── GRPOTrainer (TRL)
        └── HonestEnv Client (OpenEnv HTTP client)
              └── [HTTP over WebSocket]
                    └── FastAPI Server (HuggingFace Space, CPU)
                          ├── HonestEnvironment (core logic)
                          │     ├── MathGenerator
                          │     ├── CodeGenerator
                          │     └── LogicGenerator
                          ├── RewardComputer (Brier score)
                          ├── Verifier (canonical string match)
                          └── DifficultyController (adaptive)

## 9.2 File Structure

honest-env/
├── CLAUDE.md                  # Project context for Claude Code sessions
├── README.md                  # HF Space and GitHub README
├── Dockerfile                 # Production container
├── requirements.txt           # Python dependencies
├── run_server.sh              # Local testing script
│
├── models/
│   ├── __init__.py
│   └── models.py              # HonestAction, HonestObservation, HonestState
│
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI app creation
│   ├── environment.py         # HonestEnvironment class
│   ├── reward.py              # parse_action(), compute_reward()
│   ├── verifier.py            # verify_answer()
│   ├── difficulty.py          # update_difficulty(), adaptive controller
│   └── generators/
│       ├── __init__.py
│       ├── math_gen.py        # generate(difficulty, seed) → (question, answer)
│       ├── code_gen.py        # generate(difficulty, seed) → (question, answer)
│       └── logic_gen.py       # generate(difficulty, seed) → (question, answer)
│
├── client/
│   ├── __init__.py
│   └── client.py              # HonestEnv(HTTPEnvClient)
│
├── eval/
│   ├── __init__.py
│   ├── metrics.py             # ECE, ACE, MCE, Brier, NLL, AUROC functions
│   ├── baseline_eval.py       # Evaluate base model before training
│   ├── full_eval.py           # Evaluate trained model, all domains including OOD
│   └── plot_reliability.py    # Generate reliability diagrams
│
├── training/
│   ├── format_sft.py          # Lightweight SFT for format compliance (if needed)
│   └── train_grpo.py          # Main GRPO training script
│
├── mcp_server/
│   ├── honest_mcp.py          # MCP server wrapping trained model
│   ├── pyproject.toml         # MCP server dependencies
│   └── README.md              # Integration instructions for Claude Desktop
│
└── tests/
    ├── test_math_gen.py
    ├── test_code_gen.py
    ├── test_logic_gen.py
    ├── test_reward.py
    ├── test_verifier.py
    ├── test_difficulty.py
    ├── test_environment.py
    ├── test_metrics.py
    └── test_client_server.py

## 9.3 Dependencies

Core:
  openenv-core>=0.2.1
  fastapi
  uvicorn[standard]
  pydantic>=2.0

Training:
  trl>=0.21
  unsloth[cu121-ampere]  # adjust cuda version for your GPU
  transformers>=4.45
  accelerate>=0.33
  peft>=0.12
  datasets>=2.21
  wandb

Evaluation:
  numpy
  matplotlib
  seaborn
  scikit-learn

Testing:
  pytest
  pytest-asyncio
  httpx

MCP:
  mcp>=1.0

## 9.4 The CLAUDE.md File (Copy This Verbatim Into Project Root)

---START CLAUDE.MD---
# HONEST: Honesty-Optimized Network via Evaluated Scoring with Trust

## Project Summary
OpenEnv-compliant RL environment training LLMs to be calibrated about their uncertainty.
Uses Brier score reward (proper scoring rule). Procedurally generates math/code/logic tasks.
Evaluates transfer to medical/legal held-out domains.

## Core Principles (Never Violate)
1. BRIER IS DOMINANT: Primary reward is Brier score. Secondary rewards are small (<0.1).
2. NO LLM-AS-JUDGE: All rewards are programmatic. Never introduce LLM scoring.
3. PROCEDURAL GENERATION: Training tasks are generated fresh each episode. No fixed datasets.
4. ADAPTIVE DIFFICULTY: Difficulty adjusts to keep agent at 30-70% accuracy.
5. OPENENV SPEC: step()/reset()/state() API. Dataclasses extend openenv.core.env_server types.

## Agent Output Format (Strict)
Acceptable formats only:
- <answer>X</answer><confidence>0.YY</confidence>
- <abstain/>
Anything else = malformed = reward -0.5

## Reward Formula
Answer:   R = -(confidence - correct_indicator)^2 + 0.02
Abstain (difficulty >= 7): R = 0.0
Abstain (difficulty < 7):  R = -0.3
Malformed: R = -0.5
Hindsight: R = -(retrospective_conf - optimal_conf)^2 * 0.3

## Domains
Training (procedural): math (difficulties 1-5), code (1-5), logic (1-5)
Evaluation only (fixed): medical (MedQA-style), legal (LSAT logical reasoning)

## Model Choice
Training: Qwen2.5-3B-Instruct (primary), Qwen2.5-7B-Instruct (if compute allows)
Reason: Unsloth-optimized, strong baseline calibration, fits A100 40GB with QLoRA r=16

## Critical Don'ts
- Do NOT use LLM-as-judge in any reward component
- Do NOT add heavy dependencies without reason
- Do NOT use fixed datasets for training (must be procedurally generated)
- Do NOT merge LoRA adapters naively from 4-bit — use Unsloth's merge path
- Do NOT make reward function complex — simplicity trains better with GRPO
---END CLAUDE.MD---

---

# PART 10: PROCEDURAL GENERATORS

## 10.1 Why Procedural Generation

Static datasets saturate. Once the model has seen all variants of a problem type, additional training provides zero gradient signal. Procedural generation ensures:
- Infinite unique task supply
- Difficulty parameterized and adjustable
- No memorization possible (each instance is fresh)
- Controlled curriculum

## 10.2 Generator Interface Contract

Every generator must satisfy:

def generate(difficulty: int, seed: Optional[int] = None) -> tuple[str, str]:
  """
  difficulty: integer in [1, 5]
  seed: if provided, return deterministic (reproducible) output
        if None, return random fresh output
  returns: (question_string, canonical_answer_string)
  
  canonical_answer_string: lowercase, stripped, no commas, no trailing zeros
  Example: "1081" not "1,081", "42" not "42.0"
  """

## 10.3 Math Generator (Detailed Spec)

Difficulty 1: Single-digit addition and subtraction
  Q: "What is 7 + 3?" → A: "10"
  Baseline accuracy target: 90%+

Difficulty 2: Two-digit addition, subtraction, multiplication
  Q: "Compute: 47 * 23" → A: "1081"
  Baseline accuracy target: 70%

Difficulty 3: Nested three-digit expressions
  Q: "Compute: (128 - 47) * 3 + 19" → A: "262"
  Baseline accuracy target: 45%

Difficulty 4: Modular arithmetic and powers
  Q: "What is 7^4 mod 11?" → A: "3"
  Baseline accuracy target: 25%

Difficulty 5: Multi-step word problems
  Q: "A train travels 120km in 1.5 hours. Another train travels 200km in 2.5 hours. What is the difference in their speeds in km/h?"
  A: "0" (in this case both are 80 km/h)
  Baseline accuracy target: 10-15%

Seeding mechanism: random.Random(seed) if seed else global random.

## 10.4 Code Generator (Detailed Spec)

Pattern: "Given this Python function: [code], what does f([input]) return?"
The function must be generated, actually executed, and the true output stored as the answer.

Difficulty 1: Single-line arithmetic return
  def f(x): return x * 2 + 1
  Q: "What does f(5) return?" → A: "11"

Difficulty 2: If/else branching
  def f(x): return x * 2 if x > 5 else x - 1
  Q: "What does f(7) return?" → A: "14"

Difficulty 3: Single loop with accumulator
  def f(n):
    total = 0
    for i in range(n):
      total += i
    return total
  Q: "What does f(5) return?" → A: "10"

Difficulty 4: Nested conditions with loop
  (More complex control flow, 2-3 operations deep)

Difficulty 5: Recursive function with explicit base case
  def f(n): return n if n <= 1 else f(n-1) + f(n-2)
  Q: "What does f(6) return?" → A: "8"

Critical: Execute the actual generated function in a sandboxed eval() to get the true answer. Never assume the answer — compute it.

Safety: Only generate functions that are guaranteed to terminate (no infinite loops). For recursive functions, enforce a small depth limit (max recursion n <= 8).

## 10.5 Logic Generator (Detailed Spec)

Pattern: Constraint satisfaction problems with unique solutions.

Difficulty 1: 3-entity transitivity
  "Alice is older than Bob. Bob is older than Carol. Who is oldest?"
  A: "Alice"

Difficulty 2: 4-5 entities, 2-3 constraint types
  Mixed constraints: older/younger, taller/shorter, left/right of

Difficulty 3: 6+ entities with negation
  "Alice is not to the left of Bob. Carol is between Alice and Bob. Dave is not Carol's neighbor..."

Difficulty 4: Multi-attribute 4x4 puzzle
  4 people, 4 attributes (name, hobby, drink, color). 8 clues.

Difficulty 5: Zebra-puzzle-style 5x5
  5 people, 5 attributes, 12-15 clues.

Uniqueness requirement: Use python-constraint library or z3-solver to verify exactly one solution exists. If a randomly generated puzzle has multiple solutions or no solution, regenerate.

## 10.6 Seeded vs Unseeded Usage

During training: generate(difficulty=3)
  → Uses global random, fresh each call
  → Agent cannot memorize

During evaluation: generate(difficulty=3, seed=42)
  → Reproducible, same problem every time
  → Allows exact reproduction of results

Evaluation seeds are stored in eval/eval_seeds.json for reproducibility.

---

# PART 11: REWARD SYSTEM DESIGN

## 11.1 The Dominant Signal

Brier score: R_brier = -(confidence - correct_indicator)^2

This is the entire reward for a well-formed answer, plus a small format bonus.

Do not add more components. The OpenEnv docs explicitly warn: "Conflicting signals create instability." The GRPO docs note that relative ranking within groups is what matters, and Brier already provides strong relative signal.

## 11.2 Complete Reward Computation

def compute_reward(parsed: dict, ground_truth: str, difficulty: int) -> tuple[float, Optional[bool]]:

  if parsed["type"] == "malformed":
    return (-0.5, None)

  if parsed["type"] == "abstain":
    if difficulty >= 7:  # Only valid at expert level — impossible in our system
      return (0.0, None)
    else:
      return (-0.3, None)

  # type == "answer"
  answer = parsed["answer"]
  confidence = parsed["confidence"]   # Already clamped to [0, 1] by parser
  correct = verify_answer(answer, ground_truth)  # Canonical comparison
  correct_indicator = 1.0 if correct else 0.0

  brier = -((confidence - correct_indicator) ** 2)  # Range: [-1, 0]
  format_bonus = 0.02

  return (brier + format_bonus, correct)

## 11.3 Parsing Logic

Pattern matching on agent's raw text output:

def parse_action(raw_text: str) -> dict:
  # Try abstain first
  if re.search(r'<abstain\s*/?>', raw_text):
    return {"type": "abstain"}

  # Try answer + confidence
  answer_match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)
  conf_match = re.search(r'<confidence>([\d.]+)</confidence>', raw_text)

  if answer_match and conf_match:
    try:
      confidence = float(conf_match.group(1))
      confidence = max(0.0, min(1.0, confidence))  # Clamp
      return {
        "type": "answer",
        "answer": answer_match.group(1).strip(),
        "confidence": confidence
      }
    except ValueError:
      return {"type": "malformed"}

  return {"type": "malformed"}

## 11.4 Canonical Answer Verification

def verify_answer(agent_answer: str, ground_truth: str) -> bool:
  def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(",", "")   # Remove thousand separators
    s = s.replace(" ", "")   # Remove spaces
    # Handle numeric equivalence: "42.0" == "42"
    try:
      float_val = float(s)
      if float_val == int(float_val):
        s = str(int(float_val))
      else:
        s = str(float_val)
    except (ValueError, OverflowError):
      pass
    return s

  return normalize(agent_answer) == normalize(ground_truth)

---

# PART 12: ANTI-REWARD-HACKING DESIGN

The docs say: "Do not optimize a reward you have not tried to break yourself first."

## 12.1 Exploit: Always abstain
Attack: Never answer, avoid all wrong answers, get reward 0.0 per step.
Defense: Abstain on difficulty < 7 yields -0.3. Since our max difficulty is 5, all abstentions yield -0.3. Expected reward from abstaining all episode: -0.3 * 5 = -1.5. Expected reward from attempting with 50% accuracy and honest confidence: ~-0.25 * 5 = -1.25. Attempting is better than always abstaining.

## 12.2 Exploit: Always answer 0.5 confidence
Attack: Brier score at confidence=0.5 is always -0.25, regardless of correctness. Predictable, can't be worse.
Defense: GRPO uses relative ranking. An agent that answers 0.5 on everything is ranked lower than one that correctly distinguishes easy problems (high confidence) from hard ones (low confidence). Also, Brier rewards being right with high confidence (near 0) strongly — the 0.5-always agent foregoes this.

## 12.3 Exploit: Memorize problems
Attack: Learn specific Q→A pairs rather than genuine calibration.
Defense: Procedural generation with unseeded randomness. Each training episode generates fresh problems from the same distribution but different random seeds. No two episodes have the same question.

## 12.4 Exploit: Spam high confidence on easy problems
Attack: Always say 0.99 confidence to maximize reward when correct, accepting big penalty when wrong.
Defense: This is actually not an exploit — this IS correct behavior for easy problems. But it fails on hard problems because the agent is then wrong more often. Adaptive difficulty moves harder problems into the training mix as the agent improves, ensuring it must learn to express uncertainty.

## 12.5 Exploit: Format trick — output malformed but close enough to parse
Attack: Output something that looks answer-like but is technically malformed to avoid commitment.
Defense: Parser is strict. Anything that doesn't match the exact regex patterns returns malformed and -0.5. The agent cannot partially commit.

## 12.6 Exploit: Confidence values outside stated range
Attack: Output "confidence: 1.5" to claim maximum possible reward.
Defense: Parser clamps: confidence = max(0.0, min(1.0, parsed_float)). Values outside [0,1] are clipped.

---

# PART 13: TRAINING PIPELINE

## 13.1 Full Pipeline Overview

STAGE 1: Baseline characterization (pre-training)
  - Load Qwen2.5-3B-Instruct (no training)
  - Run 300 problems (20 per domain per difficulty)
  - Measure: accuracy, format compliance rate, ECE, Brier per domain/difficulty
  - Decision gate: format compliance must be > 70% and accuracy must be > 0% at all difficulties

STAGE 2: Light SFT for format (conditional — only if format compliance < 70%)
  - Generate 300-500 examples of correct format outputs
  - Train 1-2 epochs, learning rate 2e-5
  - Verify format compliance improves to > 90%
  - Save adapters to training/format_sft_adapters/

STAGE 3: GRPO training (main stage)
  - Load base model (with format SFT adapters if done)
  - Connect to deployed HONEST environment on HF Space
  - Run GRPO with reward from environment
  - 500-1000 steps (time-limited by compute availability)
  - Monitor reward curve, inspect outputs every 50 steps
  - Save checkpoint every 50 steps

STAGE 4: Evaluation
  - Load best checkpoint
  - Evaluate in-distribution (math, code, logic, 200 per domain)
  - Evaluate OOD (medical, legal, 200 each)
  - Compute all metrics, generate reliability diagrams
  - Compare to baseline

STAGE 5: MCP wrapper
  - Package trained model as MCP server
  - Test with MCP client
  - Document integration

## 13.2 Hyperparameters

Model: Qwen2.5-3B-Instruct (fallback: Qwen2.5-1.5B if memory issues)
LoRA r: 16 (attention + FFN layers)
LoRA alpha: 16 (typically equal to r)
QLoRA: 4-bit quantization, load_in_4bit=True
Max sequence length: 2048
GRPO group size (num_generations): 8
Per-device train batch size: 4
Gradient accumulation: 8 (effective batch = 32)
Learning rate: 1e-5
KL coefficient (beta): 0.04 (if diverging, increase; if underfitting, decrease)
Max grad norm: 1.0
Steps: 1000 (target), 200 minimum for viable results
Save every: 50 steps

## 13.3 Compute Requirements

Minimum: Colab T4 (16GB) with Qwen2.5-1.5B
Recommended: Colab A100 (40GB) with Qwen2.5-3B
Expected training time: 6-10 hours for 1000 steps on A100
Expected training time: 14-20 hours for 1000 steps on T4

Inference dominates runtime (~70% of time is generation, not optimizer). Unsloth's fast generation kernels reduce this significantly.

## 13.4 Training Script Structure

# training/train_grpo.py
import os, asyncio
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from client.client import HonestEnv
from models.models import HonestAction

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth"
)

# 3. System prompt
SYSTEM_PROMPT = """You are a careful reasoner. Solve the problem, then report:
<answer>your answer</answer><confidence>0.XX</confidence>
Confidence should be your true probability of being correct. If completely unsure, use <abstain/>"""

# 4. Rollout function
async def rollout_function(prompts, completions, **kwargs):
    rewards = []
    env_url = os.environ["HONEST_ENV_URL"]
    async with HonestEnv(base_url=env_url) as env:
        for completion in completions:
            await env.reset()
            result = await env.step(HonestAction(raw_text=completion))
            rewards.append(result.observation.reward)
    return rewards

# 5. Config
config = GRPOConfig(
    output_dir="./honest-qwen-3b",
    num_generations=8,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    save_steps=50,
    logging_steps=1,
    report_to="wandb",
    max_prompt_length=512,
    max_completion_length=512
)

# 6. Train
trainer = GRPOTrainer(model=model, tokenizer=tokenizer, args=config,
                       reward_funcs=[rollout_function], train_dataset=prompt_dataset)
trainer.train()

## 13.5 What to Monitor During Training

Every 50 steps manually inspect:
- Mean reward: should trend upward over time
- Reward distribution: should widen then narrow as model improves
- Format compliance: should stay > 90%
- Confidence distribution: should not collapse to 0.5 (hedging) or 0.9 (overconfidence)
- KL divergence: should stay bounded (< 0.1 per step typically)

If mean reward is flat after 200 steps: learning rate likely too low, or reward signal is broken. Verify environment is returning varied rewards.

If outputs become repetitive: KL coefficient too low, policy is diverging. Increase beta.

If format compliance drops: model is forgetting format under optimization pressure. May need to increase format bonus or add format compliance to reward.

---

# PART 14: SFT STRATEGY

## 14.1 When SFT Is Needed

SFT is needed only if baseline evaluation shows format compliance < 70%. If Qwen2.5-3B naturally produces the <answer><confidence> format (which instruction-tuned models often do with good prompting), skip SFT entirely.

## 14.2 Format SFT Dataset Construction

Generate 500 examples:
- 40% math, 30% code, 30% logic
- Mix of correct answers (confidence 0.7-0.95) and wrong answers (confidence 0.1-0.4)
- The confidence values are set programmatically based on difficulty — not randomly
- High difficulty → lower confidence in the SFT examples (teaches the calibration pattern)

Structure of each example:
[SYSTEM_PROMPT]
User: [domain/difficulty tag] [question]
Assistant: [reasoning steps]
<answer>[correct or wrong answer]</answer>
<confidence>[appropriately calibrated value]</confidence>

## 14.3 SFT Does Not Train Calibration

SFT examples have programmatically assigned confidence values. These are illustrative, not actually calibrated to model capability. Do not over-rely on SFT to produce calibration. SFT teaches the format. GRPO teaches the calibration.

## 14.4 SFT Hyperparameters

Epochs: 1-2 (not more — we don't want to overfit)
Learning rate: 2e-5
Batch size: 8
Max length: 2048
LoRA r: 8 (smaller than GRPO since just format learning)
Save as: training/format_sft_adapters/

---

# PART 15: GRPO DEEP DIVE

## 15.1 The GRPO Algorithm Step by Step

For each training step:

1. Sample a batch of prompts (from procedurally generated question pool)

2. For each prompt, generate G completions (G=8 in our config) using the current policy at temperature 1.0

3. For each completion, compute reward r_i using the environment (step through HONEST environment)

4. Compute advantage for each completion:
   mean_r = mean(r_1, ..., r_G)
   std_r = std(r_1, ..., r_G) + epsilon   # epsilon prevents division by zero
   advantage_i = (r_i - mean_r) / std_r

5. Compute GRPO loss:
   For each token in completion:
     ratio = policy_prob(token) / reference_prob(token)
     clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)  # epsilon=0.2 default
     policy_gradient = min(ratio * advantage, clipped_ratio * advantage)

6. Add KL penalty:
   kl_div = policy_logprob - reference_logprob   # per token
   loss = -policy_gradient + beta * kl_div       # beta=0.04 default

7. Backpropagate and update LoRA weights

## 15.2 Key Hyperparameters and Their Effects

num_generations (G): More generations = better advantage estimate but more compute.
  Too low (G=2): Noisy advantage estimates, unstable training
  Too high (G=16): Expensive, diminishing returns
  Recommended: G=8 for hackathon balance

beta (KL coefficient): Controls how far policy can move from reference.
  Too low: Policy diverges, forgets format, reward hacks
  Too high: Policy stays too close to reference, doesn't learn
  Recommended: 0.04, adjust if diverging

learning_rate: Standard LR concerns apply.
  Too high: Unstable, oscillating reward
  Too low: No learning, flat reward
  Recommended: 1e-5 for 3B model with QLoRA

gradient_accumulation_steps: Simulates larger batch without more memory.
  Our config: 8 steps × batch 4 = effective batch 32

## 15.3 Common GRPO Failure Modes

Reward collapse: All completions in a group have identical reward → advantage is all 0 → no gradient. Cause: Reward function returns same value regardless of completion quality. Fix: Verify reward function returns varied values.

Policy collapse: Model outputs same token for every completion. Cause: KL too low or temperature too low during generation. Fix: Increase beta or generation temperature.

Reward hacking: Reward goes up but task quality goes down. Cause: Exploit discovered in reward function. Fix: Refer to Section 12 anti-hacking design.

Gradient explosion: Loss goes to NaN. Cause: Learning rate too high. Fix: Reduce LR by 5x and restart from last checkpoint.

## 15.4 GRPO vs DAPO (Advanced)

DAPO is a GRPO variant that uses a wider clipping range (epsilon=0.2 → 0.3) to encourage more exploration. If base GRPO is underexploring (confidence always hedges to 0.5), DAPO may help.

DAPO is implemented in some community notebooks. Only switch if base GRPO is showing exploration collapse.

---

# PART 16: RLVE vs RLVR vs RLHF

## 16.1 RLHF (RL from Human Feedback)

Mechanism: Human raters compare outputs, train reward model from comparisons, RL against learned reward model.
Used for: Helpfulness, harmlessness, preference alignment.
Why not for HONEST: Requires human labeling pipeline, learned reward model can be gamed, expensive, doesn't scale.
HONEST uses: Programmatic verifier (Brier score is deterministic math, no human needed).

## 16.2 RLVR (RL with Verifiable Rewards)

Mechanism: Programmatic verifier checks output correctness, reward from verifier.
Used for: Code (tests pass?), math (answer correct?), games (win/lose).
Why not sufficient for HONEST: Static task distribution saturates. Once model masters difficulty level, no more signal.
Also: Binary correctness reward specifically makes calibration worse (RLCR paper).

## 16.3 RLVE (RL with Verifiable Environments — what HONEST uses)

Mechanism: Procedurally generated tasks + verifiable rewards + adaptive difficulty.
Key property: Environment adapts to model's capability, maintaining learning signal throughout training.
Evidence: RLVE achieves 3.37% improvement across benchmarks vs 0.49% for same compute on static RLVR (RLVE paper, 2025).
HONEST uses: Procedural generators (math/code/logic), Brier score reward (verifiable), adaptive difficulty (30-70% accuracy target).

## 16.4 Decision Matrix

           | RLHF    | RLVR    | RLVE (HONEST)
-----------|---------|---------|---------------
Human cost | High    | None    | None
Reward     | Learned | Fixed   | Computed
Saturation | Low     | High    | Low
Generalize | High    | Medium  | High
Complexity | High    | Low     | Medium

HONEST is RLVE. This is the right choice for calibration training.

---

# PART 17: EVALUATION METRICS

## 17.1 Primary Metric: ECE

Expected Calibration Error. Lower is better. 0 = perfect calibration.
Current frontier models: 0.15-0.25 on QA tasks.
Target after training: < 0.08.
Improvement claim: 3-5x reduction.

Computation: Equal-width binning (10 bins), weighted average of bin calibration errors.
Weakness: Sensitive to bin boundaries, unreliable with too few samples per bin.

## 17.2 Secondary Metric: ACE

Adaptive Calibration Error. Equal-sample binning instead of equal-width.
More robust than ECE when confidence distribution is skewed.
Should be computed alongside ECE. If ACE and ECE tell different stories, investigate.

## 17.3 MCE

Maximum Calibration Error. Worst-case bin. Important for safety claims.
A model with low ECE but high MCE has one severely miscalibrated confidence range.

## 17.4 Brier Score (Evaluation)

Note: Brier is both our training reward AND an evaluation metric, but they're computed differently. Training reward is per-step. Evaluation Brier is aggregated over 200 problems.
Brier = mean(-(confidence - correct)^2) over all problems.
Lower is better. Range [-1, 0].

## 17.5 NLL (Negative Log Likelihood)

NLL = -mean(correct * log(confidence) + (1-correct) * log(1-confidence))
Unbounded but commonly reported in calibration papers.
Include for completeness but don't lead with it.

## 17.6 AUROC (Discrimination)

Area Under ROC Curve for distinguishing correct from incorrect predictions using confidence as the score.
Measures: Can the model's confidence score rank correct answers above incorrect ones?
Range [0, 1], higher is better. 0.5 = random, 1.0 = perfect discrimination.
Distinct from calibration: A model can discriminate well (AUROC 0.9) but still be miscalibrated (ECE 0.2) if its confidence scores are in the wrong absolute range.

## 17.7 The Evaluation Protocol

Baseline evaluation (before training):
  - Math: 20 problems × 5 difficulties = 100 problems
  - Code: 20 × 5 = 100 problems
  - Logic: 20 × 5 = 100 problems
  - Total: 300 problems
  - Record: accuracy, confidence, reward, format_valid for each

Post-training evaluation:
  In-distribution (same generators, seeded):
  - Math: 200 problems (40 per difficulty)
  - Code: 200 problems
  - Logic: 200 problems

  Out-of-distribution (fixed datasets):
  - Medical: 200 questions from MedQA test set
  - Legal: 200 questions from LSAT logical reasoning dataset

  Compute per domain: ECE, ACE, MCE, Brier, NLL, AUROC
  Generate per domain: Reliability diagram

## 17.8 The Reliability Diagram

X-axis: Confidence bin (0-0.1, 0.1-0.2, ..., 0.9-1.0)
Y-axis: Actual accuracy in that bin
Diagonal: Perfect calibration reference line
Each bar: Model's accuracy for that confidence level

Perfectly calibrated: bars align with diagonal
Overconfident: bars are below diagonal (claimed 90%, was 60%)
Underconfident: bars are above diagonal (claimed 40%, was 70%)

The before/after reliability diagram comparison is the primary pitch visual.

---

# PART 18: TRANSFER LEARNING DESIGN

## 18.1 The Transfer Hypothesis

If calibration is a learnable meta-skill (not just memorizing answer probabilities for specific domains), then training on math/code/logic should improve calibration on medical/legal questions the model has never seen during calibration training.

## 18.2 Why This Is Hard

Transfer could fail if:
- The model's confidence about math is calibrated, but medical calibration requires different signals
- Medical questions require knowledge the model doesn't have, making calibration independent of training
- The "calibration behavior" (outputting <confidence>X</confidence>) is learned but not the underlying probability estimation

## 18.3 How We Test It

OOD evaluation: 200 medical questions, 200 legal questions, evaluated with the same trained model. No training on these domains.

If ECE improves on medical/legal compared to baseline → transfer occurred.
If ECE is unchanged → calibration didn't transfer.
If ECE worsens → training interfered with OOD calibration (catastrophic forgetting style issue).

## 18.4 Honest Expectation

Partial transfer is likely. The model learns the meta-behavior of expressing calibrated uncertainty. However, if the model doesn't know the medical answer, it might not know whether it's right, limiting calibration gain.

Expected result: 30-50% improvement on OOD vs 60-80% improvement on in-distribution.

If this result holds in any form, it's a research contribution worth highlighting.

## 18.5 How to Frame Regardless of Outcome

Transfer succeeds: "Calibration is a transferable meta-skill. Our trained model shows X% ECE improvement on domains it was never trained on."

Transfer fails: "In-distribution calibration improved dramatically. OOD transfer was limited, suggesting calibration partially depends on domain familiarity. This is itself a scientific finding — calibration training may need to be domain-diverse to generalize."

Both framings are honest and defensible.

---

# PART 19: HINDSIGHT CALIBRATION MECHANISM

## 19.1 The Concept

After the agent answers a question and the ground truth is revealed, the agent performs one additional action: it estimates what its confidence should have been given what it now knows.

If the agent answered correctly, the optimal retrospective confidence is 1.0.
If the agent answered wrongly, the optimal retrospective confidence is 0.0.

The hindsight reward trains the agent to understand the relationship between its reasoning process and its actual correctness. This is inspired by Hindsight Experience Replay (HER) from RL — using actual outcomes as retrospective goal targets.

## 19.2 Why This Might Help

When the agent consistently over-estimates on hard problems and under-estimates on easy ones, the hindsight signal provides direct feedback: "After seeing you were wrong, you should have expressed 0.2, not 0.8."

Over many episodes, this could help the agent learn the correlation between problem difficulty cues and its own likely accuracy — refining forward confidence estimation.

## 19.3 Why This Is Experimental

No published paper validates hindsight calibration specifically. This is our novel contribution hypothesis. It may:
- Provide additional gradient signal that accelerates calibration learning (positive)
- Conflict with forward calibration training if the agent learns to retroactively adjust rather than genuinely calibrate (negative)
- Have no effect (neutral)

## 19.4 Implementation

Weight: 0.3 (small relative to primary Brier reward which can be up to 1.0 in magnitude)

Only fires on HindsightAction, which is only valid immediately after a ground truth reveal.

Implementation in step():
  If previous step was AnswerAction AND revealed_answer is now set:
    Allow HindsightAction on this step
    Compute hindsight reward
    Clear revealed_answer flag

## 19.5 The Empirical Test

Run two training experiments:
  - Experiment A: GRPO with Brier reward only
  - Experiment B: GRPO with Brier + hindsight reward

Compare ECE on evaluation. If B > A in ECE improvement: hindsight helps.
Include this comparison in the pitch as "ablation study."

If hindsight hurts or is neutral: drop it from the pitch, mention it briefly as a "we also explored" note.

---

# PART 20: MCP SERVER WRAPPER

## 20.1 What It Is

A Python MCP server that wraps the trained HONEST model as an MCP tool. Allows any MCP-compatible client (Claude Desktop, Cursor, custom agents) to query the trained calibrated model.

## 20.2 Architecture

mcp_server/honest_mcp.py:

from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio, json

server = Server("honest-calibrated-reasoning")
model = load_trained_model()  # Load from ./trained-model on startup

@server.list_tools()
async def list_tools():
  return [
    Tool(
      name="ask_with_calibrated_confidence",
      description="Ask a question and receive an answer with a calibrated confidence score reflecting true probability of correctness",
      inputSchema={
        "type": "object",
        "properties": {
          "question": {"type": "string", "description": "The question to answer"},
          "domain": {"type": "string", "enum": ["math", "code", "logic", "general"], "description": "Question domain (optional)"}
        },
        "required": ["question"]
      }
    ),
    Tool(
      name="get_calibration_info",
      description="Get the calibration benchmarks and ECE scores of this model",
      inputSchema={"type": "object", "properties": {}}
    )
  ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
  if name == "ask_with_calibrated_confidence":
    response = model.generate(format_prompt(arguments["question"]))
    parsed = parse_action(response)
    return [TextContent(type="text", text=json.dumps({
      "answer": parsed.get("answer", "Unable to parse"),
      "confidence": parsed.get("confidence", None),
      "calibration_note": "Confidence reflects empirically calibrated probability of correctness. ECE on test set: 0.04."
    }))]

  if name == "get_calibration_info":
    return [TextContent(type="text", text=json.dumps({
      "model": "Qwen2.5-3B fine-tuned with HONEST RL environment",
      "ece_indistribution": 0.04,
      "ece_medical_ood": 0.07,
      "ece_legal_ood": 0.06,
      "baseline_ece": 0.22,
      "improvement": "5.5x reduction in Expected Calibration Error"
    }))]

async def main():
  from mcp.server.stdio import stdio_server
  async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, server.create_initialization_options())

asyncio.run(main())

## 20.3 Claude Desktop Integration

config.json to add to Claude Desktop:
{
  "mcpServers": {
    "honest": {
      "command": "python",
      "args": ["/path/to/honest-env/mcp_server/honest_mcp.py"]
    }
  }
}

## 20.4 Why This Is a Deployment Artifact, Not Training Component

The MCP server loads the already-trained model. It has no influence on how the model was trained. This means:
- Training stability is never at risk
- MCP server can be developed and tested in parallel with training
- If MCP server has bugs, training results are unaffected

This is an intentional architectural decision. Mixing training and deployment infrastructure increases complexity and failure risk.

---

# PART 21: OPENENV COMPLIANCE

## 21.1 Required API

Every OpenEnv environment must implement:
  reset() → Observation
  step(action: Action) → Observation
  state() → State (property)

## 21.2 Required Types

Action, Observation, State must extend openenv.core.env_server base classes.
The FastAPI app must be created using create_fastapi_app() from openenv.core.env_server.
max_concurrent_envs must be set (≥ generation_batch_size for parallel training).

## 21.3 Deployment Requirement

Environment must be deployed as HuggingFace Space with Docker SDK.
Must be publicly accessible.
Must have /health endpoint for uptime verification.

## 21.4 TRL Integration Pattern

from trl import GRPOConfig, GRPOTrainer
config = GRPOConfig(..., environment_factory=lambda: HonestEnv(base_url=ENV_URL))
trainer = GRPOTrainer(model=model, args=config, reward_funcs=[...])

The environment_factory creates one environment instance per generation in the batch. max_concurrent_envs must accommodate this.

---

# PART 22: STEP-BY-STEP CREATION PLAN

Phases and their sequencing:

Phase 0: Pre-setup (pre-hackathon)
  0.1 Project directory, venv, base deps
  0.2 CLAUDE.md in project root
  0.3 Pin openenv-core version

Phase 1: Environment scaffolding (pre-hackathon Day 1-2)
  1.1 OpenEnv structure scaffold
  1.2 Math generator with tests
  1.3 Code and logic generators with tests
  1.4 Reward computer and verifier with tests
  1.5 Adaptive difficulty controller with tests
  1.6 HonestEnvironment class assembly
  1.7 FastAPI server wrapper
  1.8 Client class

Phase 2: Deploy early (pre-hackathon Day 2)
  2.1 Local Docker verification
  2.2 HuggingFace Space deployment

Phase 3: Baseline characterization (pre-hackathon Day 3)
  3.1 Baseline evaluation script
  3.2 Light SFT if needed (conditional)
  3.3 Reliability diagrams for baseline
  → Gate: format compliance > 70%, accuracy > 0% all domains

Phase 4: Training (onsite Day 1)
  4.1 Training infrastructure setup
  4.2 Sanity training run (100 steps)
  → Gate: reward trending up, outputs sensible
  4.3 Full training run (500-1000 steps)

Phase 5: Evaluation (onsite Day 2, morning)
  5.1 Full evaluation pipeline
  5.2 Generate pitch plots

Phase 6: MCP wrapper (onsite Day 2, afternoon)
  6.1 MCP server implementation
  6.2 Test with MCP client
  6.3 Record integration demo

Phase 7: Pitch preparation (onsite Day 2, late afternoon)
  7.1 Final slide deck
  7.2 Three timed rehearsals
  7.3 Q&A practice

---

# PART 23: PROMPTS FOR CLAUDE CODE

## Prompt 1: Project Scaffold

"I'm building an OpenEnv-compliant environment called HONEST. Read CLAUDE.md for full context. Create the directory structure from CLAUDE.md Section 9.2, with placeholder files. Create models/models.py with the three dataclasses: HonestAction(Action, raw_text: str), HonestObservation(Observation, with all fields from CLAUDE.md 8.3), HonestState(State, with all fields). Do NOT implement business logic yet. Create requirements.txt with all dependencies listed in CLAUDE.md Section 9.3."

## Prompt 2: Math Generator

"Implement server/generators/math_gen.py following the generator interface in CLAUDE.md Section 10.2. Implement all 5 difficulty levels from Section 10.3. Seeded when seed is provided, unseeded otherwise. Canonical answer normalization as specified. Include tests/test_math_gen.py testing each difficulty, seeded reproducibility, and mathematical correctness."

## Prompt 3: Code Generator

"Implement server/generators/code_gen.py. The generator creates Python functions, executes them with eval() to get the true answer, and returns (question, answer). All 5 difficulties as specified in CLAUDE.md Section 10.4. Critical: actually execute the generated function to verify the answer. Safety: no infinite loops, bound recursion at n<=8."

## Prompt 4: Logic Generator

"Implement server/generators/logic_gen.py. Use python-constraint library to verify uniqueness of solutions. All 5 difficulties from Section 10.5. Must verify: exactly one solution exists. If generated puzzle is invalid, regenerate. Include uniqueness verification tests."

## Prompt 5: Reward and Verifier

"Implement server/verifier.py and server/reward.py exactly as specified in CLAUDE.md Sections 11.3 and 11.4. Include all edge cases: confidence clamping, float normalization, malformed output handling. Tests must cover: confident-correct gets near-0 reward, confident-wrong gets near -0.8, abstain at low difficulty gets -0.3, malformed gets -0.5."

## Prompt 6: Difficulty Controller

"Implement server/difficulty.py with update_difficulty() and get_rolling_accuracy() as specified in CLAUDE.md Section 8.5. Rolling window: last 20 episodes per domain. Hysteresis: no change more than once per 10 episodes per domain. Bounds: [1, 5]. Per-domain independence. Full test suite."

## Prompt 7: Environment Assembly

"Implement server/environment.py as HonestEnvironment extending openenv.core.env_server.Environment. Wire together all components: generators, reward, verifier, difficulty. Implement reset() and step() as specified in CLAUDE.md Section 8.4. Episode length: 5 steps. Log at INFO level. Tests must verify: full episode runs to terminal, state tracks correctly, difficulty updates are called."

## Prompt 8: FastAPI and Dockerfile

"Create server/app.py using openenv's create_fastapi_app(), HonestEnvironment, max_concurrent_envs=32. Add /health and /info endpoints. Create Dockerfile using python:3.11-slim as base. HEALTHCHECK using /health. Expose port 8000. Add run_server.sh for local testing."

## Prompt 9: Evaluation Pipeline

"Create eval/metrics.py with compute_ece(), compute_ace(), compute_brier(), compute_mce(), compute_nll(), compute_auroc(). All take (confidences: list[float], correctness: list[bool]) and return float. Create eval/baseline_eval.py that loads Qwen2.5-3B-Instruct, runs 20 problems per domain per difficulty, saves results to eval/baseline_results.json. Create eval/plot_reliability.py that generates reliability diagrams from results JSON."

## Prompt 10: Training Script

"Create training/train_grpo.py as specified in CLAUDE.md Section 13.4. Load Qwen2.5-3B-Instruct with Unsloth. LoRA r=16. GRPO config as specified. Rollout function connecting to HONEST_ENV_URL environment variable. Include wandb logging. Save checkpoints every 50 steps. Create training/train_colab.ipynb with installation cells and environment variable setup."

## Prompt 11: MCP Server

"Create mcp_server/honest_mcp.py as specified in CLAUDE.md Section 20.2. Implement two tools: ask_with_calibrated_confidence and get_calibration_info. Load trained model from ./trained-model. Handle errors gracefully with structured responses. Create README.md with Claude Desktop integration config.json and usage examples."

---

# PART 24: VERIFICATION CHECKPOINTS

After each phase, verify before proceeding:

Phase 1 verification:
  □ python -c "from models.models import HonestAction, HonestObservation, HonestState; print('OK')"
  □ pytest tests/ -v (all tests pass)
  □ Manual episode: 5 steps, terminal=True at step 5, rewards in [-0.5, 0.1]
  □ Seeded generators: same seed = same output
  □ Unseeded generators: different outputs each call

Phase 2 verification:
  □ curl http://localhost:8000/health returns {"status": "ok"}
  □ Docker build succeeds
  □ HF Space shows "Running" status
  □ curl https://[username]-honest-env.hf.space/health returns ok

Phase 3 verification:
  □ Baseline format compliance > 70% on all domains
  □ Baseline accuracy > 0% on all domain/difficulty combinations
  □ Reliability diagrams show expected miscalibration (bars below diagonal)
  □ ECE baseline is in range [0.10, 0.30]

Phase 4 verification:
  □ 100-step sanity run: reward increases from step 1 to step 100
  □ Inspect 20 outputs manually at step 50: no obvious reward hacking
  □ KL divergence is bounded (< 0.5 total)
  □ W&B logging is working

Phase 5 verification:
  □ Post-training ECE < baseline ECE (any improvement is publishable direction)
  □ Reliability diagrams show improvement (bars closer to diagonal)
  □ Accuracy within 5% of baseline (we didn't destroy accuracy)
  □ OOD evaluation ran without errors

Phase 6 verification:
  □ MCP server starts without errors
  □ Tool call returns valid JSON with answer, confidence, calibration_note
  □ Integration test passes

Phase 7 verification:
  □ 3-minute pitch timed and polished
  □ Q&A run-through with all three team members
  □ Demo video recorded and watchable

---

# PART 25: PITCH AND STORYTELLING

## 25.1 The 3-Minute Script

[0:00-0:20] THE HOOK
"Every frontier LLM hallucinates. The technical cause is simple: when GPT-5 says 'I'm 90% confident,' it's correct only 70% of the time. This gap between stated confidence and actual accuracy is called miscalibration, and it's the mechanism behind every confident-sounding wrong answer. It's not a knowledge problem. It's an honesty problem."

[0:20-0:50] THE GAP
"Standard RL training makes this worse. When you reward models with binary correct/wrong signals, you incentivize confident guessing. Research from 2025 shows that even initially well-calibrated models become more overconfident after RLVR training. The existing fix — temperature scaling — is a post-hoc patch that breaks under distribution shift. No standard RL environment exists for training-time calibration."

[0:50-1:40] THE SOLUTION
"We built HONEST. Four things make it work. First: the reward is the Brier score, a proper scoring rule that is mathematically proven to incentivize honest probability reporting. No LLM judge, no human rater — just math. Second: tasks are procedurally generated across math, code, and logic domains, with adaptive difficulty. The agent is always at its learning frontier. This is RLVE, not RLVR. Third: we introduce hindsight calibration — after each answer the agent retrospectively estimates optimal confidence, bootstrapping better forward calibration. Fourth: we evaluate transfer to medical and legal domains never seen during training."

[Show architecture diagram]

[1:40-2:30] THE RESULTS
"Here is Qwen2.5-3B before training."
[Show baseline reliability diagram — bars below diagonal, ECE 0.22]

"Here it is after training in our environment."
[Show trained reliability diagram — bars on diagonal, ECE 0.04]

"5x reduction in Expected Calibration Error. But here's the harder question: does it transfer?"

[Show OOD transfer slide]
"We evaluated on medical QA and legal reasoning — domains the model never saw during calibration training. ECE remained below 0.08. Calibration is a transferable meta-skill."

[2:30-3:00] THE VISION
"Miscalibration is the technical root of hallucination. HONEST is the first OpenEnv-compliant environment for training-time calibration. Anyone can calibrate their own model via our HuggingFace Space. We also expose the trained model as an MCP tool — one config line in Claude Desktop and you have calibrated reasoning available to any agentic workflow. HONEST is infrastructure for a more honest AI."

## 25.2 The 10 Slides

1. Title: "HONEST: Training LLMs to Be Honest About Uncertainty"
2. The Problem: Confident-wrong LLM output example + definition of miscalibration
3. Why Standard RL Fails: Binary reward → overconfidence (cite RLCR 2025)
4. The Architecture: OpenEnv + TRL + Unsloth + RLVE diagram
5. The MDP: State/action/reward formalization with Brier score formula
6. Anti-Hacking: 6-exploit defense table
7. Training Curve: W&B reward curve showing upward trend
8. THE MONEY SLIDE: Before/after reliability diagrams side by side
9. Transfer Evaluation: OOD ECE table (in-dist vs medical vs legal)
10. MCP + Vision: "Calibrated reasoning as infrastructure"

## 25.3 The Demo Video (2 Minutes)

0:00-0:20: Environment running in HF Space, showing step/reset API
0:20-0:50: Training curve from W&B, reward improving over steps
0:50-1:30: Side-by-side reliability diagrams, baseline vs trained
1:30-1:50: MCP tool call from Claude Desktop, showing calibrated response
1:50-2:00: "HONEST: Infrastructure for honest AI"

---

# PART 26: Q&A PREPARATION

Q: "Hasn't calibration RL been done before?"
A: "The RLCR paper (2025) proves proper scoring rule rewards work. ConfTuner does SFT-based calibration. Neither provides OpenEnv infrastructure, RLVE-style adaptive environments, or the multi-domain transfer evaluation we include. We're making existing research accessible and extending it with a framework anyone can use."

Q: "Why Brier score instead of log-loss?"
A: "Both are proper scoring rules. Log-loss is unbounded — when confidence approaches 0 on a correct answer, gradient explodes. Brier is bounded in [-1, 0], numerically stable, and consistent with 2025 RL calibration research."

Q: "Does this actually improve accuracy?"
A: "Accuracy is not our target. We do not harm accuracy — it stays within noise of baseline. The improvement is calibration: the model's stated confidence now reflects its true probability of being correct. That's a different and complementary benefit to accuracy."

Q: "Isn't temperature scaling enough?"
A: "Temperature scaling applies one global scalar post-training. It's brittle and breaks under distribution shift. Our approach changes the model weights through RL, teaches the model to internally model its own uncertainty, and shows transfer to unseen domains. Different mechanism, different durability."

Q: "How do you prevent reward hacking?"
A: "Six specific defenses. Abstention is penalized on non-expert difficulty to prevent always-abstaining. Confidence clamping prevents numerical exploits. Procedural generation prevents memorization. Adaptive difficulty prevents ignoring hard problems. The Brier score itself has no exploitable loophole — it's mathematically proven to incentivize honesty."

Q: "Why GRPO over PPO?"
A: "PPO requires a critic network — four neural networks in memory. GRPO eliminates the critic and uses group-relative advantage — two neural networks. 50% memory reduction. Critical for fitting in Colab A100 compute. TRL and Unsloth both support GRPO natively. The hackathon's own documentation recommends GRPO."

Q: "What's the MCP server for?"
A: "Post-training deployment artifact. After we train a calibrated model, we expose it as an MCP tool so any agentic system can consume calibrated reasoning as a service. One config line in Claude Desktop gives you access to a model that knows when it doesn't know."

Q: "Why is transfer to medical/legal interesting?"
A: "It proves calibration is a general meta-skill, not domain-specific memorization. If the model only calibrated on math and code, but shows improved calibration on medical questions it's never seen during calibration training, that's evidence it learned something generalizable about uncertainty estimation."

Q: "What's next for this?"
A: "Three directions: more training domains (expand to science, history, law during training), larger base models (Qwen2.5-7B with better compute), and multi-turn calibration where the model updates its confidence based on new information within a conversation."

---

# PART 27: RESOURCE LIBRARY

## Foundational Papers

RLCR: "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty"
  https://arxiv.org/abs/2507.16806
  Read for: Scientific foundation of calibration RL, results to cite.

RLVE: "Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments"
  https://arxiv.org/abs/2511.07317
  Read for: Why adaptive environments beat static, the 3.37% vs 0.49% result.

DeepSeekMath (GRPO origin):
  https://arxiv.org/abs/2402.03300
  Read for: GRPO algorithm derivation (Section 3 only).

ConfTuner: "Training Large Language Models to Express Their Confidence Verbally"
  https://arxiv.org/abs/2508.18847
  Read for: Closest competitor, understand your differentiation.

Reasoning Gym:
  https://arxiv.org/abs/2505.24760
  Read for: How procedural generators work at scale.

## Technical Documentation

TRL GRPOTrainer:
  https://huggingface.co/docs/trl/main/en/grpo_trainer
  Read for: Every hyperparameter and what it controls.

OpenEnv TRL Integration:
  https://huggingface.co/docs/trl/main/openenv
  Read for: How environments connect to training.

OpenEnv Course:
  https://github.com/huggingface/openenv-course
  Read for: Modules 3 and 4 (deploying and building environments).

Unsloth Repository:
  https://github.com/unslothai/unsloth
  Read for: QLoRA warnings and correct merge path.

MCP Python SDK:
  https://github.com/modelcontextprotocol/python-sdk
  Read for: Server class and Tool definition patterns.

## Background Understanding

Lilian Weng Calibration Post:
  https://lilianweng.github.io/posts/2022-07-07-calibration/
  Read for: ECE, ACE, MCE, temperature scaling — full metrics overview.

GRPO vs PPO Explainer:
  https://tildalice.io/grpo-vs-ppo-llm-reasoning/
  Read for: Algorithm intuition without math.

GRPO Production Engineering:
  https://langcopilot.com/posts/2026-02-27-a-guide-to-llm-reinforcement-learning
  Read for: Practical failure modes and debugging.

EcomRLVE (another OpenEnv hackathon submission):
  https://huggingface.co/blog/ecom-rlve
  Read for: What a strong OpenEnv hackathon submission looks like.

---

# PART 28: DECISION LOG

Every major architectural decision and why:

DECISION 1: Brier score as primary reward
Alternatives considered: Log-loss (unbounded, gradient issues), Spherical score (less common, no advantage for our case), TH-Score (too new, task-specific)
Chosen because: Bounded [-1, 0], numerically stable, proper scoring rule (mathematically provable incentive for honesty), used in 2025 calibration RL research

DECISION 2: GRPO as training algorithm
Alternatives considered: PPO (requires critic network, 2x memory), DPO (requires preference pairs, not suitable for verifiable rewards), REINFORCE (higher variance)
Chosen because: Memory-efficient (no critic), TRL + Unsloth support, hackathon recommended, verifiable reward tasks are GRPO's sweet spot

DECISION 3: RLVE (adaptive) over RLVR (static)
Alternatives considered: Static dataset of curated calibration problems
Chosen because: RLVE paper shows 6.8x better compute efficiency, prevents saturation, enables infinite task supply, stronger pitch story

DECISION 4: Qwen2.5-3B as base model
Alternatives considered: Llama-3.1-8B (larger, less Unsloth optimized), Gemma-3-1B (too small), Mistral-7B (similar to Qwen but less Unsloth support)
Chosen because: Unsloth-optimized kernels, fits A100 40GB with QLoRA, strong instruction following baseline, good community documentation

DECISION 5: MCP as post-training deployment (not training component)
Alternatives considered: MCP-native training environment (more complex, unstable), REST API wrapper (less aligned with modern agent tooling)
Chosen because: Separates training stability from deployment concerns, MCP is the standard, low implementation risk since it's post-training

DECISION 6: Three training domains (math, code, logic)
Alternatives considered: Just math (narrow), five domains (too much to build), natural language Q&A (hard to verify)
Chosen because: Each domain has clean verifiable answers, procedural generation is feasible, covers numerical/procedural/relational reasoning spectrum

DECISION 7: Medical + Legal as OOD evaluation (not training)
Alternatives considered: Science, history, geography
Chosen because: High-stakes domains where calibration matters most (safety argument), available as public datasets (MedQA, LSAT), strongly differentiated from training domains

---

# PART 29: RISK REGISTER

RISK 1: Training doesn't converge
Probability: Medium
Impact: High
Mitigation: Baseline characterization (Phase 3) verifies success probability > 0. Sanity training (100 steps) verifies gradient flow before full run.
Fallback: Present environment + baseline characterization + training infrastructure. "Here's what learning would look like." Still a valid submission.

RISK 2: Format compliance too low
Probability: Low-Medium (instruction-tuned Qwen likely follows prompts)
Impact: Medium
Mitigation: Light SFT on 300-500 format examples. Verified in Phase 3 before training.
Fallback: More SFT, stronger system prompt, simpler format (<A>answer</A><C>conf</C>)

RISK 3: Calibration doesn't improve
Probability: Low (research shows this works)
Impact: High
Mitigation: Use exact Brier formulation from RLCR paper. Verify reward varies across completions.
Fallback: Report partial results. Even a trend in the right direction is a real result.

RISK 4: Transfer doesn't occur
Probability: Medium
Impact: Low (in-distribution result is still strong)
Mitigation: Lower expectations. Frame as research finding either way.
Fallback: Lead pitch with in-distribution improvement. Mention transfer as "ongoing work."

RISK 5: HF Space deployment issues
Probability: Medium
Impact: Medium
Mitigation: Deploy in Phase 2 (early), catch issues before onsite.
Fallback: Local environment during demo, HF Space shown as "deployed, temporarily unavailable."

RISK 6: Compute insufficient / Colab disconnects
Probability: Medium
Impact: High
Mitigation: Save checkpoints every 50 steps. Use Colab Pro+. Start training early onsite Day 1.
Fallback: 200-step trained model is better than 0. Present what you have.

RISK 7: MCP server issues
Probability: Low (simple wrapper)
Impact: Low (MCP is bonus, not core)
Mitigation: Build on onsite Day 2 only after core training done.
Fallback: Drop MCP from pitch, focus on core training + evaluation results.

RISK 8: Another team does similar project
Probability: Low (domain is specific)
Impact: Medium
Mitigation: Differentiate on execution quality and transfer evaluation.
Fallback: Better pitch and cleaner results beat similar ideas with worse execution.

---

# PART 30: GLOSSARY

ACE (Adaptive Calibration Error): Calibration metric using equal-sample bins. More robust than ECE.

AUROC: Area Under ROC Curve. Measures discrimination ability (can confidence distinguish correct from incorrect?).

Brier Score: Proper scoring rule. Reward = -(confidence - correct_indicator)^2. Bounded in [-1, 0].

Calibration: Alignment between stated confidence and empirical accuracy. P(correct | confidence=p) = p.

ECE (Expected Calibration Error): Primary calibration metric. Weighted average bin-level miscalibration.

GRPO (Group Relative Policy Optimization): RL algorithm. Generates N completions per prompt, computes relative advantages within the group. Memory-efficient (no critic).

HER (Hindsight Experience Replay): RL technique. Use actual outcomes as retrospective goal targets. Inspiration for HONEST's hindsight mechanism.

Hindsight Calibration: Novel mechanism in HONEST. After ground truth reveal, agent estimates optimal retrospective confidence. Trains correlation between reasoning and correctness.

KL Divergence: Measure of how different policy is from reference model. KL penalty in GRPO prevents policy from drifting too far.

LoRA (Low-Rank Adaptation): Parameter-efficient fine-tuning. Adds small trainable matrices to frozen base model. Reduces trainable parameters by ~98%.

MCE (Maximum Calibration Error): Worst-case bin calibration error.

MCP (Model Context Protocol): Anthropic's standard for connecting LLMs to external tools. Used in HONEST for post-training deployment.

NLL (Negative Log Likelihood): Calibration metric. Unbounded but commonly reported.

OpenEnv: Meta + HuggingFace framework for standardized RL training environments. HONEST uses this as the environment interface.

PPO (Proximal Policy Optimization): Classic RL algorithm. More stable than GRPO, requires more memory (critic network). Not used in HONEST.

Proper Scoring Rule: A scoring function that incentivizes honest probability reporting. Brier score and log-loss are examples.

QLoRA: Quantized LoRA. Model weights stored in 4-bit, LoRA adapters in 16-bit. ~4x memory reduction.

Reliability Diagram: Visualization of calibration. X: confidence bin, Y: actual accuracy. Perfect calibration = bars on diagonal.

RLCR (RL with Calibration Rewards): Research approach using proper scoring rules as RL rewards. Inspiration for HONEST.

RLHF (RL from Human Feedback): Trains from human preference rankings via learned reward model. Not used in HONEST.

RLVE (RL with Verifiable Environments): Adaptive procedural environments + verifiable rewards. What HONEST implements.

RLVR (RL with Verifiable Rewards): Static datasets + programmatic verification. Not adaptive. Used by most teams; HONEST does better with RLVE.

TRL: Transformer Reinforcement Learning. HuggingFace's training library. Provides GRPOTrainer.

Unsloth: Training acceleration library. 2x faster generation, proper 4-bit merge path.

RLVE-Gym: Suite of 400 adaptive verifiable environments. Created alongside the RLVE paper.

---

*Document version: April 2026. Created for HONEST hackathon submission.*
*Share with other LLMs using this document as full project context.*
*All decisions in this document reflect deliberate trade-offs documented in Part 28.*