---
title: HONEST Env
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Calibration-aware OpenEnv for LLM agents (Brier-shaped GRPO)
---

# HONEST-RL-Calibrator

**Honesty-Optimized and Normalized Environment for Self-Triage** — an
[OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement
learning environment that trains language models to **report calibrated
confidence** alongside every answer.

> ## Submission deliverables
>
> | Artifact | Link |
> | -------- | ---- |
> | 🤗 **Hugging Face Space (live env)** | **<https://huggingface.co/spaces/Rushhaabhhh/HONES-RL-Calibrator>** |
> | 🏗️ Source repository | <https://github.com/Rushhaabhhh/HONEST-RL-Calibrator> |
> | 📓 Training notebook (Colab-ready) | [`training/train_colab.ipynb`](training/train_colab.ipynb) — [open in Colab](https://colab.research.google.com/github/Rushhaabhhh/HONEST-RL-Calibrator/blob/main/training/train_colab.ipynb) |
> | 🐍 Training script (Python) | [`training/train_grpo.py`](training/train_grpo.py) |
> | 📝 Project writeup | [`docs/WRITEUP.md`](docs/WRITEUP.md) |
> | 📈 Training curves (PNG) | [`docs/training/`](docs/training/) — rendered by `bin/plot_training_curves.py` from each run's `trainer_state.json`; regenerate with `make plots-demo` or from your own state file |
> | 🔌 MCP deployment wrapper | [`mcp_server/`](mcp_server/) — [`mcp_server/README.md`](mcp_server/README.md) |
> | 🛠️ End-to-end runbook | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |
> | 🧪 Self-learning research memo | [`docs/SELF_LEARNING.md`](docs/SELF_LEARNING.md) |
>
> **Quickstart for judges (60 seconds, no GPU):**
> ```bash
> git clone https://github.com/Rushhaabhhh/HONEST-RL-Calibrator.git
> cd HONEST-RL-Calibrator && make validate   # passes openenv validate
> ./bin/run_server.sh                         # boots the env locally on :8000
> ```
> All paths above resolve from a clean `git clone` — no external
> dependencies are required to inspect the deliverables.

The agent does not just answer; it must declare *how confident it is* in
that answer, or explicitly abstain. Reward is a strictly proper scoring
rule (Brier score), so the only way to maximize return is to make
confidences **match empirical correctness**.

```
Question  ──►  <reasoning>...</reasoning>
                <answer>42</answer><confidence>0.83</confidence>
                              │
                              ▼
                Reward = -1.5·(c - y)² + format/abstain shaping
```

After training, the model is exposed via a Model Context Protocol (MCP)
server so any MCP-compatible client (Claude Desktop, Cursor, LangGraph
agents) can consume calibrated reasoning as a service.

---

## Why this exists

LLMs are notoriously **overconfident**. They emit a fluent answer with a
fluent justification regardless of whether they know it. Two failure
modes follow:

1. **Silent errors** — high-confidence wrong answers that downstream
   systems trust.
2. **Worthless confidence** — a number between 0 and 1 that has no
   relationship to actual `P(correct)`.

This project fixes both with a single training loop:

* **Strictly proper scoring rule** — Brier score gradients reward
  *honest* probabilities, not just correct answers.
* **Adaptive curriculum** — difficulty rises with rolling accuracy, so
  the model never sits at a saturated reward signal.
* **Self-learning extensions** (opt-in) — hindsight reasoning,
  prioritized replay, self-mutating curriculum, generator/solver
  self-play (see [`docs/SELF_LEARNING.md`](docs/SELF_LEARNING.md)).

After training, expected calibration error (ECE), Brier score, and
AUROC are reported in-distribution and on a five-slice OOD suite the
model never saw during training:

| OOD slice      | source                                  | random floor |
|----------------|-----------------------------------------|:------------:|
| `commonsense`  | `tau/commonsense_qa`                    |     0.20     |
| `science_easy` | `allenai/ai2_arc` ARC-Easy              |     0.25     |
| `science_hard` | `cais/mmlu` astronomy                   |     0.25     |
| `medical`      | `cais/mmlu` professional_medicine       |     0.25     |
| `legal`        | AGIEval LSAT-LR (MMLU law fallback)     |     0.20     |

The set is **tier-aware**: tiny models (Qwen-0.5B, Llama-1B) only have
measurable accuracy headroom on the first three slices, so
`full_eval.py --ood-slices auto` skips `medical` and `legal` for those
sizes. The transferability claim ("calibration trained on math/code/logic
generalises to OOD") is rendered by `eval/compare_runs.py` as a
**Calibration Transfer** table with per-slice ΔECE 95 % paired-bootstrap
CIs and a single transfer-ratio number — see
[`docs/RUNBOOK.md` §5](docs/RUNBOOK.md#5--comparison--success-metrics).

---

## Training curves

Training curves (reward, loss, KL) are generated from each run's
`trainer_state.json` by `bin/plot_training_curves.py`. To regenerate
after your own run completes:

```bash
python bin/plot_training_curves.py \
    --trainer-state ./honest-<preset>-grpo/checkpoint-final/trainer_state.json \
    --out docs/training \
    --label "<preset> · <steps> steps"
```

What the three curves track:

| Curve | What to watch |
|-------|--------------|
| **Reward** | Primary Brier term `−1.5·(c−y)²` plus `+0.15` format bonus. Starts negative (overconfident base model); climbs as emitted `confidence` aligns with empirical correctness. Expect steep recovery in first ~100 steps, slow consolidation thereafter. |
| **Policy loss** | GRPO surrogate loss under cosine LR. Tracks advantage variance, not a "lower is better" target — monitor for dead-batch spikes. |
| **KL** | `KL(π ‖ π_ref)`. `AdaptiveBetaCallback` clamps this below the 0.5 early-stop threshold; the callback auto-kills runs that breach it for 20 consecutive steps. |

---

## Empirical findings and projections

Calibration RL on a Brier-shaped GRPO objective is a non-standard
regime: most published runs optimise *accuracy* under PPO/GRPO and
report calibration as an after-the-fact metric. We optimise it
directly, which exposes failure modes the standard recipe hides. The
findings below are split into **measured** on the small-model pilot
regimen (≤ 1 B parameters) and **projected** for v2 hindsight (CASR)
and 3 B / 7 B-class extrapolation. Specific numbers we treat as
projections are explicitly labelled as such.

### What we ran

| Regimen                       | Models                                                                  | Steps | Status                | Evidence                                                                |
| ----------------------------- | ----------------------------------------------------------------------- | ----- | --------------------- | ----------------------------------------------------------------------- |
| SFT-then-GRPO (tiny tier)     | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct                            | 250   | ✅ pilot complete     | `eval/full_results_<preset>.json` (drop in alongside this README)        |
| GRPO direct (medium tier)     | Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Phi-4-mini-Instruct         | 350   | 🟡 in flight          | `outputs/<preset>/trainer_state.json` once each run lands               |
| GRPO + CASR (v2 hindsight)    | Across the preset matrix                                                | —     | 🔵 projected          | smoke-tested via `make smoke-train --hindsight-mode refined`            |

### Findings on the small-model pilots

1. **Hindsight v1 is a silent channel under single-pass GRPO.** The
   legacy `<hindsight>` head rewards a retrospective confidence `r`
   against ground truth `y` with `−k(r−y)²`. The optimal policy under
   this reward composed with the primary Brier is identical to the
   optimal policy under Brier alone — both push `c = r = E[y|x]` —
   *and* the base models have no prior on the `<hindsight>` tag, so
   the channel returns 0.0 for the entire run. We verified this
   directly with `bin/audit_hindsight.py` on the v1 trajectories
   before designing v2. The implication is that **hindsight as
   originally formulated in HER does not transfer to single-pass
   calibration RL** — the information content is identically zero
   unless the post-hoc revision is conditioned on a strictly larger
   info set than the original confidence (which is precisely what
   CASR does in §2.5 of the self-learning memo).

2. **Tiny tier collapses without an SFT warmup.** Qwen-0.5B and
   Llama-1B cannot reliably emit the 3-tag XML contract from a system
   prompt alone. Direct GRPO produces `frac_reward_zero_std ≈ 1.0` in
   the first 100 steps — the advantage normaliser divides by zero, no
   calibration gradient flows, and the run silently wastes its
   compute on the malformed-penalty floor. A short Calibration-SFT
   pass (≈ 1500 examples × 2 epochs, ~8 min on a single A100)
   bootstraps three priors at once: format compliance, a
   correctness-conditioned confidence prior, and the hindsight tag.
   After SFT the tiny tier shows the *same* reward-trajectory shape
   as the medium tier with a lower absolute floor — calibration
   mechanism transfers across scale, base reasoning capacity does not.

3. **Anti-hedge regularisation is fragile.** A symmetric anti-hedge
   penalty on `c ∈ [0.4, 0.6]` was removed in commit `3690671` after
   we found the model could exploit a 0.7-confidence band edge:
   *technically* outside the penalty zone, *semantically* hedging,
   and losing less Brier than under honest calibration. The proper
   fix is to let the strictly proper scoring rule do the work and
   rely on KL to keep the policy from collapsing to a delta —
   auxiliaries that *look* like they punish hedging often actively
   reward the wrong solution.

4. **OOD transfer is tier-bounded by the random-MCQ floor.** The
   transferability claim ("calibration trained on math/code/logic
   generalises to held-out OOD") is provable only on slices where the
   model scores meaningfully above its random-MCQ floor. For tiny
   models that means CommonsenseQA, ARC-Easy, MMLU-astronomy
   (25–65 % accuracy band); MMLU-professional_medicine and
   AGIEval LSAT-LR pin at the 25 % / 20 % floor and produce no
   measurable ΔECE. We surfaced this as tier-aware OOD slice
   selection (`recommended_ood_slices` per preset in
   `calibration_profiles.py`) so the comparison report doesn't claim
   transfer where the underlying metric has no signal — see
   `eval/compare_runs.py` for the per-slice paired-bootstrap CI
   rendering.

### Projections — v2 hindsight (CASR) and the 3 B / 7 B regime

The forecasts below are grounded in (a) the published mechanism papers
referenced in [`docs/SELF_LEARNING.md`](docs/SELF_LEARNING.md) §2.5,
(b) the smoke-test runs of CASR (`make smoke-train` invoked with
`--hindsight-mode refined` against a 32-step budget), and (c) the
structural invariance observed in the small-model pilots. They are
explicitly **not** measured numbers from a completed sweep.

| Question                                          | Projection                                                                                                                              | Basis                                                                                                                                                                                                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Will CASR fire on a cold-start medium model?      | Non-zero hindsight rate within ~50 GRPO steps.                                                                                          | The `+β` format-bonus term on the new `<critique>` / `<refined_confidence>` tags produces a positive gradient even before the model can write a useful critique — same mechanism that drives medium-tier 3-tag format compliance to ~90 % by step ~30. |
| Does CASR add a *new* gradient channel beyond Brier? | Yes, strictly.                                                                                                                          | `ΔBrier = (c−y)² − (r−y)²` is non-redundant with the primary reward because `r` is conditioned on `(x, c, critique)` rather than `x` alone (Self-Refine 2023; Process Reward Models 2021). v1's optimum was redundant; v2's is not.                |
| Effect on `frac_reward_zero_std`?                  | ~50 % reduction on dead batches where the group agrees on `(c, y)` but diverges on critique emission.                                   | Format bonus produces non-zero group-relative advantage when the primary reward variance is zero — recovering signal that v1 lost.                                                                                                                |
| ΔECE on 3 B-class in-distribution?                 | 30–50 % relative reduction projected.                                                                                                   | Brier-scoring-rule literature (Gneiting & Raftery 2007) sets the theoretical ceiling; post-hoc temperature scaling (Guo et al. 2017) and RL-from-feedback calibration (Tian et al. 2023) bracket the empirically achievable range.                |
| ΔECE on the OOD suite?                             | Calibration transfer ratio ≈ 0.4–0.7 of the in-distribution gain on slices clear of floor.                                              | Bracketed by the simulated bootstrap CIs from `eval/compare_runs.py` against synthetic before/after pairs; matches the transfer-ratio range typical of Brier-shaped RL recipes in the literature.                                                  |
| ΔECE at 7 B+?                                      | Larger absolute, smaller relative — base ECE on Qwen-2.5-7B / Llama-3-8B is roughly half the 3 B floor.                                  | Calibration scales with capacity; the Brier gradient delivers diminishing returns once the base policy is close to the proper-scoring-rule equilibrium.                                                                                            |
| Compute envelope?                                  | Tiny: ~70 min/model on A100 (8 min SFT + 60 min GRPO + 0 incremental for CASR). Medium 3 B: ~3–4 h on A100 / L4. 7 B: ~10–12 h on A100. | Step-time × step-count from the pilot runs; CASR adds ≤ 5 % wall-clock overhead because it shares the rollout with the primary reward (single forward pass).                                                                                       |

### What is intentionally **not** claimed in this README

* No specific post-RL ECE / Brier number is reported here as a
  headline. The full metric battery is rendered by
  `eval/compare_runs.py` against your own
  `eval/full_results_<preset>.json` outputs, with paired-bootstrap
  95 % CIs on the deltas, so every number in the submission is
  reproducible from a clean clone — not a stat we asked you to trust.
* The v2 hindsight ablation across the full preset matrix exceeded
  the hackathon compute budget. The CASR mechanism is shipped,
  audited (`bin/audit_hindsight.py`), and smoke-tested; the full
  empirical sweep is documented as the natural next experiment in
  [`docs/SELF_LEARNING.md`](docs/SELF_LEARNING.md) §2.5.

---

## Architecture

```
┌────────────────────────────── HONEST-Env ──────────────────────────────┐
│                                                                        │
│  data/                  server/                  training/             │
│  ├── ingestion/         ├── environment.py       ├── train_grpo.py     │
│  │   (Hendrycks MATH,   │   (OpenEnv MDP)        │   (TRL GRPO)        │
│  │    MBPP, APPS,       ├── reward.py            └── format_sft.py     │
│  │    ZebraLogic)       │   (Brier + shaping)        (optional Stage 2)│
│  ├── verifiers/         ├── verifier.py                                │
│  │   (math/code/logic)  │   (XML parser, GT match)                     │
│  ├── sampler/           ├── difficulty.py                              │
│  │   (UnifiedSampler)   │   (adaptive controller)                      │
│  └── processed/*.jsonl  ├── hindsight.py    ┐                          │
│                         ├── replay_buffer.py │  Self-learning          │
│                         ├── mutators.py      │  (docs/SELF_LEARNING.md)│
│                         └── self_play.py    ┘                          │
│                                                                        │
│  eval/                                          mcp_server/            │
│  ├── baseline_eval.py   (pre-RL anchor)         ├── honest_mcp.py      │
│  ├── full_eval.py       (post-RL ID + OOD)      ├── __main__.py        │
│  ├── compare_runs.py    (Δ + bootstrap CI)      └── README.md          │
│  ├── plot_reliability.py                                               │
│  └── ood/fetch_ood_data.py                                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

| Layer            | Responsibility                                                                      |
| ---------------- | ----------------------------------------------------------------------------------- |
| **`data/`**      | Ingest external datasets, verify ground truth, expose a unified sampler.            |
| **`server/`**    | OpenEnv environment, reward, verifier, adaptive curriculum, self-learning pillars.  |
| **`training/`**  | GRPO training loop with W&B logging, KL adaptive beta, dead-batch guard.            |
| **`eval/`**      | Baseline + full evaluation, OOD generalization, comparison report, plots.           |
| **`mcp_server/`**| Stateless MCP wrapper around the trained adapter for external clients.              |

---

## Domains

The environment generates problems across three domains, five difficulty
levels each. All problems carry a verifiable ground truth.

| Domain   | Source                                | Verifier                             |
| -------- | ------------------------------------- | ------------------------------------ |
| **Math** | Hendrycks MATH (~12.5k problems)      | SymPy equivalence                    |
| **Code** | MBPP + APPS (~427+ MBPP, APPS streamed)| Sandboxed execution against tests   |
| **Logic**| Regenerated ZebraLogic (CSP puzzles)  | python-constraint / Z3 unique-sol    |

Procedural generators (`server/generators/`) provide an additional
fallback when curated data is missing, but the **unified sampler** in
`data/sampler/` is the production source.

---

## Action interface

The agent emits XML and is parsed strictly:

```xml
<reasoning>chain of thought, free-form</reasoning>
<answer>42</answer>
<confidence>0.83</confidence>
```

or, when uncertain:

```xml
<reasoning>...</reasoning>
<abstain/>
```

* `confidence` ∈ [0, 1] — strictly proper scoring punishes
  miscalibration in either direction.
* `<abstain/>` — small penalty on easy problems, near-zero on the
  hardest ones; the model learns *when* to refuse.
* Anything else → fixed malformed penalty.

---

## Reward scheme

The full reward formula (`server/reward.py`):

```
R = -1.5 · (confidence - correct)²       # Brier (primary)
    + 0.15 · 1[strict_format]            # format bonus
    +  0.0 · 1[abstain]                  # abstain neutral
    - 1.00 · 1[malformed]                # malformed penalty
    - 0.25 · 1[hint_in_reasoning]        # anti-leak penalty
```

| Outcome                       | Approximate reward       |
| ----------------------------- | ------------------------ |
| Correct + high confidence     | `~ +0.15`                |
| Wrong + high confidence (1.0) | `~ -1.35`                |
| Wrong + low confidence (0.05) | `~ -0.13`                |
| Abstain on hard problem       | `~ +0.15` (format only)  |
| Malformed                     | `-1.00` floor            |

Why these constants: Brier scale of 1.5 is large enough that calibration
gradients dominate format gradients but small enough that one bad token
doesn't blow up advantage normalization in GRPO. See
`server/reward.py` for the full derivation.

### Adaptive difficulty (`server/difficulty.py`)

Per-domain rolling window of 20 episodes:

* Rolling accuracy > **0.70** → bump difficulty (capped at 5, or higher
  if `--self-mutate` is enabled).
* Rolling accuracy < **0.30** → drop difficulty (floor at 1).
* **Hysteresis**: 10-episode cooldown between changes prevents oscillation.

---

## Self-learning calibration

Four opt-in mechanisms turn the fixed-task GRPO loop into a recursive
skill amplification system. See [`docs/SELF_LEARNING.md`](docs/SELF_LEARNING.md)
for the full research memo.

| Pillar                                | Flag                  | What it adds                                                        |
| ------------------------------------- | --------------------- | ------------------------------------------------------------------- |
| Hindsight Calibration Reward (HCR)    | `--hindsight`         | Retrospective confidence head. Two modes: `legacy` (default) grades a `<hindsight>` tag with `-k(r-y)²`; `refined` (`--hindsight-mode refined`) uses Calibration-Aware Self-Refinement — model critiques its own answer and refines confidence. See [`docs/SELF_LEARNING.md` §2.5](docs/SELF_LEARNING.md#25-v2--calibration-aware-self-refinement-casr). |
| Calibration-Prioritized Replay (CPR)  | `--replay-priority`   | Re-sample miscalibrated prompts (PER on `\|c−y\|`).                 |
| Self-Mutating Curriculum (SMC)        | `--self-mutate`       | Extend ceiling above d=5 via deterministic problem mutators.        |
| Generator/Solver Self-Play (GSS)      | `--self-play`         | PAIRED-style generator rewarded for solver miscalibration.          |

Quick verification without GPU:

```bash
make smoke-train   # train_grpo --dry-run --hindsight --replay-priority --self-mutate --self-play
make test          # full pytest suite
```

---

## Quick start

```bash
# Environment
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Smoke test the entire stack (no GPU needed)
./venv/bin/python -m pytest tests/ data/tests/
make smoke-train
make mcp-smoke

# Validate the OpenEnv contract (passes all four deployment modes)
make validate

# Run the OpenEnv server locally
./bin/run_server.sh
# Or via Docker (HuggingFace Spaces ready)
docker build -t honest-rl-calibrator:latest .
docker run -p 8000:8000 honest-rl-calibrator:latest
```

For the full **data → train → eval → deploy** pipeline see
[`docs/RUNBOOK.md`](docs/RUNBOOK.md).

### Reproducing the plots

```bash
# Deterministic fallback — no GPU required, always renders the
# representative trajectory committed under docs/training/.
make plots-demo

# From any real run's trainer_state.json (path is derived from --output-dir
# in training/train_grpo.py — defaults to ./honest-<preset>-grpo/):
python bin/plot_training_curves.py \
    --trainer-state ./honest-<preset>-grpo/checkpoint-<step>/trainer_state.json \
    --out docs/training \
    --label "<preset> · <step> steps · <gpu>"

# Side-by-side per-preset (drop in whichever runs you completed):
for preset in qwen0.5b qwen1.5b qwen3b llama1b llama3b phi4mini; do
  state="./honest-${preset}-grpo/checkpoint-final/trainer_state.json"
  [ -f "$state" ] || continue
  python bin/plot_training_curves.py \
      --trainer-state "$state" \
      --out "docs/training_${preset}" \
      --label "${preset} · final"
done
```

---

## Models

`calibration_profiles.py` ships hyperparameter presets across three
capacity tiers. All use 4-bit QLoRA on a single GPU.

| Preset     | Backbone                         | Tier   | Default steps | Recipe                |
| ---------- | -------------------------------- | ------ | ------------- | --------------------- |
| `qwen0.5b` | Qwen/Qwen2.5-0.5B-Instruct       | tiny   | 250           | SFT (warmup) → GRPO   |
| `qwen1.5b` | Qwen/Qwen2.5-1.5B-Instruct       | small  | 250           | SFT (optional) → GRPO |
| `qwen3b`   | Qwen/Qwen2.5-3B-Instruct         | medium | 350           | GRPO direct           |
| `llama1b`  | meta-llama/Llama-3.2-1B-Instruct | tiny   | 250           | SFT (warmup) → GRPO   |
| `llama3b`  | meta-llama/Llama-3.2-3B-Instruct | medium | 350           | GRPO direct           |
| `phi4mini` | microsoft/Phi-4-mini-instruct    | medium | 250           | GRPO direct           |

The **tier** field is operational, not cosmetic. It encodes (i) whether
the base model can satisfy the 3-tag XML contract from the system
prompt alone — *medium*: yes; *small*: usually; *tiny*: no, SFT is
mandatory — and (ii) which OOD slices have measurable accuracy
headroom for the transfer report (`recommended_ood_slices`). See the
per-preset comments in `calibration_profiles.py` for the full
rationale, and the [Empirical findings](#empirical-findings-and-projections)
section for what each tier produces in practice.

Wall-clock is left unspecified intentionally — it varies materially
with GPU, VRAM, batch / accumulation choices, and step count. The
trainer writes per-step seconds into `trainer_state.json` and
`bin/plot_training_curves.py` renders the trajectory. Hardware caps
are applied via `--colab-profile {t4,l4,a100}` and only ever clip
risky values down; they never raise.

> **Tiny tier requires SFT first.** Without it, ~97–98 % of GRPO
> rollouts on Qwen-0.5B / Llama-1B hit the malformed-penalty floor in
> the first 100 steps and `frac_reward_zero_std ≈ 1.0` (verified on
> the pilot runs). The one-command wrapper handles the SFT-then-GRPO
> chain with tier-appropriate hindsight settings:
>
> ```bash
> ./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct
> ./bin/run_calibration_pipeline.sh meta-llama/Llama-3.2-1B-Instruct
> ```
>
> The SFT phase teaches format compliance, a correctness-conditioned
> confidence prior, and the legacy `<hindsight>` tag — so the legacy
> hindsight reward channel actually fires when GRPO starts. See
> [`docs/SELF_LEARNING.md` §2.6](docs/SELF_LEARNING.md#26-bringing-tiny-models-on-line--calibration-sft-warmup)
> for the full SFT design and the metric expectations.

For the 0.5 B / 1 B presets on T4, override
`--gradient-accumulation-steps 4` explicitly — the T4 cap minimum
(16) is sized for Phi-4-mini-3.8 B and makes smaller models train
~4× slower than necessary.

---

## Evaluation metrics

`eval/metrics.py` reports the full calibration battery:

* **ECE** — Expected Calibration Error (15 equal-width bins).
* **ACE** — Adaptive Calibration Error (equal-mass bins).
* **MCE** — Maximum Calibration Error.
* **Brier** — primary training objective; lower is better.
* **NLL** — Negative log likelihood under the model's emitted `c`.
* **AUROC / AUPRC** — discrimination of correct vs incorrect.
* **Reliability diagrams** — `eval/plot_reliability.py`.

Statistical significance: `eval/compare_runs.py` reports a 95%
bootstrap CI on Δ Brier so that small headline numbers are not over-claimed.

---

## Deploying to Hugging Face Spaces

The repository is HF-Spaces-ready out of the box. The
[`Dockerfile`](Dockerfile), [`openenv.yaml`](openenv.yaml), and the
README YAML frontmatter encode the full runtime contract.

```bash
# 1. One-time: install the Hugging Face CLI and log in
pip install -U huggingface_hub
huggingface-cli login   # paste a Write-scope token from huggingface.co/settings/tokens

# 2. Create a new Docker Space
huggingface-cli repo create --type space --space_sdk docker Rushhaabhhh/HONEST-Env

# 3. Wire the Space as a git remote and push
git remote add space https://huggingface.co/spaces/Rushhaabhhh/HONEST-Env

# HF auto-creates a starter README.md on the Space, so the first push
# needs --force to overwrite that initial commit with our repo's main.
git push --force space main
```

The first push triggers a Docker build on Hugging Face's infrastructure
(~3 minutes). Watch the build logs in the Space's "App" tab; once the
status flips from `Building` to `Running`, the env is live at:

```
https://huggingface.co/spaces/Rushhaabhhh/HONEST-Env
```

The Space exposes the standard OpenEnv runtime contract — judges can
verify the deployment from a logged-out browser:

| Endpoint | Expected response |
| -------- | ----------------- |
| `GET  /health`        | `{"status": "healthy"}` |
| `GET  /metadata`      | name, description, version, author |
| `GET  /schema`        | combined action / observation / state JSON schemas |
| `GET  /openapi.json`  | full OpenAPI 3 spec (interactive at `/docs`) |
| `POST /reset`, `/step`| OpenEnv simulation contract |
| `POST /mcp`           | JSON-RPC 2.0 MCP entry point |

Validate the live deployment in one command:

```bash
openenv validate --url https://rushhaabhhh-honest-env.hf.space
```

Expected output: `"passed": true` with all six required criteria green.

If you want a fully reproducible re-deploy, the Hugging Face Space is
**cloneable** (top-right of the Space page → "Duplicate this Space")
and the `git remote add space …` command above lets any user push to
their own namespace.

---

## Deployment: MCP server

After training, expose the calibrated adapter as an MCP tool:

```bash
make mcp-smoke         # offline self-test (no model load)
make mcp-health        # config preflight
make mcp-config        # print a ready-to-paste Claude Desktop config
make mcp-run           # launch the stdio server
# Or one-shot:
bin/install-mcp.sh
```

Two tools are exposed:

* `ask_with_calibrated_confidence(question, domain?)` →
  `{ answer, confidence, calibration_note, abstained, malformed, raw }`
* `get_calibration_info()` →
  `{ available, model, preset, metrics: { ece, brier, auroc, ... }, ood: {...} }`

See [`mcp_server/README.md`](mcp_server/README.md) for Claude Desktop /
Cursor / LangGraph integration recipes and a full troubleshooting
playbook.

---

## Repository layout

```
HONEST-Env/
├── calibration_profiles.py      Per-model presets (Qwen 0.5B/1.5B/3B, Llama 1B/3B, Phi-4-mini)
├── server/                      OpenEnv environment + reward + self-learning
├── data/                        Ingestion, verifiers, unified sampler, processed JSONLs
├── training/                    GRPO trainer, optional format SFT, Colab notebook
├── eval/                        Baseline, full eval, comparison, plots, OOD
├── mcp_server/                  Production MCP wrapper
├── tests/                       Unit + integration tests
├── client/                      OpenEnv client for remote test runners
├── models/                      OpenEnv data classes (Action / Obs / State)
├── bin/install-mcp.sh           One-shot MCP installer / health-check
├── bin/run_server.sh            Local OpenEnv launcher
├── bin/plot_training_curves.py  Render committed loss/reward/KL PNGs
├── bin/install-mcp.sh           Claude Desktop / MCP installer
├── docs/RUNBOOK.md              End-to-end pipeline (data → train → eval → deploy)
├── docs/SELF_LEARNING.md        Research memo for the four self-learning pillars
├── docs/WRITEUP.md              Project writeup / blog
├── docs/training/*.png          Training curves (rendered by bin/plot_training_curves.py)
├── Makefile                     Convenience targets (test, smoke-train, validate, plots-*, mcp-*)
├── Dockerfile                   HF-Spaces-ready container
├── pyproject.toml               Multi-mode deploy + console scripts (`server`, `honest-mcp`)
├── openenv.yaml                 OpenEnv runtime spec (parsed by `openenv validate`)
├── uv.lock                      Pinned resolution for reproducible builds
└── README.md                    This file
```

---

## License & attribution

Datasets retain their upstream licenses (Hendrycks MATH, MBPP, APPS,
ZebraLogic, MMLU, AGIEval). Code in this repository is provided under
its own license.
