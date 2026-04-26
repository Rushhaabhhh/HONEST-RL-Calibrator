# HONEST · Self-Learning Calibration

> Research memo, design and implementation notes for the four self-learning
> pillars added on top of the GRPO pipeline. The goal is *recursive skill
> amplification*: the agent drives its own capability growth instead of
> optimising a fixed task distribution.
>
> Status: experimental. None of these mechanisms are individually published
> for **calibration**. The combination is, to our knowledge, novel.

---

## 0. Problem statement

The base GRPO pipeline (`training/train_grpo.py`) trains an LLM to emit
*honest* confidence under a Brier-score reward. It is a fixed-task RL loop:

```
sample (prompt, gt) ── π ──► (answer, conf) ── R(c, y) ──► ∇θ J
        ▲                                                    │
        └─────────── DifficultyController ◄──────────────────┘
```

Two things are missing for **self-learning** in the strong sense
("agents that learn to generate new challenges, escalate difficulty, and
improve through self-play or adaptive curricula"):

1. The agent never **revises** its own confidence after seeing the outcome.
2. The curriculum has a **fixed ceiling** (d=5) and a **fixed task source**
   (the unified sampler).

We close both gaps with four composable mechanisms, each opt-in via a CLI
flag so they can be ablated cleanly.

---

## 1. The four pillars

| Pillar                              | Acronym | Inspiration                | What it adds                                                                    | Cost      |
| ----------------------------------- | ------- | -------------------------- | ------------------------------------------------------------------------------- | --------- |
| Hindsight Calibration Reward        | **HCR** | HER (Andrychowicz 2017)    | Two-step protocol: answer, see GT, emit retrospective confidence. Auxiliary reward. | +1 fwd pass |
| Calibration-Prioritized Replay      | **CPR** | PER (Schaul 2015)          | Buffer of past prompts, re-sampled by `\|conf − correct\|`.                      | O(buf)    |
| Self-Mutating Curriculum            | **SMC** | POET (Wang 2019)           | When d=5 acc > τ, mutate d=5 problems into d=6,7,...                            | rule-based|
| Generator/Solver Self-Play          | **GSS** | PAIRED (Dennis 2020)       | A frozen LLM proposes problems; rewarded for solver's calibration error.        | extra LLM |

HCR + CPR address gap 1; SMC + GSS address gap 2. SMC is implemented as a
production-ready rule system; GSS ships as a stubbed protocol with a
deterministic fallback generator (real generator-policy training is left
as v2 because it requires its own RL loop).

---

## 2. Pillar 1 — Hindsight Calibration Reward (HCR)

### 2.1 Theory

Let `y ∈ {0,1}` be the correctness indicator and `c` the confidence the
agent emitted *before* seeing GT. The Brier reward

$$R_B = -(c - y)^2$$

trains the **forward** confidence head. After the answer is graded, we
reveal `y` and ask the agent for a **retrospective** confidence `r`:

$$R_H = -k \cdot (r - y)^2,\quad k = 0.3$$

Both are proper scoring rules (Gneiting & Raftery 2007), so adding HCR to
the loss does **not** introduce a perverse incentive. Geometrically, HCR's
gradient

$$\frac{\partial R_H}{\partial r} = -2k(r - y)$$

points the same direction as Brier's, but conditional on a strictly larger
information set (the agent has now seen `y`). The retrospective head can
therefore reach the optimal `r* = y` more easily, and via parameter sharing
it pulls the forward head toward calibration.

### 2.2 Why this is not just "ground-truth supervision"

Two reasons:
1. The gradient flows through the model's reasoning trace, not a label.
   The model learns *which kinds of reasoning correlate with being right*.
2. It works for **abstain** too: optimal retrospective confidence after an
   abstain is undefined, so HCR is **only** active after a real
   AnswerAction. This prevents the degenerate "always abstain" exploit.

### 2.3 Action protocol

```
S_t  : (question_t, episode_step_t)
A_t  : <answer>x</answer><confidence>c</confidence>
S_t+1: question_t+1, previous_correctness=y, revealed_answer=gt   ← already in §8.4
A_t+1: <hindsight>r</hindsight>            ← NEW; ε-probability per step
S_t+2: next problem
```

`<hindsight>r</hindsight>` is parsed by `server.hindsight.parse_hindsight`.
If the model emits a regular Answer/Abstain at step t+1 instead, the
hindsight slot is silently skipped — so HCR is *opt-in for the model* and
the policy can choose to ignore it. An advantage signal still flows because
emitting a well-calibrated retrospective confidence has positive expected
reward whenever the agent is uncertain.

### 2.4 Reward integration

HCR is a separate `reward_hindsight(...)` function passed to TRL alongside
`reward_brier`. It returns 0 for every completion that is *not* a
HindsightAction, so it adds no noise to forward-only training. Weighting
defaults to `0.3` (auxiliary reward, intentionally smaller than the
primary Brier signal so it shapes behaviour without dominating it).

### 2.5 v2 — Calibration-Aware Self-Refinement (CASR)

> Status: shipped in `server.hindsight_v2`, opt-in via `--hindsight-mode refined`.
> The legacy v1 head from §2.1–§2.4 is preserved for reproducibility and stays
> the default.

#### Why v2 exists — diagnosing the v1 silent channel

In the v1 design above, the trainer-time hindsight reward
(`make_train_time_hindsight_reward` in `training.train_grpo`) returns
`-k(r-y)²` when the completion contains a `<hindsight>` tag, and `0.0`
otherwise. We observed in the Qwen-1.5B run (350 GRPO steps) that this
reward channel was **identically zero on every step** —
`bin/audit_hindsight.py` confirms this empirically. Three independent root
causes compound:

1. **The system prompt never describes `<hindsight>`.** The base model has
   zero prior on the tag and never emits it, so `parse_hindsight()` falls
   through to "malformed" on 100 % of completions.
2. **The reward gates AND, with no positive gradient toward the tag.**
   The reward is `-k(r-y)² ≤ 0` — emitting hindsight can only *cost*
   reward, never earn it. There is no incentive structure that pulls
   the policy toward producing the tag in the first place. Chicken-and-egg.
3. **The design is informationally redundant with Brier.** Inside one
   completion, `c` and `r` are emitted from the same context window and
   graded against the same `y` with the same scoring rule. The optimal
   policy under both rewards combined is `c = r = E[y|x]` — *identical*
   to the optimal policy under Brier alone. No new information enters
   the gradient. Compare to true HER (Andrychowicz 2017), where
   re-labelling injects new information from the realised outcome.

#### v2 design — reward refinement, not retrospection

Instead of asking for a redundant retrospective number, CASR asks the
model to do something genuinely useful in a single pass: **critique its
own answer and refine the confidence**. The completion now contains
five tags:

```
<reasoning>...</reasoning>
<answer>X</answer>
<confidence>c</confidence>
<critique>spot any errors in the reasoning above</critique>
<refined_confidence>r</refined_confidence>
```

The reward decomposes into four terms with carefully designed gradients:

$$R_h = \alpha \cdot \underbrace{[(c-y)^2 - (r-y)^2]}_{\Delta\text{Brier}} \;+\; \beta \cdot \mathbb{1}[\text{critique}_{\text{ok}}] \;-\; \gamma \cdot \mathbb{1}[r \approx c] \;-\; \delta \cdot \mathbb{1}[\text{partial}]$$

with defaults `α=1.0, β=0.05, γ=0.05, δ=0.05`, the final scalar clipped
to `±0.30` so the head cannot dominate the primary Brier signal.

| Term | Triggers when … | Why it's needed |
| ---- | --------------- | --------------- |
| `α·ΔBrier` | full structure + graded answer | Core gradient. POSITIVE iff the refinement actually improved calibration. |
| `+β` (format bonus) | non-trivial critique present (≥16 chars) | Provides the *positive* gradient that v1 was missing — pulls the policy toward emitting the new tags from cold start. |
| `−γ` (anti-copy) | `\|r-c\| < 0.02` | Prevents the trivial-copy exploit (set `r=c` and farm β with no real refinement). |
| `−δ` (partial structure) | critique XOR refined_confidence | Forces the model to commit to the protocol; emits both or neither. |

**Why this provides signal Brier alone cannot:**

- *Already-calibrated case:* `ΔBrier ≈ 0` and the anti-copy penalty fires
  ⇒ reward goes to 0. No double-counting on Brier-optimal completions.
- *Mis-calibrated case:* refining `r` toward `y` after critique gives
  positive `ΔBrier`. The gradient flows *through the critique trace* —
  the model learns *which patterns of critique correlate with successful
  re-calibration*, not just final numbers.
- *Wasted-step case (zero-σ groups):* when GRPO rollouts agree on `(c, y)`,
  group-relative advantage on Brier collapses. But if 2/4 rollouts emit a
  critique and 2/4 don't, the format bonus produces non-zero advantage —
  recovering signal that the primary reward loses.

#### Research grounding

CASR combines four lines of recent work, none of which addresses
calibration directly but each of which contributes a piece:

| Paper | Year | Contribution |
| ---- | --- | ------------ |
| Self-Refine (Madaan et al., NeurIPS) | 2023 | Iterative self-critique improves single-pass LLM outputs. |
| Self-Verification (Weng et al., EMNLP) | 2023 | Asking the model to verify its own answer reduces hallucination AND improves calibration. |
| Process Reward Models (Cobbe et al.) | 2021 | Step-level verification correlates strongly with outcome correctness — a critique step is a learnable signal. |
| Reflexion (Shinn et al., NeurIPS) | 2023 | Verbal self-reflection beats next-token prediction alone for sequential decision making. |
| HER (Andrychowicz et al., NeurIPS) | 2017 | The original idea that hindsight relabelling injects new information into the gradient. CASR's "new information" is the model's own critique, not an exogenous goal-relabel. |

#### Predicted impact on each metric

| Metric | Mechanism |
| ------ | --------- |
| **ECE / Brier ↓** | `ΔBrier` is *literally* the calibration-error-improvement signal — a direct optimisation target on the same scoring rule as the primary reward, but conditional on a strictly larger info set (the critique). |
| **Wasted steps (σ_R=0) ↓** | Format bonus produces non-zero group advantage when rollouts agree on `(c, y)` but differ on critique emission. New gradient channel. |
| **Format compliance ↑** | Structural bonus generalises: a model rewarded for cleanly emitting *new* tags gets pulled toward cleanly emitting *all* tags. |
| **Logic / hard-domain accuracy ↑** | Self-Refine and Reflexion show critique steps measurably improve reasoning on multi-step problems — exactly the domain where the v1 run saw 0% logic accuracy. |

#### How to enable

```bash
python training/train_grpo.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --hindsight \
    --hindsight-mode refined          # ← the new flag, default = "legacy"
    # reasoning_mode is auto-promoted to "refined" so the prompt teaches
    # the <critique> and <refined_confidence> tags. No other flags change.
```

Key invariants:

- `--hindsight-mode legacy` (the default) is **bit-for-bit identical** to
  the v1 path — in-flight runs see no behavioural change.
- `--hindsight-mode refined` auto-switches `--reasoning-mode refined` so
  the system prompt actually describes the new tags. Setting both
  explicitly is fine (no double-promotion).
- The CASR reward is silent (returns `0.0`) on completions that emit no
  refinement structure, so during early training when the model is still
  learning the new tags, the head adds zero noise to standard rollouts.

#### How to verify hindsight is firing in any run

```bash
python bin/audit_hindsight.py --trainer-state ./honest-qwen-1-5b-grpo/trainer_state.json
```

Output reports the fraction of steps the hindsight head returned non-zero.
On the legacy v1 run with `reasoning_mode=required`, this is ~100 % zero
(silent channel). On a CASR run with `reasoning_mode=refined`, expect
non-zero on most steps within ~50 GRPO steps of cold start (the format
bonus drives initial emission of the new tags).

### 2.6 Bringing tiny models on-line — Calibration SFT warmup

The two hindsight modes above (legacy v1, refined CASR) both assume the
base model can already emit the strict 3-tag XML format from the system
prompt alone. Empirically that is true for Qwen-3B and larger; it is
**catastrophically false** for Qwen-0.5B and Llama-1B. On those tiers,
~97 % of GRPO rollouts hit the malformed-penalty floor in the first 100
steps, `frac_reward_zero_std ≈ 1.0`, and the GRPO advantage signal is
identically zero — the model never receives any calibration gradient.

The fix is not subtle: a single short Calibration SFT pass before the RL
phase, taught by `training/calibration_sft.py`. Each SFT example bundles
three priors into a single assistant target:

1. **Format compliance** — every target uses the exact 3-tag contract
   (or `<abstain/>`) so the model sees the strict format thousands of
   times before GRPO starts grading it.
2. **Correctness-conditioned confidence prior** — when the SFT target's
   answer is the actual ground truth the confidence is sampled from a
   high-band (≈ 0.85 ± 0.10); when it has been deliberately perturbed
   the confidence is sampled from a low-band (≈ 0.25 ± 0.15). The model
   learns "wrong answer → low confidence" *before* GRPO ever shapes it.
3. **Hindsight tag prior** — half of the examples include
   `<hindsight>r</hindsight>` with `r` bound to the ground-truth
   correctness of the displayed answer. This is precisely what
   `server.hindsight.compute_hindsight_reward` grades, so once SFT runs
   the legacy hindsight reward channel actually fires during GRPO
   instead of staying at 0.0 forever.

#### Tier-aware defaults

`calibration_profiles.py` tags each preset with a `tier` (`tiny` /
`small` / `medium`) and four SFT recommendations: `n_examples`,
`epochs`, `max_difficulty`, `hindsight_frac`. The SFT script auto-resolves
all four from `--model-id`:

| Preset      | tier   | sft_n | epochs | max_d | hindsight_frac | recommended `--hindsight-mode` |
|-------------|--------|-------|--------|-------|----------------|--------------------------------|
| qwen0.5b    | tiny   | 1500  | 2      | 2     | 0.50           | legacy                         |
| llama1b     | tiny   | 1500  | 2      | 2     | 0.50           | legacy                         |
| qwen1.5b    | small  | 1000  | 2      | 3     | 0.40           | refined                        |
| qwen3b      | medium | 600   | 1      | 4     | 0.30           | refined                        |
| llama3b     | medium | 700   | 1      | 4     | 0.30           | refined                        |
| phi4mini    | medium | 500   | 1      | 4     | 0.30           | refined                        |

CASR is intentionally *not* recommended for tiny models — it asks the
model to critique its own reasoning, which requires a generative
capacity 0.5B / 1B simply does not have. Legacy hindsight is a tractable
self-prediction regression target that fits comfortably inside a tiny
LoRA once the SFT phase has taught the tag.

#### One-command recipe

```bash
# Tiny models — SFT is REQUIRED.
./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct
./bin/run_calibration_pipeline.sh meta-llama/Llama-3.2-1B-Instruct

# Medium models — SFT optional but accelerates calibration.
./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-3B-Instruct
```

The script chains:

1. `python training/calibration_sft.py --model-id ... --output-dir ./sft-<slug>`
2. `python training/train_grpo.py --model-id ... --init-adapter ./sft-<slug> --hindsight --hindsight-mode {legacy|refined}`

Pass extra GRPO args after the model id; pass `--skip-sft` to skip the
warmup phase entirely (only sensible on medium tier or when reproducing
a baseline).

#### What to look for in the run

* The `--init-adapter` warmup increases initial format compliance from
  ~0–3 % to ~85–95 % at GRPO step 0.
* `frac_reward_zero_std` drops from ~1.0 (no signal) to ~0.2–0.4 within
  the first 30 GRPO steps.
* The legacy hindsight channel returns non-zero for the majority of
  steps (verify with `bin/audit_hindsight.py`) — same diagnostic as for
  the medium-tier CASR runs.
* Brier reward visibly *moves*: for Qwen-0.5B, expect a trajectory from
  ~ -1.2 → -0.8 over 250 steps; for Llama-1B, ~ -1.3 → -0.85. Absolute
  numbers are softer than the 3B presets but the *shape* of the curve
  finally exists, which is the whole point of demonstrating calibration
  on small models.

---

## 3. Pillar 2 — Calibration-Prioritized Replay (CPR)

### 3.1 Theory

PER (Schaul 2015) replays transitions with probability

$$p_i = \frac{|TD_i|^\alpha}{\sum_j |TD_j|^\alpha}$$

For calibration the natural priority is **calibration error**:

$$p_i = \frac{(|c_i - y_i| + \epsilon)^\alpha}{\sum_j (|c_j - y_j| + \epsilon)^\alpha}$$

A perfectly-calibrated example (`|c-y| = 0`) is replayed with weight `ε`
(rare). A maximally-miscalibrated example (`c=1, y=0` or `c=0, y=1`) is
replayed at full weight.

### 3.2 Why this matters for GRPO

GRPO computes group-relative advantages within a single prompt's rollouts.
With a fixed prompt distribution, *uniformly-easy* prompts (c=y trivially)
contribute zero advantage and waste compute. CPR shifts the prompt
distribution toward the model's current calibration frontier — exactly the
high-information zone.

### 3.3 Implementation

`server.replay_buffer.CalibrationPrioritizedReplay` exposes:

```python
buffer.add(prompt, gt, domain, difficulty, conf, correct)
buffer.sample(n, alpha=0.6, eps=1e-3) -> list[dict]
buffer.snapshot() -> dict     # for logging
```

Internally a single ring buffer (default 4096 entries) with importance
weights re-computed lazily on `sample()`. We do *not* implement
sum-tree; for ≤10K entries the linear-scan sampling is < 1ms and avoids a
non-trivial dependency.

### 3.4 Wiring

`build_prompt_dataset(...)` keeps its current "fresh sampler" behaviour
for the first `--replay-warmup` steps (default 100). Once the buffer is
warm, each fresh prompt is drawn from the buffer with probability
`--replay-mix` (default 0.3) and from the unified sampler otherwise. This
keeps the curriculum from getting stuck: 70% of the time we still trust
the controller, 30% we revisit our own recent miscalibration.

The reward wrapper writes back into the buffer after every group rollout
(majority-vote correctness, mean confidence, same aggregation as the
controller feedback path).

---

## 4. Pillar 3 — Self-Mutating Curriculum (SMC)

### 4.1 Theory

POET (Wang et al. 2019) and PAIRED (Dennis et al. 2020) both rely on a
*generator* that proposes increasingly difficult environments. SMC is the
deterministic, rule-based version:

> When rolling accuracy at the controller's max difficulty crosses an
> upper threshold for at least `min_episodes_at_max` episodes, the
> controller *raises its ceiling* by one tier. Problems at the new tier
> are produced by mutating sampled-tier-N problems via a registered
> mutator pipeline.

This makes the curriculum **unbounded** in principle. In practice we cap
at `MAX_DIFFICULTY_HARD = 8` to avoid pathological mutator chains.

### 4.2 The three deterministic mutators

All three preserve a verifiable ground truth, which is essential — RL
without a graded signal collapses.

#### 4.2.1 Numeric mutator (math)

```
Original: "3 × 4 + 5"  → 17
Mutated : "37 × 41 + 53" → 1570
```

Multiply every literal numeric token by a per-problem random factor
`s ∈ {7, 11, 13, …, 97}` (small primes to avoid trivial common factors).
Ground truth is recomputed by re-evaluating the AST. Verified by reusing
the math verifier.

#### 4.2.2 Compositional mutator (any domain)

```
Original P1: "What is 2^5?"   → 32
Original P2: "What is 3 × X?" → ?
Mutated   : "Let X = answer to (P1). What is (P2)?"  → 96
```

Chain two same-domain problems P1, P2 of the **current** max difficulty.
The mutator substitutes a placeholder `X` in P2's question with the GT of
P1. Verifier: P2's verifier on the recomputed GT.

This mutator is *the recursive amplification primitive*: every time the
ceiling rises, today's hard problems become tomorrow's primitives.

#### 4.2.3 Distractor mutator (any domain)

```
Original: "What is 7^4 mod 11?"
Mutated : "Yesterday Alice baked 12 cookies. ... <100 tokens of irrelevant prose>
           ... What is 7^4 mod 11?"
```

Prepend irrelevant-but-plausible context drawn from a 50-snippet pool.
GT and verifier are unchanged. Tests robustness to long context and
distraction — a known calibration weakness in small models.

### 4.3 Promotion / demotion logic

`server.mutators.SelfMutatingCurriculum` wraps the existing
`DifficultyController`:

```python
smc = SelfMutatingCurriculum(controller, max_hard_difficulty=8,
                             promote_threshold=0.75,
                             min_episodes_at_max=20)
smc.maybe_promote(domain)  # called after every record_outcome
problem = smc.sample(domain, rng)  # routes to base sampler or mutator
```

Demotion (lowering the ceiling) is symmetric: if rolling acc at the
mutated tier drops below `demote_threshold = 0.20` for `min_episodes_at_max`
episodes, the ceiling collapses by one. This protects against the curriculum
running away when the model has actually regressed.

### 4.4 Logging

The current ceiling per domain is exposed in
`DifficultyController.snapshot()` as `max_unlocked_difficulty` and
plotted by `DifficultyControllerLogCallback` to W&B as
`difficulty/{domain}/ceiling`.

---

## 5. Pillar 4 — Generator/Solver Self-Play (GSS)

### 5.1 Theory

PAIRED (Dennis 2020) trains a generator-protagonist pair against a
solver-antagonist. The generator's reward is the *regret* — the gap
between the antagonist's performance and the protagonist's. For
calibration we substitute regret with **calibration error**:

$$R_G(p) = |c_S(p) - y(p)|$$

where `p` is the generated problem, `c_S(p)` is the solver's confidence
on it, and `y(p)` is whether the solver was correct. This pushes the
generator toward the **learning frontier** — problems where the solver is
neither hopelessly lost nor trivially confident.

### 5.2 Why we ship a stub for v1

Training a generator requires its own RL loop, its own dataset, and its
own KL-stable schedule. We have ~hours of compute budget. So:

- **v1 (shipped)**: a stubbed deterministic generator that *samples* from
  the existing unified sampler + applies a pillar-3 mutator. Effectively a
  "generator policy = identity + random-mutator". This still exercises
  the GSS protocol end-to-end.
- **v2 (roadmap)**: replace the stub with a frozen LLM problem-generator
  whose outputs are filtered for verifiability. Promote to "trainable"
  when the calibration-error signal stabilises.

### 5.3 Protocol

```python
generator = ProblemGenerator(...)            # stub or LLM
solver    = the_grpo_model                   # the policy under training
loop:
    p = generator.propose()
    a, c = solver.answer(p)                  # via env.step
    y = verify(a, p.gt)
    r_solver    = -(c - y)^2  + format_bonus    # ← Pillar 1+2
    r_generator = |c - y|                       # high if solver miscalibrated
    update(solver, r_solver)
    update(generator, r_generator)              # ← stubbed in v1
```

In v1, `update(generator, ...)` is a no-op; the generator's diversity
is provided by the deterministic mutator pool.

### 5.4 Hooks

`server.self_play.SelfPlayLoop.run_step()` returns a typed
`SelfPlayTransition` so a future generator policy can be slotted in
without changing the caller. The loop is exercised by a separate flag
`--self-play` on the trainer; default is off.

---

## 6. Composability matrix

|           | HCR | CPR | SMC | GSS |
| --------- | --- | --- | --- | --- |
| HCR       | —   | ✓   | ✓   | ✓   |
| CPR       | ✓   | —   | ✓   | ✓   |
| SMC       | ✓   | ✓   | —   | partially overlaps |
| GSS       | ✓   | ✓   | ⚠   | —   |

SMC and GSS partially overlap (both produce harder problems). Recommended
combinations:

- **Minimum-risk**: HCR alone. Adds one auxiliary reward, isolated from
  the curriculum.
- **Recommended default**: HCR + SMC. Best demonstration of
  "self-learning": hindsight + recursive curriculum, with the smallest
  surface area of additional risk.
- **Maximum**: HCR + CPR + SMC. GSS only after the first three have
  shipped a working before/after delta.

---

## 7. CLI flags

```bash
python training/train_grpo.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --colab-profile l4 \
  --max-steps 350 \
  # ─ Self-learning ─
  --hindsight                 # Pillar 1 (HCR)
  --hindsight-prob 0.3        # Probability of injecting a hindsight slot per step
  --hindsight-weight 0.3      # k in §2.1
  --replay-priority           # Pillar 2 (CPR)
  --replay-buffer-size 4096
  --replay-mix 0.3
  --replay-warmup 100
  --replay-alpha 0.6
  --self-mutate               # Pillar 3 (SMC)
  --smc-max-hard-difficulty 8
  --smc-promote-threshold 0.75
  --self-play                 # Pillar 4 (GSS) — v1 stubbed generator
```

All four flags default to **off** so the existing pipeline is unchanged.

---

## 8. Evaluation protocol

For each of the four pillars we report:

1. **Δ ECE** vs. the base GRPO run (same seed, same data, same steps).
2. **Δ Brier**.
3. **Mean reward trajectory** — does the auxiliary reward stabilise?
4. **Curriculum trajectory** — `target_difficulty` and (for SMC)
   `max_unlocked_difficulty` over time.
5. **Calibration histogram** — sanity-check the model isn't collapsing
   onto a degenerate confidence value.

A pillar is considered to "work" if Δ ECE ≤ -0.01 with the same step
budget. A pillar that does not pass this bar is reported under
"experiments we ran" rather than as a headline result.

---

## 9. Failure modes & guardrails

| Failure                                          | Detector                                             | Guardrail                                       |
| ------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| HCR collapses to all-0.5 retrospective confidence| `RewardHealthCallback` (existing)                    | Falls back to brier-only when reward_std<1e-4   |
| CPR replays the same 1-2 prompts forever         | `buffer.entropy_of_priorities()` < log(2)            | Auto-disable replay mix for next 50 steps       |
| SMC ceiling races up too fast (one lucky window) | Cool-down identical to base controller (10 episodes) | + min_episodes_at_max = 20                      |
| SMC mutator produces unverifiable problems       | Verifier returns False on its own GT                 | Drop the mutated problem, retry up to 3 times   |
| GSS generator collapses to one trivial problem   | `len(set(generated_pids)) / n_steps < 0.1`           | Force re-seed of the stub generator             |

All five detectors are wired into `server.health` (new module) and emit
W&B events.

---

## 10. References

- Andrychowicz et al. (2017). *Hindsight Experience Replay*. NeurIPS.
- Schaul et al. (2015). *Prioritized Experience Replay*. ICLR.
- Wang et al. (2019). *POET: Open-Ended Coevolution of Environments and
  their Optimized Solutions*. GECCO.
- Dennis et al. (2020). *Emergent Complexity and Zero-shot Transfer via
  Unsupervised Environment Design (PAIRED)*. NeurIPS.
- Gneiting & Raftery (2007). *Strictly Proper Scoring Rules*. JASA.
- Kadavath et al. (2022). *Language Models (Mostly) Know What They Know*.
  Anthropic.
- Damani et al. (2024). *RLCR: Reinforcement Learning with Calibration
  Rewards*.

---

## 11. Code artefacts produced for this memo

| File                                  | Pillar | Lines (approx) |
| ------------------------------------- | ------ | -------------- |
| `server/hindsight.py`                 | 1      | ~190           |
| `server/replay_buffer.py`             | 2      | ~210           |
| `server/mutators.py`                  | 3      | ~290           |
| `server/self_play.py`                 | 4      | ~210           |
| `server/environment.py` (additions)   | 1,3    | +60            |
| `training/train_grpo.py` (additions)  | all    | +90            |
| `tests/test_hindsight.py`             | 1      | ~120           |
| `tests/test_replay_buffer.py`         | 2      | ~100           |
| `tests/test_mutators.py`              | 3      | ~110           |
| `tests/test_self_play.py`             | 4      | ~80            |

All four pillars are independently testable, independently togglable, and
collectively additive to the existing GRPO trainer.
