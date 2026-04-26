# HONEST · Operational Runbook

End-to-end pipeline: **data → RL training → OOD eval → comparison →
success metrics → MCP deployment → self-learning verification**.

Every step is a single command. Every command produces a JSON or
markdown artifact that the next step consumes. There are no implicit
dependencies between steps; each one fails fast if its inputs are
missing.

```
┌───────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ 1 · Ingestion │─▶│ 2 · Baseline   │─▶│ 3 · GRPO train │─▶│ 4 · Full eval  │
│   (data/)     │  │   (anchor)     │  │   (LoRA out)   │  │  (ID + OOD)    │
└───────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
                                                                    │
┌───────────────┐  ┌────────────────┐  ┌────────────────┐           │
│ 7 · Self-     │◀─│ 6 · MCP serve  │◀─│ 5 · Compare    │◀──────────┘
│   learning    │  │   (deploy)     │  │   (Δ + CI)     │
└───────────────┘  └────────────────┘  └────────────────┘
```

---

## 0 · Prerequisites

```bash
# From the project root
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Optional: log in to W&B and HuggingFace
venv/bin/wandb login
venv/bin/huggingface-cli login

# Activate (we'll prefix commands with ./venv/bin/python below)
source venv/bin/activate          # POSIX shells
```

Verify the environment is healthy before running anything expensive:

```bash
make test                # 438 unit + integration tests should pass
make smoke-train         # train_grpo --dry-run --hindsight --replay-priority --self-mutate --self-play
make mcp-smoke           # offline MCP self-test
```

---

## 1 · Data ingestion

The unified sampler reads `data/processed/{math,code_mbpp,code_apps,logic}.jsonl`.
If any of those files is missing, `train_grpo.py` fails fast at startup
with a clear error.

```bash
# Math (Hendrycks MATH, 7 subjects, 5 difficulty levels)
PYTHONPATH=. python data/ingestion/ingest_hendrycks_math.py
# → data/processed/math.jsonl  (~12.5k problems)

# Code · MBPP (sandboxed verifier)
PYTHONPATH=. python data/ingestion/ingest_mbpp.py
# → data/processed/code_mbpp.jsonl  (~427 problems)

# Code · APPS (streamed JSONL shards from HuggingFace)
PYTHONPATH=. python -m data.ingestion.ingest_apps
# → data/processed/code_apps.jsonl

# Logic · ZebraLogic-style CSP puzzles (regenerated, unique-solution)
PYTHONPATH=. python data/ingestion/regenerate_zebralogic.py
# → data/processed/logic.jsonl  (~75 problems)
```

Every ingestion script prints a JSON summary on the last line — capture
it if you want to assert dataset shape in CI.

### Sanity check

```bash
./venv/bin/python -c "
from data.sampler.unified_sampler import get_sampler
s = get_sampler()
print('total:', s.total_count())
print('buckets (domain, difficulty):')
for k, v in sorted(s.bucket_counts().items()):
    print(' ', k, '=', v)
"
```

You should see ~13k problems spread across (math, 1..5),
(code, 3..5), (logic, 1..5). A bucket with `0` count will *not* error —
the controller will just never dispense that condition.

### OOD data (5-slice transfer suite)

OOD data is fetched from public HuggingFace datasets and is **never**
seen during training. The five-slice suite spans the difficulty range
that small *and* medium models can engage with:

| slice          | source (HF dataset)                         | floor | tiny-model headroom |
|----------------|---------------------------------------------|-------|---------------------|
| `commonsense`  | `tau/commonsense_qa` (validation)           | 0.20  | ~30-45 %            |
| `science_easy` | `allenai/ai2_arc` ARC-Easy (test)           | 0.25  | ~45-60 %            |
| `science_hard` | `cais/mmlu` astronomy (test)                | 0.25  | ~25-35 %            |
| `medical`      | `cais/mmlu` professional_medicine (val)     | 0.25  | floor (medium+ only)|
| `legal`        | AGIEval LSAT-LR (with MMLU law fallback)    | 0.20  | floor (medium+ only)|

**Why five slices instead of two?** The transferability claim ("RL-trained
calibration generalises to OOD") is empirically *unprovable* on slices
where the model sits at the random-MCQ floor: ECE/Brier deltas collapse
into bootstrap noise. Tiny models (Qwen-0.5B, Llama-1B) only have
measurable headroom on `commonsense`/`science_easy`/`science_hard`, so
the tier-aware fetcher and `full_eval.py --ood-slices auto` skip the
hard slices for those models.

```bash
# Tier-aware fetch (recommended — picks slices appropriate to your tier)
PYTHONPATH=. python eval/ood/fetch_ood_data.py --tier tiny     # 3 slices: commonsense + science_easy + science_hard
PYTHONPATH=. python eval/ood/fetch_ood_data.py --tier small    # 4 slices: tiny + medical
PYTHONPATH=. python eval/ood/fetch_ood_data.py --tier medium   # 5 slices: full suite
PYTHONPATH=. python eval/ood/fetch_ood_data.py --tier all      # alias for medium

# Or pick exactly what you want:
PYTHONPATH=. python eval/ood/fetch_ood_data.py --slices commonsense,science_easy --n 200

# Default (no flags) = full medium suite, backward-compatible with prior reports.
PYTHONPATH=. python eval/ood/fetch_ood_data.py --n 200
```

Outputs (one JSONL per slice in `eval/ood/`, auto-discovered by `full_eval.py`):
`commonsense_qa_sample.jsonl`, `arc_easy_sample.jsonl`,
`mmlu_astronomy_sample.jsonl`, `medqa_sample.jsonl`, `lsat_sample.jsonl`.

---

## 2 · Baseline characterization

Before training, anchor the **pre-RL** behaviour of the same model on
the same evaluation conditions. Without this anchor, the post-RL
numbers are uninterpretable.

```bash
# Default: 100 samples × 3 domains × 5 difficulties (~1500 generations)
./venv/bin/python eval/baseline_eval.py \
    --model         Qwen/Qwen2.5-3B-Instruct \
    --model-preset  qwen3b \
    --output        eval/baseline_results.json

# Headline run (200 samples per condition, ~3000 generations)
./venv/bin/python eval/baseline_eval.py \
    --model         Qwen/Qwen2.5-3B-Instruct \
    --model-preset  qwen3b \
    --samples       200 \
    --output        eval/baseline_results.json
```

Outputs (`eval/baseline_results.json`):

```jsonc
{
  "model_id":        "Qwen/Qwen2.5-3B-Instruct",
  "preset":          "qwen3b",
  "n_samples":       100,
  "metrics": {
    "ece": 0.18,    "ace": 0.21,    "mce": 0.42,
    "brier": 0.27,  "nll": 0.71,
    "auroc": 0.62,  "auprc": 0.55,
    "format_rate": 0.82,    "abstain_rate": 0.04
  },
  "per_domain":  { "math": {...}, "code": {...}, "logic": {...} },
  "per_difficulty": { "1": {...}, ..., "5": {...} }
}
```

If `format_rate < 0.70`, the strict XML parser is rejecting too many
completions. **Run the optional Stage-2 format SFT** first:

```bash
./venv/bin/python training/format_sft.py \
    --model-id     Qwen/Qwen2.5-3B-Instruct \
    --output-dir   ./honest-qwen-3b-sft
```

Then resume from the SFT adapter (`--resume-from ./honest-qwen-3b-sft`)
in Step 3.

---

## 3 · GRPO calibration training

Single command. Defaults come from `calibration_profiles.py`; CLI flags
override per-run.

### 3a · Vanilla GRPO (recommended starting point)

```bash
./venv/bin/python training/train_grpo.py \
    --model-preset    qwen3b \
    --colab-profile   l4 \
    --max-steps       350 \
    --output-dir      ./honest-qwen-3b-grpo
```

### 3a-bis · Tiny models (Qwen-0.5B, Llama-1B) — SFT first, GRPO second

Tiny models cannot reliably emit the 3-tag XML format from the system
prompt alone. Without a Calibration SFT warmup, ~97 % of GRPO rollouts
hit the malformed-penalty floor and the GRPO advantage signal stays at
zero. The provided one-command recipe handles both phases:

```bash
# Qwen-0.5B (Colab T4 friendly, ~10 min SFT + ~50 min GRPO on T4)
./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct

# Llama-1B
./bin/run_calibration_pipeline.sh meta-llama/Llama-3.2-1B-Instruct

# Override anything by appending GRPO flags after the model id:
./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct \
    --max-steps 200 --replay-priority --self-mutate
```

The script picks tier-aware defaults from `calibration_profiles.py`:
1500 SFT examples × 2 epochs at max_difficulty=2 with hindsight tag
on half the examples, followed by GRPO with `--init-adapter` pointing
at the SFT output and `--hindsight-mode legacy` (the tractable head
for tiny models).

For deeper detail and what to expect from the metrics see
[`docs/SELF_LEARNING.md` §2.6](SELF_LEARNING.md#26-bringing-tiny-models-on-line--calibration-sft-warmup).

To reproduce the SFT phase manually (e.g. to swap in a different mix):

```bash
./venv/bin/python training/calibration_sft.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --output-dir ./sft-qwen-0.5b \
    --n-examples 1500 --epochs 2 --hindsight-frac 0.5

./venv/bin/python training/train_grpo.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --init-adapter ./sft-qwen-0.5b \
    --hindsight --hindsight-mode legacy \
    --max-steps 250 --output-dir ./honest-qwen-0-5b-grpo
```

### 3b · GRPO + self-learning (recommended for the headline result)

```bash
./venv/bin/python training/train_grpo.py \
    --model-preset    qwen3b \
    --colab-profile   l4 \
    --max-steps       350 \
    --hindsight \
    --self-mutate \
    --replay-priority \
    --output-dir      ./honest-qwen-3b-grpo
```

`--self-play` is supported but disabled by default — turn it on only
after the first three pillars show a clean Δ ECE.

### 3c · What you get

```
honest-qwen-3b-grpo/
├── final_adapters/             ← LoRA adapter (load with --adapter-path)
├── difficulty_state.json       ← rolling controller state per domain
├── smc_state.json              ← (if --self-mutate) ceiling per domain
├── replay_state.json           ← (if --replay-priority) buffer snapshot
├── checkpoint-50/              ← intermediate checkpoints
├── checkpoint-100/
└── trainer_state.json
```

### 3d · Live monitoring

W&B (or stdout if `--no-wandb`) reports the metrics that matter for a
healthy GRPO run:

| Metric                             | Healthy range                      |
| ---------------------------------- | ---------------------------------- |
| `train/reward`                     | rising, plateauing > 0             |
| `train/reward_std`                 | > 1e-4 (else dead-batch guard fires)|
| `train/kl`                         | < 0.05 (else AdaptiveBetaCallback) |
| `train/grad_norm`                  | < `max_grad_norm = 1.0`            |
| `controller/<domain>/target`       | tracking accuracy band (0.30–0.70) |
| `smc/<domain>/max_unlocked`        | promotes only when ready (≥ d=6+)  |
| `replay/buffer_size`               | rises to ~ buffer_size after warmup|
| `replay/priority_entropy`          | > 1.0 (else replay disables itself)|

### 3d.1 · Audit whether hindsight is firing

If you are using `--hindsight`, verify the head is *actually contributing
signal* — the legacy v1 head can be silently zero on every step when
`reasoning_mode` does not teach the `<hindsight>` tag. After the run:

```bash
./venv/bin/python bin/audit_hindsight.py \
    --trainer-state ./honest-qwen-1-5b-grpo/trainer_state.json
```

The script reports:

* The fraction of steps where the hindsight reward channel was exactly 0
  (= the model never emitted a parseable hindsight tag in that step's group).
* The non-zero magnitude (compared against Brier as a sanity ratio).
* A verdict: "structurally silent" / "intermittent" / "healthy".

If the verdict is "structurally silent", relaunch with
`--hindsight-mode refined` (which auto-promotes `--reasoning-mode refined`
so the prompt teaches the new `<critique>` and `<refined_confidence>` tags).
See `docs/SELF_LEARNING.md` §2.5 for the design rationale.

```bash
./venv/bin/python training/train_grpo.py \
    --model-preset qwen1.5b --colab-profile a100 \
    --hindsight --hindsight-mode refined \
    --max-steps 250 \
    --output-dir ./honest-qwen-1-5b-grpo-refined
```

### 3e · Render committed training plots

After training, regenerate the committed evidence PNGs from
`trainer_state.json`:

```bash
make plots TRAINER_STATE=./honest-qwen-3b-grpo/trainer_state.json
# or directly:
./venv/bin/python bin/plot_training_curves.py \
    --trainer-state ./honest-qwen-3b-grpo/trainer_state.json \
    --out docs/training \
    --label "qwen3b · 350 steps · L4"
```

This overwrites `docs/training/loss_curve.png`, `reward_curve.png`,
and `kl_curve.png` — the same images embedded in the project README.
Commit them so the submission carries real training evidence:

```bash
git add docs/training/*.png
git commit -m "docs: refresh training curves from completed run"
```

### 3f · Multi-model recipe

For Llama-3B and Phi-4-mini (~3.8B), use L4 instead of A100:

```bash
# Llama-3.2-3B
./venv/bin/python training/train_grpo.py \
    --model-preset llama3b --colab-profile l4 \
    --max-steps 400 --hindsight --self-mutate \
    --output-dir ./honest-llama-3b-grpo

# Phi-4-mini-instruct
./venv/bin/python training/train_grpo.py \
    --model-preset phi4mini --colab-profile l4 \
    --max-steps 400 --hindsight --self-mutate \
    --output-dir ./honest-phi4mini-grpo
```

---

## 4 · Full evaluation (in-distribution + OOD)

Same metrics battery as Step 2, run on the **trained adapter**, with a
tier-aware OOD pass that auto-discovers JSONL slices written by
`fetch_ood_data.py`.

```bash
# Headline run (medium tier, all 5 OOD slices)
./venv/bin/python eval/full_eval.py \
    --model-id          Qwen/Qwen2.5-3B-Instruct \
    --adapter-path      ./honest-qwen-3b-grpo/final_adapters \
    --baseline-results  eval/baseline_results.json \
    --ood-dir           eval/ood \
    --ood-slices        auto \
    --samples           100 \
    --output            eval/full_results.json

# Tiny-model run (Qwen-0.5B / Llama-1B): tier-auto picks the 3
# small-model-friendly slices (commonsense, science_easy, science_hard)
# and skips medical+legal.
./venv/bin/python eval/full_eval.py \
    --model-id        Qwen/Qwen2.5-0.5B-Instruct \
    --adapter-path    ./grpo-qwen-0.5b/final_adapters \
    --ood-slices      auto \
    --output          eval/full_results_qwen05b.json
```

> **Important:** to make the calibration-transfer claim provable in
> Step 5, also run `full_eval.py` *without* `--adapter-path` against
> the same `--ood-slices` and save it as `eval/baseline_full.json`.
> The compare_runs.py transfer report needs OOD samples on **both**
> sides to show before/after deltas. Without the no-adapter run,
> the transfer table will only have post-RL OOD numbers.

Output (`eval/full_results.json`):

```jsonc
{
  "model_id":     "Qwen/Qwen2.5-3B-Instruct",
  "adapter_path": "./honest-qwen-3b-grpo/final_adapters",
  "preset":       "qwen3b",
  "ood_slices":   ["commonsense", "science_easy", "science_hard", "medical", "legal"],
  "in_distribution": { "math_1": {...}, ..., "logic_5": {...} },
  "ood": {
    "commonsense":  { "ece": ..., "brier": ..., "random_floor": 0.20, "samples": [...] },
    "science_easy": { "ece": ..., "brier": ..., "random_floor": 0.25, "samples": [...] },
    "science_hard": { "ece": ..., "brier": ..., "random_floor": 0.25, "samples": [...] },
    "medical":      { "ece": ..., "brier": ..., "random_floor": 0.25, "samples": [...] },
    "legal":        { "ece": ..., "brier": ..., "random_floor": 0.20, "samples": [...] }
  },
  "overall": { "ece": ..., "brier": ..., "auroc": ..., "accuracy": ..., ... }
}
```

`--ood-slices` accepts:
- `auto` (default): tier-aware default for `--model-id` / `--model-preset`.
- `all`: full registry (5 slices).
- a comma-separated subset, e.g. `commonsense,science_easy`.

To skip ID or OOD individually: `--skip-indist` / `--skip-ood`.
To debug locally without a GPU: `--dry-run`.

---

## 5 · Comparison & success metrics

```bash
# Recommended: pass two full_eval JSONs (one before-RL, one after-RL).
# Both need OOD samples for the calibration-transfer table.
./venv/bin/python eval/compare_runs.py \
    --baseline    eval/baseline_full.json \
    --after       eval/full_results.json \
    --output      eval/comparison.md \
    --plot --plot-output eval/plots/comparison.png

# Backward-compat: a baseline_eval.py JSON (in-dist only) also works,
# but the transfer table will be empty for the baseline column.
./venv/bin/python eval/compare_runs.py \
    --baseline    eval/baseline_results.json \
    --after       eval/full_results.json \
    --output      eval/comparison.md
```

`comparison.md` is the deliverable artefact — paste it directly into a
report or pitch deck. It now contains six sections:

1. **Headline table** (ECE, Brier, AUROC) before vs after, with Δ and a
   95% bootstrap CI on Δ Brier.
2. **Per-domain breakdown** (math / code / logic).
3. **In-distribution vs OOD (after training)** — generalization gap row.
4. **Calibration Transfer (HEADLINE CLAIM)** — per-slice ΔECE table
   with 95% paired-bootstrap CIs, status flags
   (`✓ transferred` / `~ partial` / `⚠ at floor` / `✗ no transfer`),
   plus a single transfer-ratio number summarising how much of the
   in-distribution calibration gain carried to OOD.
5. **Confidence histogram** (text bars, before vs after).
6. **Operating-mode shifts** (format/abstain/malformed rate deltas).

Plus the reliability diagram PNG when you pass `--plot`.

### Success criteria

A pillar/run "ships" only if **all** the following are met. These are
the pass/fail gates for a publishable headline.

| Gate                                | Threshold                                     |
| ----------------------------------- | --------------------------------------------- |
| **Δ ECE (in-distribution)**         | ≤ -0.03 (lower is better)                     |
| **Δ Brier (in-distribution)**       | ≤ -0.02, with 95% CI excluding 0              |
| **Calibration transfer ratio**      | ≥ 0.5× on slices clear of floor               |
| **Δ ECE on ≥2 OOD slices**          | < 0 with 95% CI upper bound < 0 (✓ transferred) |
| **AUROC (in-distribution)**         | ≥ 0.65 (no discrimination collapse)           |
| **Format rate**                     | ≥ 0.90 (parsing did not regress)              |
| **Abstain rate at d=5**             | > abstain rate at d=1                         |

For tiny models, the OOD transfer gate applies to the
`commonsense` / `science_easy` / `science_hard` slices — `medical` and
`legal` are at the random-MCQ floor at this scale and the transfer
ratio averages exclude them automatically (status `⚠ at floor`).

If a gate fails on the headline run but passes per-domain (e.g. math
improves but code regresses), report per-domain and investigate the
losing slice — usually a verifier sharpness or a curriculum balance
issue, not a fundamental training failure.

---

## 6 · MCP deployment

The trained adapter is a self-contained artifact: it can run in any
process that can load the base model + adapter. The MCP server is a
thin, **stateless** wrapper that exposes that artifact to MCP clients.

### 6a · Pre-flight (offline, no GPU)

```bash
make mcp-smoke         # offline self-test (no model load)
make mcp-health        # config preflight: are model + adapter + calibration present?
```

`make mcp-health` should print:

```
[ok] model_id                Qwen/Qwen2.5-3B-Instruct
[ok] adapter_path            ./honest-qwen-3b-grpo/final_adapters
[ok] calibration_info        eval/full_results.json
```

### 6b · Generate a Claude Desktop config

```bash
HONEST_MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
HONEST_ADAPTER_PATH=$PWD/honest-qwen-3b-grpo/final_adapters \
HONEST_CALIBRATION_INFO=$PWD/eval/full_results.json \
make mcp-config
```

Paste the printed JSON snippet into Claude Desktop's
`~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows).

For Cursor and LangGraph, see `mcp_server/README.md`.

### 6c · Launch (manual)

```bash
./venv/bin/python -m mcp_server \
    --model-id          Qwen/Qwen2.5-3B-Instruct \
    --adapter-path      ./honest-qwen-3b-grpo/final_adapters \
    --calibration-info  eval/full_results.json
```

The server speaks MCP over stdio (the standard transport). Once
connected, the client can call:

```jsonc
// ask_with_calibrated_confidence
{
  "answer":           "42",
  "confidence":       0.83,
  "calibration_note": "Confidence is empirically calibrated: ECE = 0.04, Brier = 0.18.",
  "abstained":        false,
  "malformed":        false,
  "raw":              "<reasoning>...</reasoning><answer>42</answer><confidence>0.83</confidence>"
}

// get_calibration_info
{
  "available": true,
  "model":     "Qwen/Qwen2.5-3B-Instruct",
  "preset":    "qwen3b",
  "metrics":   { "ece": 0.04, "brier": 0.18, "auroc": 0.71 },
  "ood":       { "medical": {...}, "legal": {...} }
}
```

### 6d · One-shot installer

```bash
bin/install-mcp.sh
# - installs serving deps if missing
# - runs smoke + health
# - prints a paste-ready Claude Desktop config
```

---

## 7 · Self-learning verification

If you ran Step 3b (with self-learning flags), confirm each pillar
contributed and didn't degrade the run.

### 7a · Hindsight (HCR)

In `eval/full_results.json` look for the per-completion hindsight
emission rate. The model should emit `<hindsight>` tags on at least
**~30 %** of episodes by step 200, and the average `|hindsight − correctness|`
should be lower than `|confidence − correctness|`.

```bash
./venv/bin/python -c "
import json
r = json.load(open('eval/full_results.json'))
print('hindsight emission rate:', r['in_distribution']['metrics'].get('hindsight_rate'))
print('mean |c-y|:           ', r['in_distribution']['metrics'].get('mean_calibration_error'))
print('mean |hindsight-y|:   ', r['in_distribution']['metrics'].get('mean_hindsight_error'))
"
```

### 7b · Replay (CPR)

```bash
./venv/bin/python -c "
import json
s = json.load(open('honest-qwen-3b-grpo/replay_state.json'))
print('buffer_size:        ', s['size'])
print('priority entropy:   ', s['priority_entropy'])   # > 1.0 → diverse, healthy
print('top-1 priority share:', s['top1_share'])         # < 0.05 → no over-replay
"
```

### 7c · Self-mutating curriculum (SMC)

```bash
./venv/bin/python -c "
import json
s = json.load(open('honest-qwen-3b-grpo/smc_state.json'))
for d, st in s.items():
    print(f'{d}: max_unlocked = {st[\"max_unlocked_difficulty\"]}, '
          f'episodes_at_max = {st[\"episodes_at_max\"]}')
"
```

A pillar "shipped" if its dedicated metric moved in the right direction
**and** the headline ECE/Brier didn't regress vs the vanilla run.

---

## 8 · Notebook variant

For interactive Colab / Kaggle runs, `training/train_colab.ipynb`
mirrors Step 3 with cell-by-cell narration. The notebook respects all
the CLI flags, just edit the `args = "--model-preset qwen3b ..."` cell
at the top.

---

## 9 · Reproducibility

* Every random sampler is seeded from `--seed` (default 42) →
  difficulty controller, sampler shuffle, replay buffer init,
  generator stub.
* Eval is seeded by `eval/eval_seeds.json`; pin the seeds before any
  comparison run.
* The adapter directory contains both `trainer_state.json` and the
  full set of CLI args under `training_args.json`.
* Re-running the full pipeline from a fresh checkout against the same
  seeds reproduces the headline numbers within 95 % bootstrap CI.

---

## 10 · Troubleshooting

| Symptom                                              | First thing to check                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------- |
| `Unified sampler is empty` at training startup       | Re-run **§1** ingestion scripts.                                     |
| `format_rate` < 0.70 in baseline                     | Run optional Stage-2 format SFT (`training/format_sft.py`) first.    |
| `train/reward_std` flat near 0                       | `RewardHealthCallback` already disabled the bad batch — keep going.  |
| `train/kl` blowing up (> 0.2)                        | `AdaptiveBetaCallback` will raise β. If still bad, lower `--learning-rate`. |
| OOM at step 0                                        | Drop `--num-generations` by 2 or `--max-completion-length` by 128.   |
| MCP client says "tool not found"                     | `make mcp-health` and confirm the adapter path is absolute.          |
| Δ ECE positive after training                        | Curriculum imbalance — pin `--domain-weights 0.5,0.35,0.15` and retry.|
| Replay buffer sampling the same prompt repeatedly    | `priority_entropy < log(2)` → CPR auto-disables for 50 steps.        |
| SMC ceiling oscillating                              | Increase `--smc-min-episodes-at-max` from 20 to 40.                  |

For server-specific MCP issues see the troubleshooting section in
`mcp_server/README.md`.
