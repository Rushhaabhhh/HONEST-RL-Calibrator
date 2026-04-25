# Data Sampler Migration Guide

> **Who is this for?** Rushabh (environment owner) — how to swap the procedural
> generators in `server/environment.py` for the external-dataset sampler.

---

## What's changing and why

The procedural generators in `server/generators/` produce synthetic arithmetic
and toy-logic problems.  The new sampler draws from **19,711 curated problems**:

| Domain | Source | Difficulty | Count |
|--------|--------|-----------|-------|
| math   | Hendrycks MATH | 1–5 | 12,496 |
| code   | MBPP (diff 1–2) + APPS (diff 3–5) | 1–5 | 5,915 |
| logic  | Z3-generated ZebraLogic | 3–5 | 1,300 |

No other part of the environment changes.

---

## Step 1 — Change three import lines in `server/environment.py`

```python
# ── BEFORE ──────────────────────────────────────────────────────────────────
from server.generators import code_gen, logic_gen, math_gen

# inside __init__:
self._generators = {
    "math":  math_gen.generate,
    "code":  code_gen.generate,
    "logic": logic_gen.generate,
}

# ── AFTER ───────────────────────────────────────────────────────────────────
from data.sampler.math_gen_adapter  import generate as math_generate
from data.sampler.code_gen_adapter  import generate as code_generate
from data.sampler.logic_gen_adapter import generate as logic_generate

# inside __init__:
self._generators = {
    "math":  math_generate,
    "code":  code_generate,
    "logic": logic_generate,
}
```

> **That's the only required change.**  The function signatures are identical to
> the procedural generators: `generate(difficulty: int, seed: Optional[int] = None) -> tuple[str, str]`.

---

## Step 2 — Unified verifier in the reward function (optional but recommended)

If `server/reward.py` currently does a plain string comparison for correctness,
replace it with the domain-aware unified verifier so math gets symbolic
equivalence checking and logic gets cell-accuracy scoring:

```python
# In compute_reward(), replace the verification call with:
from data.sampler.environment_adapter import get_sampler

sampler = get_sampler()          # singleton — no repeated loading
correct = sampler.verify(problem_id, model_answer)
```

> **Note:** `verify()` requires a `problem_id` (the stable ID stored in each
> `UnifiedProblem`).  The sampler needs to be informed which problem was just
> generated.  The simplest approach: store the `problem_id` alongside
> `_current_answer` in the environment state (same pattern already used for
> `_current_metadata`).

---

## Step 3 — Bump `max_completion_length` for logic problems

Logic ZebraLogic problems require a full grid JSON in the answer, which is
significantly longer than a single number or a function.  In
`training/train_grpo.py`, change:

```python
# BEFORE
max_completion_length=512,

# AFTER
max_completion_length=1024,
```

Logic problem questions already embed the JSON output instruction
(`"Respond in JSON format: {\"House 1\": ...}"`) — no additional prompt
engineering is needed.

---

## Step 4 — Logic generator routing (procedural + ZebraLogic)

The logic adapter routes by difficulty:

- **Difficulties 1-2:** the procedural generator in
  `server/generators/logic_gen.py` (transitivity puzzles and small CSPs;
  short single-token string answers like `"Alice"`).  Synthesised
  `problem_id` prefix: `procedural_logic_`.
- **Difficulties 3-5:** the curated ZebraLogic dataset (JSON-grid answers).

Verification dispatches accordingly: procedural problems use plain
normalised string-match against the canonical answer (they are generated
on the fly and never enter the sampler's `_by_id` table); ZebraLogic
problems use the JSON-grid cell-accuracy verifier.

Because difficulties 1-2 are populated again, all domains start at
difficulty 1 (`INITIAL_DIFFICULTIES = {"math": 1, "code": 1, "logic": 1}`).
The adaptive controller then ramps up from there.

---

## What does NOT change

| Component | Status |
|-----------|--------|
| `server/environment.py` reward formula | ✅ Unchanged |
| `server/reward.py` Brier-score computation | ✅ Unchanged |
| `server/difficulty.py` adaptive scheduler | ✅ Unchanged |
| `models/models.py` Pydantic schemas | ✅ Unchanged |
| OpenEnv API surface | ✅ Unchanged |
| Training script (except `max_completion_length`) | ✅ Unchanged |

---

## Sampler data coverage notes

**Empty buckets (graceful fallback):**

| Domain | Missing difficulties |
|--------|---------------------|
| logic  | 1, 2 (no data — ZebraLogic minimum grid is 3×3) |
| code   | no difficulty-5 MBPP; APPS fills 3–5 |

When a difficulty bucket is empty the sampler emits a `warnings.warn` and
falls back to the nearest populated difficulty.  The environment log will
show which difficulty was actually used.

---

## Verification

Run these tests to confirm everything works end-to-end before merging:

```bash
# Activate the project venv
source /Users/kananarora/Desktop/HonestEnv/venv/bin/activate

# From project root
PYTHONPATH=. pytest data/tests/test_unified_sampler.py \
                    data/tests/test_integration.py \
                    data/tests/test_logic_verifier.py \
                    -v
```

Expected: **all tests pass** (currently 84 total across the three files).

---

## File map

```
data/
├── sampler/
│   ├── unified_sampler.py       # Core class: loads data, exposes *_generate() + verify()
│   ├── environment_adapter.py   # Singleton get_sampler() + module-level shim functions
│   ├── math_gen_adapter.py      # Exposes generate() for math  ← swap import here
│   ├── code_gen_adapter.py      # Exposes generate() for code  ← swap import here
│   └── logic_gen_adapter.py     # Exposes generate() for logic ← swap import here
├── verifiers/
│   ├── math_verifier.py         # SymPy-based symbolic equivalence
│   ├── code_verifier.py         # Subprocess test-runner
│   └── logic_verifier.py        # Cell-accuracy (threshold ≥ 0.9)
├── processed/
│   ├── math.jsonl               # 12,496 Hendrycks MATH problems
│   ├── code_mbpp.jsonl          #    427 MBPP problems (diff 1–2)
│   ├── code_apps.jsonl          #  5,488 APPS problems (diff 3–5)
│   └── logic_zebralogic.jsonl   #  1,300 ZebraLogic problems (diff 3–5)
└── MIGRATION.md                 # ← you are here
```
