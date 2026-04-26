"""Shared calibration profiles for training and evaluation.

These presets make cross-model comparisons fair by standardizing:
1) prompt style (reasoning mode),
2) data mixture (domain + difficulty weights),
3) model-aware defaults for GRPO knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


SUPPORTED_PRESETS = ("qwen0.5b", "qwen1.5b", "qwen3b", "llama1b", "llama3b", "phi4mini")
# "required": baseline 3-tag protocol (reasoning + answer + confidence).
# "refined":  Calibration-Aware Self-Refinement protocol — adds a critique
#             slot and a refined_confidence slot the model uses to revise its
#             first-pass confidence after self-critiquing. Pairs with
#             ``--hindsight-mode refined`` in the trainer; see
#             docs/SELF_LEARNING.md §2.5 for the design rationale.
REASONING_MODES = ("required", "refined")


@dataclass(frozen=True)
class CalibrationPreset:
    """Per-model calibration preset.

    Splits cleanly into three blocks:
      1) Data composition  (domain + difficulty mixture, dataset size).
      2) GRPO defaults     (model-aware sampling / optimization knobs).
      3) Reward & KL plan  (auxiliary weights, KL-beta schedule, controller
         seed).

    All values are research-justified for short calibration RL runs
    (≤ 400 GRPO steps, single LoRA stage). They are NOT general SFT
    defaults — they are tuned for stable Brier-score gradients on
    Qwen-3B / Llama-3B / Phi-4-mini under the committed reward scheme
    in ``server/reward.py`` (Brier scale -1.5, FORMAT_BONUS 0.15,
    accuracy bonus +0.85 / -0.15).

    NOTE on anti-hedge: the project deliberately *removed* the
    anti-hedge auxiliary in commit ``3690671`` to plug a 0.7-confidence
    exploit (the model could sit just outside the [0.4, 0.6] band and
    avoid the penalty while still hedging). We intentionally do NOT
    add it back here — calibration is shaped purely by the Brier
    gradient + accuracy bonus, with KL keeping the policy honest.
    """

    name: str
    model_hint: str

    # --- Data composition ---------------------------------------------------
    domain_weights: Dict[str, float]
    difficulty_weights: Dict[int, float]
    default_prompt_dataset_size: int

    # --- GRPO defaults ------------------------------------------------------
    default_num_generations: int
    default_max_completion_length: int
    default_temperature: float
    default_learning_rate: float
    default_beta: float
    default_lora_r: int
    default_max_steps: int

    # --- Reward composition -------------------------------------------------
    # Primary signal is `make_brier_with_curriculum_feedback` (weight 1.0,
    # hard-coded — combines Brier reward + curriculum feedback in one tap).
    # Auxiliaries below are independent reward functions weighted per-preset.
    reward_format_weight: float    # multiplier on +0.15 format bonus
    reward_accuracy_weight: float  # multiplier on the +0.85/-0.15 correctness reward

    # --- KL schedule (start tight, relax after stabilization) --------------
    # beta_start kicks in at step 0 (prevents early policy explosion);
    # beta_end takes over after kl_relax_frac × max_steps (allows
    # calibration consolidation in the second half of training).
    beta_end: float
    kl_relax_frac: float

    # --- Adaptive difficulty controller -----------------------------------
    # Initial per-domain target_difficulty for DifficultyController. Stronger
    # base models benefit from starting at 2 (the bulk of curriculum signal
    # lives at diff 2-3); weaker models need to start at 1 to avoid an
    # all-zero rolling accuracy that would force the controller to plateau
    # at MIN_DIFFICULTY without ever exploring the easy band's calibration.
    default_initial_target: int


MODEL_PRESETS: Dict[str, CalibrationPreset] = {
    # ─────────────────────────────────────────────────────────────────────
    # Qwen2.5-0.5B-Instruct  →  Colab T4 16 GB (free tier) — fastest iteration
    # Tier. ~10-13 s/step on T4 with G=4, max_len=256, ga=4 → 250 steps in
    # ~50 minutes. Smaller capacity means absolute reward/miscal numbers
    # are softer than 1.5B (final reward ~ -0.85 vs -0.70, miscal ~ 0.75
    # vs 0.60), but the trajectory shape is identical: reward climbs,
    # miscal drops, accuracy rises. Ideal for reward-shape sweeps and
    # ablations (hindsight on/off, replay on/off) where you want 3-4
    # runs in the time budget of a single 1.5B run. lr lifted to 4e-6
    # because the smaller policy tolerates steeper updates and noisier
    # rollouts need a bigger gradient. G=4 is small but the GRPO
    # advantage is still well-conditioned (group std stays > 0.1
    # within ~5 steps). Difficulty mixture leans heavily toward d=1-2
    # (0.45 + 0.35) since 0.5B reliably solves only the easy band.
    # ─────────────────────────────────────────────────────────────────────
    "qwen0.5b": CalibrationPreset(
        name="qwen0.5b",
        model_hint="Qwen/Qwen2.5-0.5B-Instruct",
        domain_weights={"math": 0.50, "code": 0.30, "logic": 0.20},
        difficulty_weights={1: 0.45, 2: 0.35, 3: 0.15, 4: 0.04, 5: 0.01},
        default_prompt_dataset_size=1500,
        default_num_generations=4,
        default_max_completion_length=256,
        default_temperature=0.85,
        default_learning_rate=4.0e-6,
        default_beta=0.05,
        default_lora_r=16,
        default_max_steps=250,
        reward_format_weight=1.0,
        reward_accuracy_weight=1.0,
        beta_end=0.02,
        kl_relax_frac=0.50,
        default_initial_target=1,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Qwen2.5-1.5B-Instruct  →  T4 16 GB / L4 24 GB / A100 (fits in bf16
    # without quantization). Iteration-tier preset: ~50 min for 250 steps
    # on A100 80 GB, lets the operator run 3-4 reward-shape sweeps in the
    # time budget of a single 3B run. Smaller policy is noisier per
    # rollout, so G is bumped to 12 (vs 10 at 3B) to stabilize advantage
    # normalization. lr lifted to 3e-6 — small Qwen models tolerate
    # steeper updates and the noisier reward needs a bigger gradient to
    # cut through. beta starts a touch tighter (0.05 vs 0.04) because
    # 1.5B's policy drifts faster early; relaxes on the same 50 % cadence.
    # Difficulty mixture leans easier (more 1-2, less 4-5) since the
    # base model can't reliably solve 4-5 — running there just adds
    # reward noise, not calibration signal.
    # ─────────────────────────────────────────────────────────────────────
    "qwen1.5b": CalibrationPreset(
        name="qwen1.5b",
        model_hint="Qwen/Qwen2.5-1.5B-Instruct",
        domain_weights={"math": 0.50, "code": 0.30, "logic": 0.20},
        difficulty_weights={1: 0.35, 2: 0.35, 3: 0.20, 4: 0.07, 5: 0.03},
        default_prompt_dataset_size=2500,
        default_num_generations=12,
        default_max_completion_length=384,
        default_temperature=0.85,
        default_learning_rate=3.0e-6,
        default_beta=0.05,
        default_lora_r=32,
        default_max_steps=250,
        reward_format_weight=1.0,
        reward_accuracy_weight=1.0,
        beta_end=0.02,
        kl_relax_frac=0.50,
        default_initial_target=1,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Qwen2.5-3B-Instruct  →  L4 24 GB (recommended) / A10G 24 GB
    # Strong reasoning per parameter (~50-55 % acc on diff-2/3) but rollouts
    # are noisier than at 7B, so we bump G from 8 → 10 to stabilize the
    # GRPO advantage normalization. lr is lifted to 2e-6 (smaller models
    # tolerate steeper updates and Qwen is the most LR-stable of the trio);
    # beta drops to 0.04 because the smaller policy drifts naturally — a
    # tighter beta would waste KL budget without adding stability. Format
    # weight 1.0 because Qwen format-compliance locks in within ~30 steps
    # even at 3B, so we don't crowd out the Brier gradient. Difficulty
    # mixture shifts slightly easier than the 7B preset (more 1-2, less
    # 4-5) because the per-rollout reward signal is noisier on harder
    # problems and 3B can't reliably extract a calibration gradient from
    # the long tail.
    # ─────────────────────────────────────────────────────────────────────
    "qwen3b": CalibrationPreset(
        name="qwen3b",
        model_hint="Qwen/Qwen2.5-3B-Instruct",
        domain_weights={"math": 0.50, "code": 0.35, "logic": 0.15},
        difficulty_weights={1: 0.25, 2: 0.35, 3: 0.25, 4: 0.10, 5: 0.05},
        default_prompt_dataset_size=3500,
        default_num_generations=10,
        default_max_completion_length=512,
        default_temperature=0.85,
        default_learning_rate=2.0e-6,
        default_beta=0.04,
        default_lora_r=32,
        default_max_steps=350,
        reward_format_weight=1.0,
        reward_accuracy_weight=1.0,
        beta_end=0.015,
        kl_relax_frac=0.50,
        default_initial_target=2,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Llama-3.2-1B-Instruct  →  Colab T4 16 GB / L4 24 GB
    # Cross-family iteration tier. Llama-1B reasoning is strong on code
    # (Llama series was distilled on coding heavily) but weaker on math at
    # this scale than Qwen-1.5B. Format compliance lags Qwen's: the
    # model occasionally emits XML before its closing tag, so we lift
    # reward_format_weight to 1.5 to anchor structure during the early
    # exploration phase. Temperature 0.9 (vs Qwen's 0.85) because Llama-1B
    # samples are tighter at lower temps and we need group diversity for
    # the GRPO advantage signal. lr=2e-6 conservative — Llama at this
    # scale format-collapses under aggressive lr (same failure mode as
    # the 3B preset, manifested earlier). Difficulty mixture mirrors
    # Qwen-0.5B (lots of d=1-2) since 1B Llama solves d=4-5 too rarely
    # to provide useful calibration gradient. Initial target=1 keeps
    # the controller out of the noise floor at d=2.
    # ─────────────────────────────────────────────────────────────────────
    "llama1b": CalibrationPreset(
        name="llama1b",
        model_hint="meta-llama/Llama-3.2-1B-Instruct",
        domain_weights={"math": 0.40, "code": 0.40, "logic": 0.20},
        difficulty_weights={1: 0.40, 2: 0.35, 3: 0.18, 4: 0.05, 5: 0.02},
        default_prompt_dataset_size=2000,
        default_num_generations=4,
        default_max_completion_length=256,
        default_temperature=0.90,
        default_learning_rate=2.0e-6,
        default_beta=0.05,
        default_lora_r=16,
        default_max_steps=250,
        reward_format_weight=1.5,
        reward_accuracy_weight=1.0,
        beta_end=0.02,
        kl_relax_frac=0.55,
        default_initial_target=1,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Llama-3.2-3B-Instruct  →  L4 24 GB
    # Weaker reasoning (~45 % on diff-2) and noisier rollouts. We bump G to
    # 10 and use temp=0.9 for exploration, but keep lr=1e-6 conservative
    # because 3B Llama tends to format-collapse under aggressive lr.
    # Format weight 1.5 because Llama is the most prone to format drift
    # (especially after the KL relaxes); a stronger format anchor
    # protects the +0.15 bonus from being overshadowed during fast lr
    # decay. Accuracy weight 0.8 because the Brier signal alone is
    # already aggressive on the smaller model.
    # ─────────────────────────────────────────────────────────────────────
    "llama3b": CalibrationPreset(
        name="llama3b",
        model_hint="meta-llama/Llama-3.2-3B-Instruct",
        domain_weights={"math": 0.45, "code": 0.35, "logic": 0.20},
        difficulty_weights={1: 0.30, 2: 0.35, 3: 0.20, 4: 0.10, 5: 0.05},
        default_prompt_dataset_size=3500,
        default_num_generations=10,
        default_max_completion_length=512,
        default_temperature=0.90,
        default_learning_rate=1.0e-6,
        default_beta=0.04,
        default_lora_r=16,
        default_max_steps=350,
        reward_format_weight=1.5,
        reward_accuracy_weight=0.8,
        beta_end=0.015,
        kl_relax_frac=0.55,
        default_initial_target=1,
    ),
    # ─────────────────────────────────────────────────────────────────────
    # Phi-4-mini-instruct  →  L4 24 GB (sequential after Llama)
    # Best format compliance of the trio; reaches reliable XML by step ~25.
    # Dataset intentionally smaller (2500) since 250 × 8 = 2000 prompts are
    # consumed. Format weight 1.0 (Phi already does it well); accuracy
    # weight 1.0 to mirror Qwen's symmetric incentive shape.
    # ─────────────────────────────────────────────────────────────────────
    "phi4mini": CalibrationPreset(
        name="phi4mini",
        model_hint="microsoft/Phi-4-mini-instruct",
        domain_weights={"math": 0.45, "code": 0.35, "logic": 0.20},
        difficulty_weights={1: 0.25, 2: 0.35, 3: 0.25, 4: 0.10, 5: 0.05},
        default_prompt_dataset_size=2500,
        default_num_generations=8,
        default_max_completion_length=384,
        default_temperature=0.75,
        default_learning_rate=1.5e-6,
        default_beta=0.04,
        default_lora_r=16,
        default_max_steps=250,
        reward_format_weight=1.0,
        reward_accuracy_weight=1.0,
        beta_end=0.015,
        kl_relax_frac=0.50,
        default_initial_target=2,
    ),
}


def infer_preset_name(model_id: str) -> str:
    """Infer preset from model id; defaults to qwen3b for unknown ids."""
    m = (model_id or "").lower()
    if "qwen" in m and ("0.5b" in m or "0_5b" in m):
        return "qwen0.5b"
    if "qwen" in m and ("1.5b" in m or "1_5b" in m):
        return "qwen1.5b"
    if "qwen" in m and "3b" in m:
        return "qwen3b"
    if "llama" in m and "1b" in m:
        return "llama1b"
    if "llama" in m and "3b" in m:
        return "llama3b"
    if "phi-4-mini" in m or ("phi" in m and "mini" in m):
        return "phi4mini"
    # Legacy fallback: many users still run phi-3.5-mini
    if "phi-3.5" in m or "phi3.5" in m:
        return "phi4mini"
    return "qwen3b"


def get_preset(model_id: str, preset_override: str = "auto") -> CalibrationPreset:
    preset_name = infer_preset_name(model_id) if preset_override == "auto" else preset_override
    if preset_name not in MODEL_PRESETS:
        valid = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown preset '{preset_name}'. Valid presets: {valid}")
    return MODEL_PRESETS[preset_name]


def _normalize_weights(weight_map: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(v, 0.0) for v in weight_map.values()))
    if total <= 0.0:
        n = len(weight_map)
        return {k: 1.0 / n for k in weight_map}
    return {k: max(v, 0.0) / total for k, v in weight_map.items()}


def parse_weight_csv(
    csv_text: Optional[str],
    keys: List[str],
) -> Optional[Dict[str, float]]:
    """Parse comma-separated weight list aligned to ``keys``."""
    if not csv_text:
        return None
    parts = [p.strip() for p in csv_text.split(",") if p.strip()]
    if len(parts) != len(keys):
        raise ValueError(f"Expected {len(keys)} weights for {keys}, got {len(parts)}")
    raw = {k: float(v) for k, v in zip(keys, parts)}
    return _normalize_weights(raw)


def parse_difficulty_csv(csv_text: Optional[str]) -> Optional[Dict[int, float]]:
    """Parse 5 comma-separated difficulty weights for levels 1..5."""
    parsed = parse_weight_csv(csv_text, ["1", "2", "3", "4", "5"])
    if parsed is None:
        return None
    return {int(k): v for k, v in parsed.items()}


_REQUIRED_SYSTEM_PROMPT = """You are a precise and well-calibrated AI assistant.

Respond in EXACTLY this format:
<reasoning>
Briefly solve the problem.
</reasoning>
<answer>YOUR_ANSWER_HERE</answer>
<confidence>0.X</confidence>

Rules:
- Confidence must be between 0.0 and 1.0
- If very unsure, output <abstain/>
- Keep reasoning concise, then provide final answer and confidence."""

_REQUIRED_USER_TEMPLATE = (
    "{question}\n\n"
    "Think briefly in <reasoning>, then provide <answer> and <confidence>."
)


# The "refined" prompt teaches the four-tag Calibration-Aware Self-Refinement
# protocol. The two-stage example (one wrong → reduces confidence, one right
# → bumps confidence) is critical: it shows the model that <refined_confidence>
# can move in *either* direction after a critique. Earlier prompt drafts that
# only showed the "lower the confidence" example caused the model to collapse
# to always lowering confidence, hurting calibration on correct answers.
_REFINED_SYSTEM_PROMPT = """You are a precise, well-calibrated AI assistant that critiques its own work.

Respond in EXACTLY this format (all five tags required):
<reasoning>
Solve the problem step by step.
</reasoning>
<answer>YOUR_ANSWER_HERE</answer>
<confidence>0.X</confidence>
<critique>
Re-read your reasoning above and explicitly look for arithmetic slips, logical
gaps, or missing cases. State concretely what (if anything) is uncertain.
</critique>
<refined_confidence>0.X</refined_confidence>

Rules:
- Both <confidence> and <refined_confidence> must be in [0.0, 1.0].
- <refined_confidence> should be DIFFERENT from <confidence> when your critique
  uncovers something — raise it if you re-verified the answer, lower it if you
  spotted a possible error. Trivially copying the same number is discouraged.
- The critique must be substantive (at least one full sentence of self-review).
- If you are extremely uncertain even after critique, you may output <abstain/>
  *instead of* the answer/confidence/critique block.

Worked example (wrong answer, confidence drops):
<reasoning>
3 + 4 = 8.
</reasoning>
<answer>8</answer>
<confidence>0.85</confidence>
<critique>
Re-checking: 3 + 4 is actually 7, not 8. I made an arithmetic slip.
</critique>
<refined_confidence>0.05</refined_confidence>

Worked example (correct answer, confidence rises):
<reasoning>
A circle has area πr². With r=2, area = 4π ≈ 12.566.
</reasoning>
<answer>12.566</answer>
<confidence>0.6</confidence>
<critique>
The formula πr² is correct; 2² = 4 and π ≈ 3.14159, so 4π ≈ 12.566 is right.
</critique>
<refined_confidence>0.95</refined_confidence>"""

_REFINED_USER_TEMPLATE = (
    "{question}\n\n"
    "Solve in <reasoning>, give <answer> and a first-pass <confidence>. "
    "Then write a substantive <critique> of your own reasoning and emit a "
    "<refined_confidence> that reflects what the critique found."
)


def prompt_templates(reasoning_mode: str) -> tuple[str, str]:
    """Return (system_prompt, user_template) for selected reasoning mode.

    "required": 3-tag baseline protocol (reasoning, answer, confidence).
    "refined":  5-tag Calibration-Aware Self-Refinement protocol. Pairs
                with the ``server.hindsight_v2.make_refinement_reward``
                head and the ``--hindsight-mode refined`` trainer flag.
    """
    mode = (reasoning_mode or "required").lower()
    if mode not in REASONING_MODES:
        valid = ", ".join(REASONING_MODES)
        raise ValueError(f"Invalid reasoning_mode '{reasoning_mode}'. Valid: {valid}")
    if mode == "refined":
        return _REFINED_SYSTEM_PROMPT, _REFINED_USER_TEMPLATE
    return _REQUIRED_SYSTEM_PROMPT, _REQUIRED_USER_TEMPLATE
