"""Action parsing, reward computation, and multi-reward functions for GRPO."""

import math
import re
from typing import List, Optional, Tuple

from server.verifier import verify_answer

_ANSWER_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*<analysis>(.*?)</analysis>\s*<confidence>(.*?)</confidence>",
    re.DOTALL | re.IGNORECASE,
)
_ABSTAIN_RE = re.compile(r"<abstain\s*/>", re.IGNORECASE)

MALFORMED_PENALTY = -0.20
ABSTAIN_PENALTY   = -0.10
FORMAT_BONUS      = 0.05


def parse_action(raw_text: str) -> dict:
    text = raw_text.strip()

    if _ABSTAIN_RE.search(text):
        return {"type": "abstain"}

    m = _ANSWER_RE.search(text)
    if m:
        # Extract all blocks but do not strictly grade their content
        reasoning_str = m.group(1).strip()
        answer_str    = m.group(2).strip()
        analysis_str  = m.group(3).strip()
        conf_str      = m.group(4).strip()

        # 🔥 FIX: Reject if the model skips thinking, answering, or analyzing
        if not answer_str or not reasoning_str or not analysis_str:
            return {"type": "malformed"}

        try:
            confidence = float(conf_str)
        except ValueError:
            return {"type": "malformed"}

        if not math.isfinite(confidence):
            return {"type": "malformed"}

        confidence = max(0.0, min(1.0, confidence))

        return {
            "type":       "answer",
            "reasoning":  reasoning_str,
            "answer":     answer_str,
            "analysis":   analysis_str,  # Captured for potential future logging/debugging
            "confidence": confidence,
        }

    return {"type": "malformed"}


def _verify(
    model_answer: str,
    ground_truth: str,
    problem_id: Optional[str],
    domain: Optional[str],
) -> bool:
    """Route to the domain-aware verifier.

    * Procedural problems (``problem_id`` prefix ``procedural_``) use plain
      normalised string-match against ``ground_truth`` — they are generated
      on the fly and not stored in the sampler's ``_by_id`` table, and their
      canonical answer is a simple string (not a JSON grid), so we must
      bypass the domain-specific JSON-grid logic verifier.
    * Curated dataset problems are dispatched through
      ``UnifiedSampler.verify(problem_id, model_answer)`` so each domain
      uses its proper verifier (SymPy / subprocess / JSON-grid).
    * If no ``problem_id`` is provided, fall back to the legacy
      ``verify_answer(...)`` signature using ``domain`` + ground_truth.
    """
    if problem_id and problem_id.startswith("procedural_"):
        # Force domain=None so verify_answer uses the plain string-normalise
        # fallback rather than the JSON-grid logic verifier.
        return verify_answer(model_answer, ground_truth, domain=None)

    if problem_id:
        try:
            from data.sampler.unified_sampler import get_sampler
            return get_sampler().verify(problem_id, model_answer)
        except Exception:
            # Fall through to the legacy path on any sampler/verifier failure.
            pass
    return verify_answer(model_answer, ground_truth, domain)


def compute_reward(
    parsed: dict,
    ground_truth: str,
    difficulty: int,
    problem_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> Tuple[float, Optional[bool]]:
    """
    Compute (reward, correctness_or_None) from a parsed action.

    Reward scheme:
    - malformed                 : (-0.20, None)
    - abstain, difficulty >= 7  : ( 0.00, None)
    - abstain, difficulty <  7  : (-0.10, None)
    - answer                    : (dampened_brier_score + format_bonus, correct)

    Dampened Brier score component:
        brier = -0.5 * ((confidence - target) ** 2)
    where target = 1.0 if correct else 0.0.
    """
    action_type = parsed.get("type")

    if action_type == "malformed":
        return (MALFORMED_PENALTY, None)

    if action_type == "abstain":
        if difficulty >= 7:
            return (0.0, None)
        return (ABSTAIN_PENALTY, None)

    if action_type == "answer":
        try:
            correct = _verify(parsed["answer"], ground_truth, problem_id, domain)
        except Exception:
            # Defensive: treat any verifier exception as an incorrect answer
            correct = False

        target = 1.0 if correct else 0.0

        # Scale Brier penalty by 0.5 to keep gradients stable
        brier = -0.5 * ((parsed["confidence"] - target) ** 2)
        reward = brier + FORMAT_BONUS

        return (reward, correct)

    return (MALFORMED_PENALTY, None)


"""
Multi-reward functions for TRL GRPOTrainer.

Smoothed Magnitude budget (Total bounds ~ [-0.52, +0.20]):
  reward_brier      [-0.45, +0.05]  Primary calibration signal (dampened)
  reward_format     [ 0.00, +0.05]  Early-training compliance bonus
  reward_accuracy   [ 0.00, +0.10]  Correctness encouragement
  reward_anti_hedge [-0.07,  0.00]  Prevents always-0.5 collapse
"""

def reward_brier(
    completions: List[str],
    prompts: List[str],
    ground_truth: List[str],
    difficulty: List[int],
    **kwargs,
) -> List[float]:
    rewards = []
    pid_list = kwargs.get("problem_id", [None] * len(completions))
    domain_list = kwargs.get("domain", [None] * len(completions))
    for comp, gt, diff, pid, dom in zip(
        completions, ground_truth, difficulty, pid_list, domain_list
    ):
        parsed = parse_action(comp)
        r, _ = compute_reward(parsed, str(gt), int(diff), problem_id=pid, domain=dom)
        rewards.append(float(r))
    return rewards


def reward_format(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Format compliance reward: +0.05 for well-formed output, 0.0 otherwise."""
    rewards = []
    for comp in completions:
        parsed = parse_action(comp)
        if parsed["type"] in ("answer", "abstain"):
            rewards.append(0.05)
        else:
            rewards.append(0.0)
    return rewards


def reward_accuracy(
    completions: List[str],
    prompts: List[str],
    ground_truth: List[str],
    **kwargs,
) -> List[float]:
    """Correctness bonus: +0.10 if the answer is correct, 0.0 otherwise."""
    rewards = []
    pid_list = kwargs.get("problem_id", [None] * len(completions))
    domain_list = kwargs.get("domain", [None] * len(completions))
    for comp, gt, pid, dom in zip(completions, ground_truth, pid_list, domain_list):
        parsed = parse_action(comp)
        if parsed["type"] == "answer":
            try:
                correct = _verify(parsed["answer"], str(gt), pid, dom)
            except Exception:
                correct = False
            rewards.append(0.10 if correct else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_anti_hedge(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Anti-hedging penalty: -0.07 when confidence is in [0.45, 0.55]."""
    rewards = []
    for comp in completions:
        parsed = parse_action(comp)
        if parsed["type"] == "answer" and 0.45 <= parsed["confidence"] <= 0.55:
            rewards.append(-0.07)
        else:
            rewards.append(0.0)
    return rewards