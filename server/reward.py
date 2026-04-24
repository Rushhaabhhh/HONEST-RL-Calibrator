"""Action parsing, reward computation, and multi-reward functions for GRPO."""

import math
import re
from typing import List, Optional, Tuple

from server.verifier import verify_answer


_ANSWER_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*<confidence>(.*?)</confidence>",
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
        # Extract reasoning but do not strictly grade its content
        reasoning_str = m.group(1).strip()
        answer_str    = m.group(2).strip()
        conf_str      = m.group(3).strip()

        if not answer_str or not reasoning_str:
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
            "confidence": confidence,
        }

    return {"type": "malformed"}


def compute_reward(
    parsed: dict,
    ground_truth: str,
    difficulty: int,
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
            correct = verify_answer(parsed["answer"], ground_truth)
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
    for comp, gt, diff in zip(completions, ground_truth, difficulty):
        parsed = parse_action(comp)
        r, _ = compute_reward(parsed, str(gt), int(diff))
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
    for comp, gt in zip(completions, ground_truth):
        parsed = parse_action(comp)
        if parsed["type"] == "answer":
            try:
                correct = verify_answer(parsed["answer"], str(gt))
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