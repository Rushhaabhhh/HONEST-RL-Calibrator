"""Action parsing and reward computation."""

import re
from typing import Optional

from server.verifier import verify_answer

# ---------------------------------------------------------------------------
# Format patterns
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    r"<answer>(.*?)</answer>\s*<confidence>(.*?)</confidence>",
    re.DOTALL | re.IGNORECASE,
)
_ABSTAIN_RE = re.compile(r"<abstain\s*/>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------


def parse_action(raw_text: str) -> dict:
    """Parse raw LLM output into a structured action dict.

    Possible returns:
    - {"type": "answer", "answer": str, "confidence": float}
    - {"type": "abstain"}
    - {"type": "malformed"}
    """
    text = raw_text.strip()

    # 1. Check for abstain first (no ambiguity)
    if _ABSTAIN_RE.search(text):
        return {"type": "abstain"}

    # 2. Try to extract answer + confidence
    m = _ANSWER_RE.search(text)
    if m:
        answer_str = m.group(1).strip()
        conf_str = m.group(2).strip()
        try:
            confidence = float(conf_str)
        except ValueError:
            return {"type": "malformed"}

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return {
            "type": "answer",
            "answer": answer_str,
            "confidence": confidence,
        }

    # 3. Nothing matched → malformed
    return {"type": "malformed"}


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

FORMAT_BONUS = 0.02


def compute_reward(
    parsed: dict,
    ground_truth: str,
    difficulty: int,
) -> tuple[float, Optional[bool]]:
    """Compute (reward, correctness_or_None) from a parsed action.

    Reward scheme
    -------------
    malformed                     : (-0.5,  None)
    abstain, difficulty >= 7      : ( 0.0,  None)
    abstain, difficulty <  7      : (-0.3,  None)
    answer                        : (brier_score + format_bonus, correct)

    Brier score component:
        brier = -((confidence - target) ** 2)
    where target = 1.0 if correct else 0.0.

    Format bonus = 0.02 for any well-formed answer tag.
    """
    action_type = parsed.get("type")

    if action_type == "malformed":
        return (-0.5, None)

    if action_type == "abstain":
        if difficulty >= 7:
            return (0.0, None)
        return (-0.3, None)

    if action_type == "answer":
        correct = verify_answer(parsed["answer"], ground_truth)
        target = 1.0 if correct else 0.0
        brier = -((parsed["confidence"] - target) ** 2)
        reward = brier + FORMAT_BONUS
        return (reward, correct)

    # Fallback — treat unknown types as malformed
    return (-0.5, None)
