"""Hindsight Calibration Reward (HCR) — Pillar 1 of self-learning calibration.

After the agent answers and the environment reveals the ground truth, the
agent may emit a *retrospective* confidence:

    <hindsight>0.10</hindsight>

The hindsight reward is a strictly proper scoring rule on the actual
correctness `y`:

    R_h = -k * (r - y)^2,   k = 0.3

This module is **stateless** — it provides:
  * `parse_hindsight(text)` — recover `r ∈ [0,1]` from a completion
  * `compute_hindsight_reward(r, y, weight)` — scalar reward
  * `reward_hindsight(...)` — TRL-compatible reward function

The two-step protocol (answer → reveal → hindsight) is wired in
``server.environment.HonestEnvironment`` and gated by a probability flag
so HCR adds zero overhead when disabled. See ``docs/SELF_LEARNING.md`` §2 for
the design rationale.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

# HCR is an auxiliary reward; default weight 0.3 keeps it from dominating
# relative to the primary Brier signal so it can never *dominate* training
# in pathological cases (e.g. solver collapses to "always retrospect 0.5").
DEFAULT_HINDSIGHT_WEIGHT: float = 0.3

# A retrospective confidence outside [0, 1] is invalid and yields the
# malformed-hindsight penalty. Same magnitude as the regular malformed
# penalty so the reward is symmetric in failure.
MALFORMED_HINDSIGHT_PENALTY: float = -0.5


_HINDSIGHT_RE = re.compile(
    r"<hindsight>\s*([0-9]*\.?[0-9]+)\s*</hindsight>",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_hindsight(raw_text: Optional[str]) -> dict:
    """Recover a retrospective confidence from a model completion.

    Returns one of:
        {"type": "hindsight", "retrospective": float}   ← well-formed, in [0,1]
        {"type": "malformed"}                            ← tag missing or value bad

    The parser is intentionally strict — same hardness contract as
    ``server.reward.parse_action`` — to prevent format drift.
    """
    if not raw_text:
        return {"type": "malformed"}

    m = _HINDSIGHT_RE.search(raw_text)
    if not m:
        return {"type": "malformed"}

    try:
        r = float(m.group(1))
    except ValueError:
        return {"type": "malformed"}

    if r < 0.0 or r > 1.0:
        # Out of range — don't silently clamp; that would let the model
        # dump a junk number and still get partial reward.
        return {"type": "malformed"}

    return {"type": "hindsight", "retrospective": r}


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def compute_hindsight_reward(
    retrospective: float,
    correctness: bool,
    weight: float = DEFAULT_HINDSIGHT_WEIGHT,
) -> float:
    """Compute the HCR scalar.

    R_h = -weight * (retrospective - y)^2

    Both ``retrospective`` and ``correctness`` come from the immediately-prior
    AnswerAction in the same episode. The caller (environment) is responsible
    for never invoking this on abstain / malformed steps.
    """
    y = 1.0 if correctness else 0.0
    r = max(0.0, min(1.0, float(retrospective)))
    return -float(weight) * (r - y) ** 2


# ---------------------------------------------------------------------------
# TRL-compatible reward function
# ---------------------------------------------------------------------------


def reward_hindsight(
    completions: List[str],
    prompts: List[str] = None,
    ground_truth: List[str] = None,
    **kwargs: Any,
) -> List[float]:
    """TRL reward function for the hindsight head.

    The trainer calls this once per batch of completions. We expect each
    completion to *optionally* be a HindsightAction. Non-hindsight
    completions return 0.0 — they are not penalised, because the hindsight
    slot is opt-in and the primary reward signal already grades them.

    Required kwargs (passed through by ``GRPOTrainer.compute_rewards``):
      - ``previous_correctness``: List[Optional[bool]] — y from the prior
        AnswerAction in this episode. ``None`` means no prior answer
        (e.g. abstain) and the hindsight slot is invalid.
      - ``hindsight_weight``: optional float (default 0.3)
    """
    n = len(completions)
    prev = kwargs.get("previous_correctness") or [None] * n
    weight = float(kwargs.get("hindsight_weight", DEFAULT_HINDSIGHT_WEIGHT))

    rewards: List[float] = []
    for idx, comp in enumerate(completions):
        y = prev[idx] if idx < len(prev) else None

        parsed = parse_hindsight(comp)
        if parsed["type"] != "hindsight":
            # Not a hindsight slot — silent zero. The slot is opt-in.
            rewards.append(0.0)
            continue

        if y is None:
            # Hindsight is only valid after a graded AnswerAction.
            # Emitting one out-of-context is a soft format violation.
            rewards.append(MALFORMED_HINDSIGHT_PENALTY * weight)
            continue

        rewards.append(
            compute_hindsight_reward(parsed["retrospective"], bool(y), weight=weight)
        )
    return rewards


# ---------------------------------------------------------------------------
# Episode-level coordinator
# ---------------------------------------------------------------------------


class HindsightCoordinator:
    """Decides when to inject a hindsight slot and feeds the reveal forward.

    The environment owns one of these. After every AnswerAction it calls
    ``maybe_request(domain, correctness, rng)``; if the coordinator returns
    ``True``, the *next* observation injects a hindsight prompt prefix.

    State is intentionally tiny (a single bool + the last (y, c)) so the
    environment stays Markov-ish: at most one hindsight slot can be in
    flight per episode.
    """

    def __init__(self, probability: float = 0.0):
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"hindsight probability must be in [0, 1], got {probability}")
        self.probability = float(probability)
        self._pending: bool = False
        self._last_correctness: Optional[bool] = None
        self._last_confidence: Optional[float] = None

    def is_active(self) -> bool:
        return self.probability > 0.0

    def maybe_request(
        self,
        correctness: Optional[bool],
        confidence: Optional[float],
        rng,
    ) -> bool:
        """Decide whether the next step should be a hindsight slot."""
        if correctness is None or self.probability <= 0.0:
            self._pending = False
            return False
        if rng.random() < self.probability:
            self._pending = True
            self._last_correctness = bool(correctness)
            self._last_confidence = float(confidence) if confidence is not None else None
            return True
        self._pending = False
        return False

    def consume(self) -> Tuple[bool, Optional[bool], Optional[float]]:
        """Pop the pending hindsight context. Returns (active, y, c_prev)."""
        active = self._pending
        y = self._last_correctness
        c = self._last_confidence
        self._pending = False
        self._last_correctness = None
        self._last_confidence = None
        return active, y, c

    def pending(self) -> bool:
        return self._pending
