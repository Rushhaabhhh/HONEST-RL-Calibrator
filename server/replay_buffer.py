"""Calibration-Prioritized Replay (CPR) — Pillar 2 of self-learning calibration.

A bounded ring buffer of past prompts, sampled with probability proportional
to *calibration error*:

    p_i = (|c_i - y_i| + eps)^alpha / sum_j (|c_j - y_j| + eps)^alpha

This is PER (Schaul 2015) with the priority replaced from TD-error to
calibration error. The intuition: training compute is most valuable on
prompts where the agent's stated confidence diverges from its actual
correctness — exactly the prompts that need to be revisited to improve
calibration.

We deliberately keep this dependency-free (no sumtree, no torch). For
buffer sizes ≤ 10K, a linear-time sample over numpy weights is sub-ms and
removes a non-trivial install path.

See ``docs/SELF_LEARNING.md`` §3 for the design rationale.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Buffer entry
# ---------------------------------------------------------------------------


@dataclass
class ReplayEntry:
    """One past rollout, suitable for re-sampling.

    Fields are kept JSON-serialisable so the buffer can be checkpointed.
    """

    prompt: str
    ground_truth: str
    domain: str
    difficulty: int
    problem_id: str
    confidence: float
    correctness: bool
    miscalibration: float  # |confidence - correctness|, cached for sampling

    @classmethod
    def make(
        cls,
        prompt: str,
        ground_truth: str,
        domain: str,
        difficulty: int,
        problem_id: str,
        confidence: float,
        correctness: bool,
    ) -> "ReplayEntry":
        c = max(0.0, min(1.0, float(confidence)))
        y = 1.0 if correctness else 0.0
        return cls(
            prompt=prompt,
            ground_truth=str(ground_truth),
            domain=str(domain),
            difficulty=int(difficulty),
            problem_id=str(problem_id),
            confidence=c,
            correctness=bool(correctness),
            miscalibration=abs(c - y),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class CalibrationPrioritizedReplay:
    """Bounded ring buffer with calibration-error-prioritized sampling.

    Parameters
    ----------
    capacity:
        Maximum number of entries. Older entries are evicted FIFO.
    alpha:
        Priority exponent. ``alpha=0`` reduces to uniform replay; ``alpha=1``
        is fully proportional to miscalibration. The literature prefers
        0.5–0.7 to avoid starving low-priority entries.
    eps:
        Floor on the priority so a perfectly-calibrated entry can still be
        sampled (rarely). Without this, well-calibrated prompts vanish from
        the buffer's view forever, skewing future curriculum sampling.
    """

    def __init__(
        self,
        capacity: int = 4096,
        alpha: float = 0.6,
        eps: float = 1e-3,
        seed: Optional[int] = None,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._buf: deque[ReplayEntry] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    # -- mutation --------------------------------------------------------

    def add(
        self,
        prompt: str,
        ground_truth: str,
        domain: str,
        difficulty: int,
        problem_id: str,
        confidence: float,
        correctness: bool,
    ) -> None:
        self._buf.append(
            ReplayEntry.make(
                prompt=prompt,
                ground_truth=ground_truth,
                domain=domain,
                difficulty=difficulty,
                problem_id=problem_id,
                confidence=confidence,
                correctness=correctness,
            )
        )

    def clear(self) -> None:
        self._buf.clear()

    # -- inspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    def is_warm(self, min_size: int) -> bool:
        return len(self._buf) >= int(min_size)

    def mean_miscalibration(self) -> Optional[float]:
        if not self._buf:
            return None
        return sum(e.miscalibration for e in self._buf) / len(self._buf)

    def entropy_of_priorities(self) -> Optional[float]:
        """Shannon entropy (nats) of the sampling distribution.

        Used as a guardrail: if entropy collapses (one or two entries
        dominate), the replay mix should be temporarily disabled to avoid
        overfitting to those prompts. See ``docs/SELF_LEARNING.md`` §9.
        """
        if not self._buf:
            return None
        probs = self._priorities(normalised=True)
        h = 0.0
        for p in probs:
            if p > 0.0:
                h -= p * math.log(p)
        return h

    def snapshot(self) -> Dict[str, Any]:
        ent = self.entropy_of_priorities()
        return {
            "size":         len(self._buf),
            "capacity":     self.capacity,
            "alpha":        self.alpha,
            "eps":          self.eps,
            "mean_miscal":  self.mean_miscalibration(),
            "entropy":      ent,
            "max_entropy":  math.log(len(self._buf)) if self._buf else None,
        }

    # -- sampling --------------------------------------------------------

    def _priorities(self, normalised: bool = False) -> List[float]:
        # priority_i = (miscal_i + eps)^alpha
        prios = [(e.miscalibration + self.eps) ** self.alpha for e in self._buf]
        if not normalised:
            return prios
        s = sum(prios) or 1.0
        return [p / s for p in prios]

    def sample(
        self,
        n: int,
        rng: Optional[random.Random] = None,
    ) -> List[ReplayEntry]:
        """Sample ``n`` entries with calibration-error priority.

        Sampling is **with replacement** so the same high-priority entry
        can be revisited multiple times in one batch. This is what PER
        does in practice — duplicates within a sample are valuable when
        a single prompt is the dominant source of miscalibration.
        """
        if not self._buf:
            return []
        if n <= 0:
            return []

        chooser = rng if rng is not None else self._rng
        weights = self._priorities(normalised=False)
        # random.choices does WR sampling in O(n log m) — fine for ≤10K.
        return chooser.choices(list(self._buf), weights=weights, k=n)
