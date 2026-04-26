"""Calibration-Aware Self-Refinement (CASR) — Hindsight v2.

Why a v2
========

The legacy ``server.hindsight`` module rewards a retrospective confidence
``r`` against ground truth ``y`` with ``-k(r-y)^2``. In the single-pass
training setup (no online HER rollout), this is **informationally redundant
with the primary Brier reward**: the model emits ``c`` and ``r`` from the
same context, and the optimal policy under both rewards combined is
``c = r = E[y|x]`` — identical to the optimal policy under Brier alone.
Worse, the legacy reward provides no positive gradient toward emitting the
``<hindsight>`` tag in the first place, so a base model with no prior on
that tag will simply never produce one (the reward channel stays at 0.0
for the entire run; verify with ``bin/audit_hindsight.py``).

The CASR head fixes this by rewarding a genuinely different behaviour:
**the model critiques its own answer and refines its confidence**. The
completion now contains four signals instead of two::

    <reasoning>...</reasoning>
    <answer>X</answer>
    <confidence>c</confidence>
    <critique>spot any errors in the reasoning above</critique>
    <refined_confidence>r</refined_confidence>

The reward decomposes into four terms with carefully designed gradients:

    R_h = α · ΔBrier      ← (c-y)^2 - (r-y)^2  (POSITIVE if refining helped)
        + β · 1[critique_ok]   ← format bonus (kickstart gradient toward emitting tags)
        − γ · 1[r ≈ c]         ← anti-trivial-copy penalty (prevent the easy exploit)
        − δ · 1[partial]       ← penalty for half-emitted structure

Final reward is clamped to ±0.30 so it cannot dominate the primary
Brier signal.

Why this contributes signal that Brier alone cannot
---------------------------------------------------

* When the model is already perfectly calibrated, ``ΔBrier ≈ 0`` *and* the
  anti-copy penalty fires, so the reward goes to 0 — no double-counting.
* When the model is mis-calibrated and updates ``r`` *toward* ``y`` after
  critiquing, ``ΔBrier > 0`` and the gradient pulls the underlying
  reasoning trace toward producing better critiques.
* When the group of GRPO rollouts agrees on ``(c, y)`` (a frequent cause
  of zero-σ "wasted steps" in the current run), but some rollouts emit a
  critique and others don't, the format bonus produces non-zero
  group-relative advantage — recovering signal that the primary reward
  loses.

Research grounding
------------------

* Self-Refine (Madaan et al., NeurIPS 2023): iterative self-critique
  improves single-pass LLM outputs.
* Self-Verification (Weng et al., EMNLP 2023): asking the model to verify
  its own answer reduces hallucination and improves calibration.
* Process Reward Models (Cobbe et al., 2021): step-level verification
  correlates strongly with outcome correctness.
* Reflexion (Shinn et al., NeurIPS 2023): verbal self-reflection beats
  next-token prediction alone for sequential tasks.
* HER (Andrychowicz et al., 2017): the original idea that hindsight
  re-labelling injects new information into the gradient — except here
  the "new information" is the model's own critique rather than an
  exogenous goal-relabel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

# Re-export the legacy parser for callers that still want raw <hindsight>
# parsing (e.g. environment-time HER protocol).
from server.hindsight import parse_hindsight  # noqa: F401  (re-export)
from server.reward import parse_action, compute_reward


# ---------------------------------------------------------------------------
# Reward weights — research-justified defaults
# ---------------------------------------------------------------------------

# α scales the Brier-improvement signal. We choose 1.0 so that a refinement
# from c=0.9 to r=0.5 on a wrong answer (Δb = 0.81 - 0.25 = 0.56) maps to a
# reward of +0.56 — comparable in magnitude to the primary Brier reward
# (which uses scale -1.5 in server.reward.compute_reward, i.e. peak |R| ≈ 1.5).
ALPHA_DELTA_BRIER: float = 1.0

# β is the format bonus for emitting a non-trivial critique. Kept small
# (5%) because we don't want the model to spam empty critiques to farm
# format reward — the anti-copy penalty backstops that, but we still want
# the dominant gradient channel to be ΔBrier.
BETA_CRITIQUE_FORMAT: float = 0.05

# γ penalises r ≈ c. Symmetric with β so a model that emits a critique but
# trivially copies confidence ends up with ~0 net reward (β - γ = 0).
GAMMA_TRIVIAL_COPY: float = 0.05

# δ penalises partial structure (e.g. emits <critique> but no
# <refined_confidence>, or vice versa). Same magnitude — partial structure
# is a soft format violation, treated like trivial-copy in cost.
DELTA_PARTIAL_STRUCTURE: float = 0.05

# What counts as a non-trivial critique. Below this character count we
# treat the critique as empty — prevents farming the format bonus with
# "<critique>x</critique>".
MIN_CRITIQUE_CHARS: int = 16

# What counts as r ≈ c (trivial copy). 0.02 is roughly the granularity at
# which Brier scores become indistinguishable for downstream metrics.
COPY_EPSILON: float = 0.02

# Final reward is clipped to this magnitude so CASR cannot dominate the
# primary Brier signal even if its components conspire to extreme values.
REWARD_CLIP: float = 0.30


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_CRITIQUE_RE = re.compile(
    r"<critique>(.*?)</critique>",
    re.DOTALL | re.IGNORECASE,
)
_REFINED_CONF_RE = re.compile(
    r"<refined_confidence>\s*([0-9]*\.?[0-9]+)\s*</refined_confidence>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RefinementParse:
    """Structured parse of the four CASR-relevant slots in a completion."""

    has_answer: bool
    has_confidence: bool
    has_critique: bool
    has_refined_confidence: bool
    confidence: Optional[float]
    refined_confidence: Optional[float]
    critique_text: str

    @property
    def has_full_structure(self) -> bool:
        return (
            self.has_answer
            and self.has_confidence
            and self.has_critique
            and self.has_refined_confidence
        )

    @property
    def has_partial_structure(self) -> bool:
        # Emitted SOME hindsight slot but not all — the soft violation
        # we want to penalise so the model commits to one mode or the
        # other, without zero-rewarding a clean answer-only completion.
        emitted = self.has_critique or self.has_refined_confidence
        return emitted and not self.has_full_structure


def parse_refinement(raw_text: Optional[str]) -> RefinementParse:
    """Parse a completion into the four CASR slots.

    The function is forgiving by design: it never raises, and missing /
    unparseable slots simply set the corresponding ``has_*`` flag to
    ``False``. Reward shaping happens downstream — keeping parsing pure
    means tests stay deterministic and the reward function can be tuned
    independently.
    """
    if not raw_text:
        return RefinementParse(
            has_answer=False, has_confidence=False,
            has_critique=False, has_refined_confidence=False,
            confidence=None, refined_confidence=None, critique_text="",
        )

    parsed_action = parse_action(raw_text)
    has_answer = parsed_action.get("type") == "answer"
    has_confidence = has_answer and parsed_action.get("confidence") is not None
    confidence = parsed_action.get("confidence") if has_confidence else None

    crit_match = _CRITIQUE_RE.search(raw_text)
    critique_text = crit_match.group(1).strip() if crit_match else ""
    has_critique = len(critique_text) >= MIN_CRITIQUE_CHARS

    rc_match = _REFINED_CONF_RE.search(raw_text)
    refined_confidence: Optional[float] = None
    has_refined_confidence = False
    if rc_match:
        try:
            r = float(rc_match.group(1))
        except ValueError:
            r = None
        if r is not None and 0.0 <= r <= 1.0:
            refined_confidence = r
            has_refined_confidence = True

    return RefinementParse(
        has_answer=has_answer,
        has_confidence=has_confidence,
        has_critique=has_critique,
        has_refined_confidence=has_refined_confidence,
        confidence=confidence,
        refined_confidence=refined_confidence,
        critique_text=critique_text,
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def compute_refinement_reward(
    parse: RefinementParse,
    correctness: Optional[bool],
    *,
    alpha: float = ALPHA_DELTA_BRIER,
    beta: float = BETA_CRITIQUE_FORMAT,
    gamma: float = GAMMA_TRIVIAL_COPY,
    delta: float = DELTA_PARTIAL_STRUCTURE,
    clip: float = REWARD_CLIP,
) -> float:
    """Compute the CASR scalar from a parsed completion + correctness label.

    Returns 0.0 when the completion has no critique structure at all
    (clean answer-only output) so the head adds no noise to standard
    rollouts. The format bonus only fires when the model commits to the
    refinement protocol.
    """
    # 1. No refinement structure attempted — silent zero. This is the
    #    common case: answer-only completions get no signal, so CASR
    #    adds zero variance to standard training.
    if not parse.has_critique and not parse.has_refined_confidence:
        return 0.0

    reward = 0.0

    # 2. Partial structure penalty (emitted critique XOR refined_confidence).
    if parse.has_partial_structure:
        reward -= delta

    # 3. Format bonus for emitting a non-trivial critique.
    if parse.has_critique:
        reward += beta

    # 4. ΔBrier improvement signal — the core gradient. Only valid when
    #    we have BOTH confidences AND a graded answer.
    if (
        parse.has_confidence
        and parse.has_refined_confidence
        and correctness is not None
    ):
        y = 1.0 if correctness else 0.0
        c = float(parse.confidence)  # type: ignore[arg-type]
        r = float(parse.refined_confidence)  # type: ignore[arg-type]
        delta_brier = (c - y) ** 2 - (r - y) ** 2
        reward += alpha * delta_brier

        # 5. Anti-trivial-copy penalty: r ≈ c means no actual refinement
        #    happened, so the model is just farming the format bonus.
        if abs(r - c) < COPY_EPSILON:
            reward -= gamma

    # Final clip prevents the auxiliary head from dominating the primary
    # Brier signal in pathological cases.
    if reward > clip:
        return clip
    if reward < -clip:
        return -clip
    return reward


# ---------------------------------------------------------------------------
# TRL-compatible reward function factory
# ---------------------------------------------------------------------------


def make_refinement_reward(
    weight: float = 1.0,
    *,
    alpha: float = ALPHA_DELTA_BRIER,
    beta: float = BETA_CRITIQUE_FORMAT,
    gamma: float = GAMMA_TRIVIAL_COPY,
    delta: float = DELTA_PARTIAL_STRUCTURE,
    clip: float = REWARD_CLIP,
):
    """Build a TRL-compatible reward function for the CASR head.

    The returned callable matches the GRPO reward signature:

        f(completions, prompts, ground_truth, **kwargs) -> List[float]

    where kwargs may contain ``domain``, ``problem_id``,
    ``verification_metadata`` (passed through by the training loop's reward
    wrapper). The function grades ``correctness`` locally via
    ``server.reward.compute_reward`` so it stays consistent with the primary
    Brier head — no risk of the two heads disagreeing on whether an answer
    is correct.

    ``weight`` is a global scale applied AFTER clipping, so the per-rollout
    reward magnitude is bounded by ``weight * clip``. With defaults
    (weight=1.0, clip=0.30) the head contributes at most ±0.30 per rollout,
    well below the primary Brier scale.
    """

    def _wrapped(
        completions: List[str],
        prompts: List[str] = None,
        ground_truth: List[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        n = len(completions)
        gts = ground_truth or [""] * n
        domains = kwargs.get("domain") or [None] * n
        pids = kwargs.get("problem_id") or [None] * n
        v_metas = kwargs.get("verification_metadata") or [{}] * n
        rewards: List[float] = []
        for idx, comp in enumerate(completions):
            parse = parse_refinement(comp)

            # Fast path: no refinement structure at all → silent zero.
            if not parse.has_critique and not parse.has_refined_confidence:
                rewards.append(0.0)
                continue

            # Grade correctness for the ΔBrier term.
            correctness: Optional[bool] = None
            if parse.has_answer and parse.has_confidence:
                action = parse_action(comp)  # already cached as parse_action result
                _r, correct = compute_reward(
                    action,
                    str(gts[idx] if idx < len(gts) else ""),
                    1,  # difficulty isn't relevant for correctness check
                    domain=domains[idx] if idx < len(domains) else None,
                    verification_metadata=v_metas[idx] if idx < len(v_metas) else {},
                )
                correctness = correct

            r = compute_refinement_reward(
                parse, correctness,
                alpha=alpha, beta=beta, gamma=gamma, delta=delta, clip=clip,
            )
            rewards.append(weight * r)
        return rewards

    _wrapped.__name__ = f"reward_refinement_x{weight:g}"
    return _wrapped
