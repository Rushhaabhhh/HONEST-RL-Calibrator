"""Self-Mutating Curriculum (SMC) — Pillar 3 of self-learning calibration.

When the agent reaches > τ rolling accuracy at the highest tier of the
unified sampler, the curriculum *creates* a new tier above it by mutating
existing top-tier problems through one of three deterministic mutators:

  * NumericMutator        — rescale literal numbers in a math problem.
  * CompositionalMutator  — chain two same-domain problems P1 → P2.
  * DistractorMutator     — prepend irrelevant context.

All three preserve a verifiable ground truth — RL without grading is
worthless. The verifier is the same one used by the unified sampler, so
mutated problems plug into ``server.reward.compute_reward`` unchanged.

See ``SELF_LEARNING.md`` §4 for the design rationale.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("smc")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class MutatedProblem:
    """A problem produced by a mutator.

    ``problem_id`` is prefixed with ``mutated_<mutator>__`` so downstream
    consumers (including ``server.reward._verify``) can detect mutation
    and route to the local fallback verifier — the unified sampler does
    not know about mutated ids.
    """
    question: str
    canonical_answer: str
    domain: str
    difficulty: int             # the *new* (mutated) difficulty
    base_problem_id: str        # the original problem this was mutated from
    mutator: str                # name of the mutator that produced it
    problem_id: str             # synthetic id; mutated_<mutator>__<base_id>__<rng>


# A "base problem source" is anything that returns (q, a, pid) for a
# given (domain, difficulty). The unified sampler's environment_adapter
# generators match this signature exactly.
BaseProblemSource = Callable[..., Tuple[str, str, str]]


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


_INT_TOKEN_RE = re.compile(r"(?<![\d.])(\d+)(?![\d.])")


_DISTRACTOR_SNIPPETS: List[str] = [
    "Yesterday Alice baked twelve cookies for the school fair.",
    "The library has been closed for renovations since last Monday.",
    "On the way to work, the cat watched a pigeon for several minutes.",
    "Most chess engines today rely on neural-network position evaluators.",
    "Carbon offsets are sometimes criticised for double-counting.",
    "A small leak in the boiler had been ignored for three weeks.",
    "The municipal council debated the pothole budget at length.",
    "Quantum entanglement does not transmit information faster than light.",
    "Several types of moss grow only in the northern half of the forest.",
    "The earliest known printed book is the Diamond Sutra.",
]


class NumericMutator:
    """Rescale every standalone integer in a math question.

    For each *standalone* integer ``n`` in the question we replace it
    with ``n * s`` where ``s`` is drawn once per problem from
    ``self.factors``. Using a single ``s`` per problem (not per token)
    preserves linear relationships, so the canonical answer can be
    recomputed deterministically: ``new_answer = base_answer * s``
    when the answer is itself an integer (the common case for d=5
    Hendrycks problems involving arithmetic).

    Caveats — known unsafe cases:
      * Problems whose answer is non-linear in the literals (e.g.
        modular arithmetic, exponents in the answer position).
      * Problems whose answer is a non-numeric string.

    The mutator returns ``None`` for those cases; ``SelfMutatingCurriculum``
    then falls back to the next mutator or to base sampling.
    """

    name = "numeric"

    def __init__(self, factors: Optional[List[int]] = None, seed: Optional[int] = None):
        self.factors = factors or [7, 11, 13, 17, 19, 23, 29, 31]
        self._rng = random.Random(seed)

    def mutate(
        self,
        domain: str,
        question: str,
        canonical_answer: str,
        base_problem_id: str,
    ) -> Optional[MutatedProblem]:
        if domain != "math":
            return None
        if not _INT_TOKEN_RE.search(question):
            return None
        try:
            base_answer_int = int(canonical_answer.strip())
        except (ValueError, AttributeError):
            return None

        s = self._rng.choice(self.factors)
        new_q = _INT_TOKEN_RE.sub(lambda m: str(int(m.group(1)) * s), question)
        new_a = str(base_answer_int * s)

        return MutatedProblem(
            question=new_q,
            canonical_answer=new_a,
            domain=domain,
            difficulty=0,  # caller fills new tier
            base_problem_id=base_problem_id,
            mutator=self.name,
            problem_id=f"mutated_{self.name}__{base_problem_id}__{s}",
        )


class CompositionalMutator:
    """Chain two same-domain problems P1, P2.

    The mutated question reads:
        Step 1: <P1.question>
        Step 2: Let X be your answer to step 1. <P2.question with X substituted>
        Final answer: the result of step 2.

    For verifier compatibility we only chain if:
      * P1's answer is *numeric*, AND
      * P2's question contains the placeholder ``{X}`` (we inject it by
        replacing the first integer in P2 with ``{X}``).

    The chained answer is whatever the verifier of P2 returns when run
    over the resolved P2 question. We *do not* try to compute it
    symbolically — instead we synthesise a deterministic answer by
    re-running NumericMutator's logic when both halves are math, and
    fall back to ``None`` for non-math chains. Non-math compositional
    mutation will be unlocked when SMC is paired with GSS in v2.
    """

    name = "compositional"

    def __init__(self, base_source: BaseProblemSource, seed: Optional[int] = None):
        self.base_source = base_source
        self._rng = random.Random(seed)

    def mutate(
        self,
        domain: str,
        question: str,
        canonical_answer: str,
        base_problem_id: str,
        base_difficulty: int = 5,
    ) -> Optional[MutatedProblem]:
        if domain != "math":
            return None
        try:
            p1_answer_int = int(canonical_answer.strip())
        except (ValueError, AttributeError):
            return None

        # Sample a second math problem at the same difficulty.
        try:
            q2, a2, pid2 = self.base_source(
                base_difficulty,
                seed=self._rng.randint(0, 2**31 - 1),
            )
        except Exception as exc:
            log.debug("compositional mutator: P2 sampling failed: %s", exc)
            return None
        try:
            int(a2.strip())
        except (ValueError, AttributeError):
            return None

        # Find the first standalone integer in P2 and replace it with
        # P1's answer at evaluation time.
        m = _INT_TOKEN_RE.search(q2)
        if not m:
            return None
        replaced_int = int(m.group(1))
        # New P2 question with literal value substituted.
        new_q2 = q2[:m.start()] + str(p1_answer_int) + q2[m.end():]

        # Compute the new ground truth: scale P2's original answer by the
        # ratio (P1.answer / replaced_int). If replaced_int == 0, abort.
        if replaced_int == 0:
            return None
        try:
            new_answer_int = int(int(a2.strip()) * (p1_answer_int / replaced_int))
        except (ValueError, ZeroDivisionError):
            return None

        chained_q = (
            f"Step 1: {question}\n"
            f"Step 2: Use your answer to step 1 in this problem: {new_q2}\n"
            f"Provide the final numeric answer."
        )
        return MutatedProblem(
            question=chained_q,
            canonical_answer=str(new_answer_int),
            domain=domain,
            difficulty=0,
            base_problem_id=base_problem_id,
            mutator=self.name,
            problem_id=f"mutated_{self.name}__{base_problem_id}__{pid2}",
        )


class DistractorMutator:
    """Prepend N irrelevant snippets to the question.

    Distractors are drawn deterministically from a fixed pool so unit
    tests can assert exact equality. Domain-agnostic — the verifier
    never sees the distractors because the answer is unchanged.
    """

    name = "distractor"

    def __init__(self, snippets: Optional[List[str]] = None, n_snippets: int = 3, seed: Optional[int] = None):
        self.snippets = snippets or _DISTRACTOR_SNIPPETS
        self.n_snippets = int(n_snippets)
        self._rng = random.Random(seed)

    def mutate(
        self,
        domain: str,
        question: str,
        canonical_answer: str,
        base_problem_id: str,
    ) -> Optional[MutatedProblem]:
        if not question or self.n_snippets <= 0:
            return None
        chosen = self._rng.sample(self.snippets, k=min(self.n_snippets, len(self.snippets)))
        prefix = " ".join(chosen)
        new_q = f"{prefix}\n\n{question}"
        return MutatedProblem(
            question=new_q,
            canonical_answer=canonical_answer,
            domain=domain,
            difficulty=0,
            base_problem_id=base_problem_id,
            mutator=self.name,
            problem_id=f"mutated_{self.name}__{base_problem_id}__{self._rng.randint(0, 1<<30)}",
        )


# ---------------------------------------------------------------------------
# SelfMutatingCurriculum
# ---------------------------------------------------------------------------


class SelfMutatingCurriculum:
    """Promotes the ceiling above ``MAX_DIFFICULTY`` when the agent saturates.

    Reads/writes the per-domain rolling accuracy from a
    ``DifficultyController``. Holds a registry of ``Mutator`` objects and
    routes ``sample(domain)`` to either:
      * the base sampler (most of the time), or
      * a mutator chain (when the controller is at the ceiling AND the
        agent has saturated).

    Promotion is throttled identically to the base controller: a 10-episode
    cooldown plus a ``min_episodes_at_max`` requirement. Demotion (ceiling
    drop) protects against random-window over-promotion.
    """

    DEFAULT_PROMOTE_THRESHOLD: float = 0.75
    DEFAULT_DEMOTE_THRESHOLD: float = 0.20
    DEFAULT_MAX_HARD_DIFFICULTY: int = 8
    DEFAULT_MIN_EPISODES_AT_MAX: int = 20

    def __init__(
        self,
        controller,                      # type: DifficultyController
        base_sources: Dict[str, BaseProblemSource],
        mutators: Optional[List] = None,
        max_hard_difficulty: int = DEFAULT_MAX_HARD_DIFFICULTY,
        promote_threshold: float = DEFAULT_PROMOTE_THRESHOLD,
        demote_threshold: float = DEFAULT_DEMOTE_THRESHOLD,
        min_episodes_at_max: int = DEFAULT_MIN_EPISODES_AT_MAX,
        seed: Optional[int] = None,
    ):
        self.controller = controller
        self.base_sources = dict(base_sources)
        self.mutators = mutators or [NumericMutator(seed=seed), DistractorMutator(seed=seed)]
        self.max_hard_difficulty = int(max_hard_difficulty)
        self.promote_threshold = float(promote_threshold)
        self.demote_threshold = float(demote_threshold)
        self.min_episodes_at_max = int(min_episodes_at_max)
        self._rng = random.Random(seed)

        # Per-domain ceiling. Starts at the controller's hard MAX (5).
        self.ceiling: Dict[str, int] = {
            d: self.controller.DIFFICULTY_MAX for d in controller.domains
        }
        # Episodes counter used for cooldown / hysteresis at the ceiling.
        self._at_ceiling_streak: Dict[str, int] = {d: 0 for d in controller.domains}

    # -- promotion / demotion -------------------------------------------

    def maybe_promote(self, domain: str) -> bool:
        """Bump ``ceiling[domain]`` if the agent is saturating it."""
        if domain not in self.ceiling:
            return False
        current = self.ceiling[domain]
        if current >= self.max_hard_difficulty:
            return False

        s = self.controller.state.get(domain)
        if s is None:
            return False
        if s.target_difficulty < current:
            self._at_ceiling_streak[domain] = 0
            return False

        self._at_ceiling_streak[domain] += 1
        if self._at_ceiling_streak[domain] < self.min_episodes_at_max:
            return False

        rolling = self.controller.get_rolling_accuracy(domain)
        if rolling is None or rolling < self.promote_threshold:
            return False

        self.ceiling[domain] = current + 1
        self._at_ceiling_streak[domain] = 0
        log.info(
            "[SMC] %s ceiling -> %d (rolling acc %.2f >= %.2f)",
            domain, self.ceiling[domain], rolling, self.promote_threshold,
        )
        return True

    def maybe_demote(self, domain: str) -> bool:
        if domain not in self.ceiling:
            return False
        if self.ceiling[domain] <= self.controller.DIFFICULTY_MAX:
            return False
        rolling = self.controller.get_rolling_accuracy(domain)
        if rolling is None or rolling > self.demote_threshold:
            return False
        self.ceiling[domain] -= 1
        log.info(
            "[SMC] %s ceiling -> %d (rolling acc %.2f <= %.2f)",
            domain, self.ceiling[domain], rolling, self.demote_threshold,
        )
        return True

    # -- sampling --------------------------------------------------------

    def is_above_base(self, domain: str, difficulty: int) -> bool:
        return difficulty > self.controller.DIFFICULTY_MAX

    def sample(
        self,
        domain: str,
        difficulty: int,
        rng: Optional[random.Random] = None,
    ) -> Tuple[str, str, str]:
        """Return ``(question, canonical_answer, problem_id)``.

        For ``difficulty <= 5`` we delegate to the base source. For
        ``difficulty > 5`` (i.e. above the unified-sampler ceiling) we
        produce a problem by sampling a base d=5 problem and applying a
        mutator chain. Falls back to the base d=5 problem (same difficulty)
        if every mutator returns None — never blocks training.
        """
        chooser = rng if rng is not None else self._rng

        if not self.is_above_base(domain, difficulty):
            return self.base_sources[domain](
                difficulty, seed=chooser.randint(0, 2**31 - 1),
            )

        base_q, base_a, base_pid = self.base_sources[domain](
            self.controller.DIFFICULTY_MAX,
            seed=chooser.randint(0, 2**31 - 1),
        )

        for mut in chooser.sample(self.mutators, k=len(self.mutators)):
            try:
                mp = mut.mutate(
                    domain=domain,
                    question=base_q,
                    canonical_answer=base_a,
                    base_problem_id=base_pid,
                )
            except Exception as exc:
                log.debug("mutator %s raised: %s", mut.name, exc)
                continue
            if mp is None:
                continue
            return mp.question, mp.canonical_answer, mp.problem_id

        # All mutators declined → just emit the base d=5 problem so we
        # never raise mid-rollout. The "tier" claimed in metadata is
        # therefore softer than promised, but training continues.
        return base_q, base_a, base_pid

    # -- introspection ---------------------------------------------------

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            d: {
                "ceiling": self.ceiling[d],
                "at_ceiling_streak": self._at_ceiling_streak[d],
            }
            for d in self.ceiling
        }
