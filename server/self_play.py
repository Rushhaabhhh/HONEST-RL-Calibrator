"""Generator/Solver Self-Play (GSS) — Pillar 4 of self-learning calibration.

PAIRED-style asymmetric self-play where:
  * The **generator** proposes problems.
  * The **solver** is the policy under training.
  * The generator is rewarded by the solver's calibration error on the
    proposed problem — high calibration error means the generator found a
    learning-frontier problem (see SELF_LEARNING.md §5).

For v1 we ship a deterministic *stubbed* generator that:
  1. Samples a base problem from the unified sampler.
  2. Applies a random pillar-3 mutator with probability ``mutate_prob``.
  3. Returns the resulting (q, a, pid).

This stub still exercises the full GSS protocol end-to-end, lets the
solver collect calibration-error feedback in a buffer, and gives us a
typed transition object that v2 can swap a real generator policy into
without changing the caller.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("self_play")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class GeneratedProblem:
    """One problem produced by the generator."""
    question: str
    canonical_answer: str
    domain: str
    difficulty: int
    problem_id: str
    generator_id: str = "stub"      # name of the generator producing this
    via_mutator: Optional[str] = None


@dataclass
class SelfPlayTransition:
    """One round of generator → solver → reward.

    Stored in a list inside ``SelfPlayLoop`` so a future generator-policy
    can be trained on these transitions later.
    """
    problem: GeneratedProblem
    solver_answer: Optional[str]
    solver_confidence: Optional[float]
    solver_correct: Optional[bool]
    solver_reward: float
    generator_reward: float


SolverFn = Callable[[str], Dict[str, Any]]
"""Signature of the solver: takes a question, returns a dict containing
``answer``, ``confidence``, ``correct``, ``reward``. The trainer-side
adapter wraps the policy + verifier + reward into this single function."""

BaseSamplerFn = Callable[[str, int], Tuple[str, str, str]]
"""Signature of the base-problem sampler: ``(domain, difficulty) -> (q, a, pid)``."""


# ---------------------------------------------------------------------------
# Stub generator
# ---------------------------------------------------------------------------


class StubProblemGenerator:
    """Deterministic generator: base sampler + optional pillar-3 mutator.

    Used as the v1 generator because a learned generator policy would
    need its own RL loop with its own KL-stable schedule. The stub
    captures the same coupling (generator difficulty ↔ solver
    miscalibration) at a fraction of the compute. See SELF_LEARNING.md §5.2.
    """

    name = "stub"

    def __init__(
        self,
        base_sampler: BaseSamplerFn,
        domains: List[str],
        difficulty_range: Tuple[int, int] = (3, 5),
        mutator=None,                              # any object with .mutate(...)
        mutate_prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.base_sampler = base_sampler
        self.domains = list(domains)
        self.difficulty_range = (int(difficulty_range[0]), int(difficulty_range[1]))
        self.mutator = mutator
        self.mutate_prob = float(mutate_prob)
        self._rng = random.Random(seed)

    def propose(self, rng: Optional[random.Random] = None) -> GeneratedProblem:
        chooser = rng if rng is not None else self._rng
        domain = chooser.choice(self.domains)
        difficulty = chooser.randint(*self.difficulty_range)

        try:
            q, a, pid = self.base_sampler(domain, difficulty)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("StubProblemGenerator.base_sampler raised: %s — falling back", exc)
            return GeneratedProblem(
                question="What is 2 + 2?",
                canonical_answer="4",
                domain="math",
                difficulty=1,
                problem_id="self_play_fallback",
                generator_id=self.name,
            )

        # Optional mutation step. We do not require the mutator to succeed —
        # if it returns None, we keep the base problem.
        if self.mutator is not None and chooser.random() < self.mutate_prob:
            mp = None
            try:
                mp = self.mutator.mutate(
                    domain=domain,
                    question=q,
                    canonical_answer=a,
                    base_problem_id=pid,
                )
            except Exception as exc:
                log.debug("self-play mutator %s raised: %s", self.mutator.name, exc)
            if mp is not None:
                return GeneratedProblem(
                    question=mp.question,
                    canonical_answer=mp.canonical_answer,
                    domain=domain,
                    difficulty=difficulty,
                    problem_id=mp.problem_id,
                    generator_id=self.name,
                    via_mutator=mp.mutator,
                )

        return GeneratedProblem(
            question=q,
            canonical_answer=a,
            domain=domain,
            difficulty=difficulty,
            problem_id=pid,
            generator_id=self.name,
        )


# ---------------------------------------------------------------------------
# Self-play loop
# ---------------------------------------------------------------------------


def generator_reward(solver_confidence: Optional[float], solver_correct: Optional[bool]) -> float:
    """Reward the generator received when the solver scored on its problem.

    R_G = |c_S - y|         in [0, 1]
        = 0  when solver was perfectly calibrated (no learning signal)
        = 1  when solver was maximally miscalibrated (high learning signal)

    Returns 0.0 if either input is None — abstain / malformed solver
    responses give no calibration signal.
    """
    if solver_confidence is None or solver_correct is None:
        return 0.0
    y = 1.0 if solver_correct else 0.0
    return abs(float(solver_confidence) - y)


class SelfPlayLoop:
    """Run generator → solver rollouts and store the transitions.

    Stateful: keeps a bounded list of ``SelfPlayTransition`` objects for
    inspection, logging, and future generator-policy training. The buffer
    is intentionally separate from the calibration replay buffer — entries
    here are tagged with the generator that produced them.
    """

    def __init__(
        self,
        generator: StubProblemGenerator,
        solver: SolverFn,
        max_transitions: int = 1024,
    ):
        self.generator = generator
        self.solver = solver
        self.max_transitions = int(max_transitions)
        self.transitions: List[SelfPlayTransition] = []

    def run_step(self, rng: Optional[random.Random] = None) -> SelfPlayTransition:
        problem = self.generator.propose(rng=rng)
        result = self.solver(problem.question)

        s_conf = result.get("confidence")
        s_correct = result.get("correct")
        s_reward = float(result.get("reward", 0.0))
        s_ans = result.get("answer")

        gen_reward = generator_reward(s_conf, s_correct)

        transition = SelfPlayTransition(
            problem=problem,
            solver_answer=s_ans,
            solver_confidence=(float(s_conf) if s_conf is not None else None),
            solver_correct=(bool(s_correct) if s_correct is not None else None),
            solver_reward=s_reward,
            generator_reward=gen_reward,
        )

        self.transitions.append(transition)
        if len(self.transitions) > self.max_transitions:
            self.transitions = self.transitions[-self.max_transitions:]

        return transition

    # -- aggregated stats for logging ----------------------------------

    def mean_generator_reward(self, last_n: int = 50) -> Optional[float]:
        if not self.transitions:
            return None
        recent = self.transitions[-last_n:]
        return sum(t.generator_reward for t in recent) / len(recent)

    def diversity_ratio(self, last_n: int = 50) -> Optional[float]:
        """Fraction of unique problem_ids in the last ``last_n`` transitions.

        Used as a guardrail: a stub generator should produce diverse
        problems. If diversity collapses below 0.1 we re-seed it to avoid
        feeding the solver the same prompt over and over (SELF_LEARNING.md
        §9, GSS row).
        """
        if not self.transitions:
            return None
        recent = self.transitions[-last_n:]
        if not recent:
            return None
        return len(set(t.problem.problem_id for t in recent)) / len(recent)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "transitions":            len(self.transitions),
            "mean_generator_reward":  self.mean_generator_reward(),
            "diversity_ratio":        self.diversity_ratio(),
        }
