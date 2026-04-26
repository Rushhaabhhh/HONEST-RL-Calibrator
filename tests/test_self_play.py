"""Unit tests for Pillar 4 — Generator/Solver Self-Play."""

import random

import pytest

from server.self_play import (
    GeneratedProblem,
    SelfPlayLoop,
    SelfPlayTransition,
    StubProblemGenerator,
    generator_reward,
)


# ---------------------------------------------------------------------------
# generator_reward
# ---------------------------------------------------------------------------


class TestGeneratorReward:
    def test_perfect_calibration_zero(self):
        # Solver was 100% confident and right → no learning signal for generator.
        assert generator_reward(1.0, True) == 0.0
        # Same when 0% and wrong.
        assert generator_reward(0.0, False) == 0.0

    def test_max_miscalibration_one(self):
        assert generator_reward(0.0, True) == 1.0
        assert generator_reward(1.0, False) == 1.0

    def test_partial(self):
        assert generator_reward(0.7, True) == pytest.approx(0.3)

    def test_none_returns_zero(self):
        # No signal when solver abstained or malformed.
        assert generator_reward(None, True) == 0.0
        assert generator_reward(0.7, None) == 0.0


# ---------------------------------------------------------------------------
# StubProblemGenerator
# ---------------------------------------------------------------------------


def _stub_sampler(domain, difficulty):
    return f"Q in {domain} at d={difficulty}", "42", f"pid_{domain}_{difficulty}"


class TestStubProblemGenerator:
    def test_propose_returns_well_formed_problem(self):
        gen = StubProblemGenerator(
            base_sampler=_stub_sampler,
            domains=["math", "code"],
            difficulty_range=(3, 5),
            mutator=None,
            mutate_prob=0.0,
            seed=0,
        )
        p = gen.propose()
        assert isinstance(p, GeneratedProblem)
        assert p.domain in {"math", "code"}
        assert 3 <= p.difficulty <= 5
        assert p.generator_id == "stub"
        assert p.via_mutator is None

    def test_no_mutation_when_prob_zero(self):
        class _NeverShouldRun:
            name = "never"

            def mutate(self, **kw):
                raise AssertionError("must not be called when mutate_prob=0")

        gen = StubProblemGenerator(
            base_sampler=_stub_sampler,
            domains=["math"],
            difficulty_range=(5, 5),
            mutator=_NeverShouldRun(),
            mutate_prob=0.0,
            seed=0,
        )
        p = gen.propose()
        assert p.via_mutator is None

    def test_mutator_runs_when_prob_one(self):
        class _AlwaysMutate:
            name = "always"

            def mutate(self, domain, question, canonical_answer, base_problem_id):
                from server.mutators import MutatedProblem
                return MutatedProblem(
                    question="MUTATED " + question,
                    canonical_answer=canonical_answer,
                    domain=domain,
                    difficulty=0,
                    base_problem_id=base_problem_id,
                    mutator=self.name,
                    problem_id=f"mutated_{self.name}__{base_problem_id}",
                )

        gen = StubProblemGenerator(
            base_sampler=_stub_sampler,
            domains=["math"],
            difficulty_range=(5, 5),
            mutator=_AlwaysMutate(),
            mutate_prob=1.0,
            seed=0,
        )
        p = gen.propose()
        assert p.via_mutator == "always"
        assert "MUTATED" in p.question


# ---------------------------------------------------------------------------
# SelfPlayLoop
# ---------------------------------------------------------------------------


def _calibrated_solver(question):
    """Always confident, always right."""
    return {"answer": "42", "confidence": 1.0, "correct": True, "reward": 0.15}


def _miscalibrated_solver(question):
    """Always confident, always wrong."""
    return {"answer": "wrong", "confidence": 1.0, "correct": False, "reward": -1.0}


def _abstain_solver(question):
    return {"answer": None, "confidence": None, "correct": None, "reward": -0.3}


class TestSelfPlayLoop:
    def _gen(self):
        return StubProblemGenerator(
            base_sampler=_stub_sampler,
            domains=["math"],
            difficulty_range=(5, 5),
            mutator=None,
            mutate_prob=0.0,
            seed=0,
        )

    def test_run_step_returns_transition(self):
        loop = SelfPlayLoop(generator=self._gen(), solver=_calibrated_solver)
        t = loop.run_step()
        assert isinstance(t, SelfPlayTransition)
        assert t.solver_correct is True
        assert t.generator_reward == 0.0  # solver was perfectly calibrated

    def test_miscalibrated_solver_high_generator_reward(self):
        loop = SelfPlayLoop(generator=self._gen(), solver=_miscalibrated_solver)
        t = loop.run_step()
        assert t.generator_reward == pytest.approx(1.0)

    def test_abstain_solver_zero_generator_reward(self):
        loop = SelfPlayLoop(generator=self._gen(), solver=_abstain_solver)
        t = loop.run_step()
        assert t.generator_reward == 0.0

    def test_transitions_buffer_capped(self):
        loop = SelfPlayLoop(generator=self._gen(), solver=_miscalibrated_solver,
                            max_transitions=10)
        for _ in range(25):
            loop.run_step()
        assert len(loop.transitions) == 10

    def test_snapshot_aggregates(self):
        loop = SelfPlayLoop(generator=self._gen(), solver=_miscalibrated_solver)
        for _ in range(5):
            loop.run_step()
        snap = loop.snapshot()
        assert snap["transitions"] == 5
        assert snap["mean_generator_reward"] == pytest.approx(1.0)
        # Diversity ratio: stub_pid is the same for every sample → very low.
        assert 0.0 <= snap["diversity_ratio"] <= 1.0
