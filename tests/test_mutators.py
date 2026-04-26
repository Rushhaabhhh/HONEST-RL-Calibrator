"""Unit tests for Pillar 3 — Self-Mutating Curriculum + mutators."""

import random
from collections import deque

import pytest

from server.difficulty import DifficultyController, DomainState
from server.mutators import (
    CompositionalMutator,
    DistractorMutator,
    NumericMutator,
    SelfMutatingCurriculum,
)


# ---------------------------------------------------------------------------
# NumericMutator
# ---------------------------------------------------------------------------


class TestNumericMutator:
    def test_scales_integer_answer(self):
        m = NumericMutator(factors=[7], seed=0)
        out = m.mutate(
            domain="math",
            question="What is 3 + 4?",
            canonical_answer="7",
            base_problem_id="base_0",
        )
        assert out is not None
        assert out.canonical_answer == str(7 * 7)
        # Original literals are scaled — every "3" → "21" and "4" → "28".
        assert "21" in out.question
        assert "28" in out.question

    def test_non_math_domain_returns_none(self):
        m = NumericMutator(seed=0)
        out = m.mutate(
            domain="logic",
            question="If A then B; A. What is B?",
            canonical_answer="true",
            base_problem_id="base_0",
        )
        assert out is None

    def test_non_integer_answer_returns_none(self):
        m = NumericMutator(seed=0)
        out = m.mutate(
            domain="math",
            question="What is 1/2 of 10?",
            canonical_answer="2.5",   # non-integer answer
            base_problem_id="base_0",
        )
        assert out is None

    def test_no_integers_in_question_returns_none(self):
        m = NumericMutator(seed=0)
        out = m.mutate(
            domain="math",
            question="What is the answer?",
            canonical_answer="42",
            base_problem_id="base_0",
        )
        assert out is None

    def test_problem_id_format(self):
        m = NumericMutator(factors=[7], seed=0)
        out = m.mutate(
            domain="math", question="3 + 4", canonical_answer="7",
            base_problem_id="base_0",
        )
        assert out.problem_id.startswith("mutated_numeric__base_0__")


# ---------------------------------------------------------------------------
# DistractorMutator
# ---------------------------------------------------------------------------


class TestDistractorMutator:
    def test_prepends_snippets(self):
        m = DistractorMutator(snippets=["Random fact one.", "Random fact two."],
                              n_snippets=2, seed=0)
        out = m.mutate(
            domain="math", question="Q", canonical_answer="A", base_problem_id="b",
        )
        assert out is not None
        assert "Random fact" in out.question
        assert "Q" in out.question
        # Answer is unchanged.
        assert out.canonical_answer == "A"

    def test_zero_snippets_returns_none(self):
        m = DistractorMutator(n_snippets=0, seed=0)
        out = m.mutate(
            domain="math", question="Q", canonical_answer="A", base_problem_id="b",
        )
        assert out is None

    def test_works_for_any_domain(self):
        m = DistractorMutator(snippets=["Fact."], n_snippets=1, seed=0)
        for d in ("math", "code", "logic"):
            out = m.mutate(domain=d, question="Q", canonical_answer="A", base_problem_id="b")
            assert out is not None


# ---------------------------------------------------------------------------
# CompositionalMutator
# ---------------------------------------------------------------------------


class TestCompositionalMutator:
    def test_chains_two_math_problems(self):
        # Stub base sampler always returns "What is 5 + 0?" / "5".
        def stub_source(diff, seed=None):
            return "What is 5 + 0?", "5", "stub_pid"

        m = CompositionalMutator(base_source=stub_source, seed=0)
        out = m.mutate(
            domain="math",
            question="What is 2 + 3?",
            canonical_answer="5",
            base_problem_id="base_0",
            base_difficulty=5,
        )
        assert out is not None
        assert "Step 1" in out.question
        assert "Step 2" in out.question
        # answer was 5 * (5/5) = 5 because the literal we replaced was 5 itself.
        assert out.canonical_answer == "5"

    def test_non_math_returns_none(self):
        def stub_source(diff, seed=None):
            return "Q2", "5", "pid2"
        m = CompositionalMutator(base_source=stub_source, seed=0)
        out = m.mutate(
            domain="logic", question="Q1", canonical_answer="5",
            base_problem_id="base_0",
        )
        assert out is None


# ---------------------------------------------------------------------------
# SelfMutatingCurriculum
# ---------------------------------------------------------------------------


def _math_source(diff, seed=None):
    return f"What is {diff} + {diff}?", str(2 * diff), f"base_d{diff}"


class TestSelfMutatingCurriculum:
    def _make(self, **overrides):
        controller = DifficultyController(domains=["math", "code", "logic"])
        defaults = dict(
            controller=controller,
            base_sources={"math": _math_source, "code": _math_source, "logic": _math_source},
            mutators=[NumericMutator(factors=[3], seed=0), DistractorMutator(seed=0)],
            min_episodes_at_max=2,
            promote_threshold=0.75,
            demote_threshold=0.20,
            max_hard_difficulty=8,
            seed=0,
        )
        defaults.update(overrides)
        smc = SelfMutatingCurriculum(**defaults)
        return controller, smc

    def test_initial_ceiling_is_5(self):
        _, smc = self._make()
        for d in ("math", "code", "logic"):
            assert smc.ceiling[d] == 5

    def test_promote_requires_target_at_max(self):
        ctrl, smc = self._make()
        # Force high accuracy but target=1 (below max).
        ctrl.state["math"].rolling_window = deque([1] * 20, maxlen=20)
        ctrl.state["math"].target_difficulty = 1
        for _ in range(5):
            assert smc.maybe_promote("math") is False

    def test_promote_when_saturated_at_max(self):
        ctrl, smc = self._make()
        ctrl.state["math"].target_difficulty = 5
        ctrl.state["math"].rolling_window = deque([1] * 20, maxlen=20)
        # Streak builds to min_episodes_at_max=2
        smc.maybe_promote("math")
        assert smc.maybe_promote("math") is True
        assert smc.ceiling["math"] == 6

    def test_promote_capped_at_max_hard(self):
        ctrl, smc = self._make(max_hard_difficulty=6, min_episodes_at_max=1)
        ctrl.state["math"].target_difficulty = 5
        ctrl.state["math"].rolling_window = deque([1] * 20, maxlen=20)
        smc.maybe_promote("math")
        assert smc.ceiling["math"] == 6
        # Already at hard max — no further promotion.
        assert smc.maybe_promote("math") is False
        assert smc.ceiling["math"] == 6

    def test_demote_pulls_ceiling_down(self):
        ctrl, smc = self._make(min_episodes_at_max=1)
        # Force ceiling to 6 manually, then demote.
        smc.ceiling["math"] = 6
        ctrl.state["math"].rolling_window = deque([0] * 20, maxlen=20)
        assert smc.maybe_demote("math") is True
        assert smc.ceiling["math"] == 5

    def test_sample_at_or_below_max_uses_base(self):
        _, smc = self._make()
        rng = random.Random(0)
        q, a, pid = smc.sample("math", 3, rng=rng)
        assert pid.startswith("base_d3")

    def test_sample_above_max_uses_mutator(self):
        _, smc = self._make()
        rng = random.Random(0)
        q, a, pid = smc.sample("math", 6, rng=rng)
        # One of the mutators should have produced a mutated_* prefix.
        assert pid.startswith("mutated_") or pid.startswith("base_d5")

    def test_snapshot_keys(self):
        _, smc = self._make()
        snap = smc.snapshot()
        assert set(snap.keys()) == {"math", "code", "logic"}
        for d in snap.values():
            assert "ceiling" in d and "at_ceiling_streak" in d
