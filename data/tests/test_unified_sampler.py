"""Tests for data/sampler/unified_sampler.py.

Run from the project root:
    PYTHONPATH=. pytest data/tests/test_unified_sampler.py -v
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from data.sampler.unified_sampler import UnifiedSampler

# ---------------------------------------------------------------------------
# Fixture — one shared sampler instance (loading ~20k problems is slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sampler() -> UnifiedSampler:
    return UnifiedSampler()


# ---------------------------------------------------------------------------
# 1. Load without error & report counts
# ---------------------------------------------------------------------------


class TestLoading:
    def test_loads_without_error(self, sampler: UnifiedSampler):
        assert sampler.total_count() > 0

    def test_bucket_counts_reported(self, sampler: UnifiedSampler, capsys):
        counts = sampler.bucket_counts()
        print("\n=== Bucket distribution ===")
        for (domain, diff), n in counts.items():
            print(f"  ({domain:5s}, diff={diff}) -> {n:5d} problems")
        print(f"  TOTAL: {sampler.total_count()}")
        out, _ = capsys.readouterr()
        # Just check it ran; the print() above is the "summary"
        assert len(counts) > 0

    def test_has_math_problems(self, sampler: UnifiedSampler):
        counts = sampler.bucket_counts()
        math_total = sum(n for (d, _), n in counts.items() if d == "math")
        assert math_total > 0, "No math problems loaded"

    def test_has_code_problems(self, sampler: UnifiedSampler):
        counts = sampler.bucket_counts()
        code_total = sum(n for (d, _), n in counts.items() if d == "code")
        assert code_total > 0, "No code problems loaded"

    def test_has_logic_problems(self, sampler: UnifiedSampler):
        counts = sampler.bucket_counts()
        logic_total = sum(n for (d, _), n in counts.items() if d == "logic")
        assert logic_total > 0, "No logic problems loaded"

    def test_by_id_populated(self, sampler: UnifiedSampler):
        assert len(sampler._by_id) == sampler.total_count()


# ---------------------------------------------------------------------------
# 2. Determinism with seed / randomness without
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_math_seed_deterministic(self, sampler: UnifiedSampler):
        q1, a1, p1 = sampler.math_generate(1, seed=42)
        q2, a2, p2 = sampler.math_generate(1, seed=42)
        assert q1 == q2 and a1 == a2 and p1 == p2

    def test_code_seed_deterministic(self, sampler: UnifiedSampler):
        q1, a1, p1 = sampler.code_generate(1, seed=99)
        q2, a2, p2 = sampler.code_generate(1, seed=99)
        assert q1 == q2 and a1 == a2 and p1 == p2

    def test_logic_seed_deterministic(self, sampler: UnifiedSampler):
        q1, a1, p1 = sampler.logic_generate(3, seed=7)
        q2, a2, p2 = sampler.logic_generate(3, seed=7)
        assert q1 == q2 and a1 == a2 and p1 == p2

    def test_different_seeds_give_different_results(self, sampler: UnifiedSampler):
        # Very unlikely to collide with a large enough pool
        q1, *_ = sampler.math_generate(3, seed=1)
        q2, *_ = sampler.math_generate(3, seed=2)
        # With >2700 math diff-3 problems this should almost always differ
        assert q1 != q2, "Same question from two different seeds (pool too small?)"


# ---------------------------------------------------------------------------
# 3. All domains at all five difficulties — graceful fallback on empty buckets
# ---------------------------------------------------------------------------


class TestDomainDifficultyAccess:
    @pytest.mark.parametrize("diff", [1, 2, 3, 4, 5])
    def test_math_all_difficulties(self, sampler: UnifiedSampler, diff):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            q, a, pid = sampler.math_generate(diff, seed=diff)
        assert isinstance(q, str) and len(q) > 0
        assert isinstance(a, str)
        assert isinstance(pid, str) and len(pid) > 0
        if caught:
            # A fallback warning is acceptable — the bucket was empty
            assert "falling back" in str(caught[0].message).lower()

    @pytest.mark.parametrize("diff", [1, 2, 3, 4, 5])
    def test_code_all_difficulties(self, sampler: UnifiedSampler, diff):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            q, a, pid = sampler.code_generate(diff, seed=diff)
        assert isinstance(q, str) and len(q) > 0
        assert isinstance(a, str)
        assert isinstance(pid, str) and len(pid) > 0
        if caught:
            assert "falling back" in str(caught[0].message).lower()

    @pytest.mark.parametrize("diff", [1, 2, 3, 4, 5])
    def test_logic_all_difficulties(self, sampler: UnifiedSampler, diff):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            q, a, pid = sampler.logic_generate(diff, seed=diff)
        assert isinstance(q, str) and len(q) > 0
        assert isinstance(a, str)
        assert isinstance(pid, str) and len(pid) > 0
        if caught:
            assert "falling back" in str(caught[0].message).lower()


# ---------------------------------------------------------------------------
# 4. Return-type contract — (str, str, str)  -- (question, answer, problem_id)
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_math_returns_str_triple(self, sampler: UnifiedSampler):
        result = sampler.math_generate(2, seed=0)
        assert isinstance(result, tuple) and len(result) == 3
        assert all(isinstance(x, str) for x in result)

    def test_code_returns_str_triple(self, sampler: UnifiedSampler):
        result = sampler.code_generate(1, seed=0)
        assert isinstance(result, tuple) and len(result) == 3
        assert all(isinstance(x, str) for x in result)

    def test_logic_returns_str_triple(self, sampler: UnifiedSampler):
        result = sampler.logic_generate(3, seed=0)
        assert isinstance(result, tuple) and len(result) == 3
        assert all(isinstance(x, str) for x in result)

    def test_logic_canonical_answer_is_valid_json(self, sampler: UnifiedSampler):
        """Logic answer must be a JSON-parseable string (dict)."""
        _, answer, _ = sampler.logic_generate(3, seed=0)
        parsed = json.loads(answer)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0


# ---------------------------------------------------------------------------
# 5. verify() dispatches correctly — returns bool
# ---------------------------------------------------------------------------


class TestVerify:
    def _pick_id(self, sampler: UnifiedSampler, domain: str) -> str:
        """Grab the first problem_id for the given domain."""
        for pid, prob in sampler._by_id.items():
            if prob.domain == domain:
                return pid
        raise AssertionError(f"No problem_id found for domain={domain}")

    def test_verify_returns_bool_math(self, sampler: UnifiedSampler):
        pid = self._pick_id(sampler, "math")
        result = sampler.verify(pid, "some_answer")
        assert isinstance(result, bool)

    def test_verify_returns_bool_code(self, sampler: UnifiedSampler):
        pid = self._pick_id(sampler, "code")
        result = sampler.verify(pid, "def f(): pass")
        assert isinstance(result, bool)

    def test_verify_returns_bool_logic(self, sampler: UnifiedSampler):
        pid = self._pick_id(sampler, "logic")
        result = sampler.verify(pid, '{"House 1": {"Name": "X"}}')
        assert isinstance(result, bool)

    def test_verify_unknown_id_returns_false(self, sampler: UnifiedSampler):
        assert sampler.verify("nonexistent_id_xyz", "anything") is False


# ---------------------------------------------------------------------------
# 6. End-to-end: sample → verify canonical_answer → True
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _sample_and_verify(
        self, sampler: UnifiedSampler, domain: str, generate_fn, difficulty: int
    ):
        """Sample a problem, use the returned problem_id, verify its canon answer."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            q, canonical, pid = generate_fn(difficulty, seed=123)

        assert pid in sampler._by_id, (
            f"Returned problem_id={pid!r} not found in _by_id for domain={domain}"
        )
        return sampler.verify(pid, canonical)

    def test_math_self_verify_true(self, sampler: UnifiedSampler):
        passed = self._sample_and_verify(sampler, "math", sampler.math_generate, 1)
        assert passed is True, "math canonical_answer failed its own verifier"

    def test_logic_self_verify_true(self, sampler: UnifiedSampler):
        passed = self._sample_and_verify(sampler, "logic", sampler.logic_generate, 3)
        assert passed is True, "logic canonical_answer failed its own verifier (should score 100%)"

    def test_code_self_verify_true(self, sampler: UnifiedSampler):
        passed = self._sample_and_verify(sampler, "code", sampler.code_generate, 1)
        assert passed is True, "code canonical_answer failed its own test suite"
