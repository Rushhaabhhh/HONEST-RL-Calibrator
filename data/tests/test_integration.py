"""End-to-end integration test for the data-sampler adapter layer.

Simulates exactly what server/environment.py does when it imports and
calls the adapter generate() functions.

Run from the project root:
    PYTHONPATH=. pytest data/tests/test_integration.py -v
"""

from __future__ import annotations

import json
import warnings

import pytest

# ---------------------------------------------------------------------------
# 1. Import the adapter functions the same way the environment would
# ---------------------------------------------------------------------------

from data.sampler.math_gen_adapter  import generate as math_generate
from data.sampler.code_gen_adapter  import generate as code_generate
from data.sampler.logic_gen_adapter import generate as logic_generate
from data.sampler.environment_adapter import get_sampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(fn, difficulty: int, seed: int = 0):
    """Call fn, suppressing fallback warnings for empty buckets."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return fn(difficulty, seed=seed)


# ---------------------------------------------------------------------------
# 2. Each adapter's generate() at every difficulty 1–5
# ---------------------------------------------------------------------------

class TestAdapterSignatures:
    """Verify (str, str, str) contract is preserved for every domain × difficulty."""

    @pytest.mark.parametrize("diff", [1, 2, 3, 4, 5])
    def test_math_generate_returns_str_triple(self, diff):
        q, a, pid = _call(math_generate, diff)
        assert isinstance(q, str) and len(q) > 0, f"diff={diff}: question is empty"
        assert isinstance(a, str),                 f"diff={diff}: answer is not str"
        assert isinstance(pid, str) and len(pid) > 0, f"diff={diff}: problem_id is empty"

    @pytest.mark.parametrize("diff", [1, 2, 3, 4, 5])
    def test_code_generate_returns_str_triple(self, diff):
        q, a, pid = _call(code_generate, diff)
        assert isinstance(q, str) and len(q) > 0, f"diff={diff}: question is empty"
        assert isinstance(a, str),                 f"diff={diff}: answer is not str"
        assert isinstance(pid, str) and len(pid) > 0, f"diff={diff}: problem_id is empty"

    @pytest.mark.parametrize("diff", [1, 2, 3, 4, 5])
    def test_logic_generate_returns_str_triple(self, diff):
        q, a, pid = _call(logic_generate, diff)
        assert isinstance(q, str) and len(q) > 0, f"diff={diff}: question is empty"
        assert isinstance(a, str),                 f"diff={diff}: answer is not str"
        assert isinstance(pid, str) and len(pid) > 0, f"diff={diff}: problem_id is empty"

    def test_logic_answer_is_valid_json_dict_at_d3(self):
        """ZebraLogic canonical answer (difficulty >= 3) must be a JSON-parseable dict."""
        _, a, _ = _call(logic_generate, 3)
        parsed = json.loads(a)
        assert isinstance(parsed, dict) and len(parsed) > 0

    def test_adapters_are_deterministic_with_seed(self):
        q1, a1, p1 = math_generate(2, seed=42)
        q2, a2, p2 = math_generate(2, seed=42)
        assert q1 == q2 and a1 == a2 and p1 == p2

    def test_adapters_vary_without_seed(self):
        """Two calls without a seed should (almost always) return different questions."""
        results = {math_generate(3)[0] for _ in range(5)}
        assert len(results) > 1, "Five un-seeded calls all returned the same question"


# ---------------------------------------------------------------------------
# 3. verify() on known-correct and known-wrong answers
# ---------------------------------------------------------------------------

class TestVerifyDispatch:
    """Confirm UnifiedSampler.verify() dispatches correctly and returns bool."""

    @pytest.fixture(scope="class")
    def sampler(self):
        return get_sampler()

    def _sample_id_and_answer(self, sampler, domain: str, generate_fn, difficulty: int):
        """Sample a problem and use the returned problem_id directly."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, canonical, pid = generate_fn(difficulty, seed=7)

        assert pid in sampler._by_id, (
            f"Returned problem_id={pid!r} not in sampler._by_id for domain={domain}"
        )
        return pid, canonical

    # -- Correct answers should pass --

    def test_math_correct_answer_passes(self, sampler):
        pid, canon = self._sample_id_and_answer(sampler, "math", math_generate, 1)
        assert sampler.verify(pid, canon) is True

    def test_code_correct_answer_passes(self, sampler):
        pid, canon = self._sample_id_and_answer(sampler, "code", code_generate, 1)
        assert sampler.verify(pid, canon) is True

    def test_logic_correct_answer_passes(self, sampler):
        pid, canon = self._sample_id_and_answer(sampler, "logic", logic_generate, 3)
        assert sampler.verify(pid, canon) is True

    # -- Wrong answers should fail --

    def test_math_wrong_answer_fails(self, sampler):
        pid, _ = self._sample_id_and_answer(sampler, "math", math_generate, 1)
        assert sampler.verify(pid, "999999") is False

    def test_logic_wrong_answer_fails(self, sampler):
        pid, _ = self._sample_id_and_answer(sampler, "logic", logic_generate, 3)
        # Completely wrong JSON grid
        wrong = json.dumps({"House 1": {"Name": "WRONG", "Pet": "WRONG", "Drink": "WRONG"}})
        assert sampler.verify(pid, wrong) is False

    def test_verify_returns_bool_type(self, sampler):
        """Ensure return type is exactly bool, not a truthy/falsy value."""
        pid, canon = self._sample_id_and_answer(sampler, "math", math_generate, 2)
        result = sampler.verify(pid, canon)
        assert type(result) is bool  # noqa: E721

    def test_verify_unknown_id_returns_false(self, sampler):
        assert sampler.verify("__nonexistent__", "42") is False

    # -- verify() never raises --

    def test_verify_does_not_raise_on_garbage_input(self, sampler):
        pid, _ = self._sample_id_and_answer(sampler, "math", math_generate, 1)
        # All of these should return False, never raise
        for bad in ["", "\x00\xff", "null", "[]", "{}", "NaN", " "]:
            result = sampler.verify(pid, bad)
            assert isinstance(result, bool), f"verify() raised or returned non-bool for input={bad!r}"


# ---------------------------------------------------------------------------
# 4. Singleton is shared across adapter modules
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_singleton_identity(self):
        """All three adapters share the exact same UnifiedSampler instance."""
        from data.sampler.environment_adapter import get_sampler as ga
        s1 = ga()
        s2 = ga()
        assert s1 is s2

    def test_singleton_loaded_once(self):
        """Second call to get_sampler() does not re-load data (same object)."""
        s1 = get_sampler()
        count_before = s1.total_count()
        s2 = get_sampler()
        assert s2.total_count() == count_before
        assert s1 is s2
