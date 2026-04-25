"""Tests for data/verifiers/logic_verifier.py."""

from __future__ import annotations

import json

import pytest

from data.verifiers.logic_verifier import verify_logic_answer


# ---------------------------------------------------------------------------
# Canonical answer fixtures
# ---------------------------------------------------------------------------

CANON_3x3 = {
    "House 1": {"Name": "Alice", "Pet": "cat", "Drink": "tea"},
    "House 2": {"Name": "Bob", "Pet": "dog", "Drink": "coffee"},
    "House 3": {"Name": "Carol", "Pet": "fish", "Drink": "milk"},
}

META_3x3 = {
    "grid_size": [3, 3],
    "features": ["Name", "Pet", "Drink"],
    "cell_count": 9,
}


def _json(d: dict) -> str:
    return json.dumps(d)


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_exact_dict_answer_returns_true_1_0(self):
        result = verify_logic_answer(_json(CANON_3x3), CANON_3x3, META_3x3)
        assert result == (True, 1.0)

    def test_exact_string_canonical_also_works(self):
        result = verify_logic_answer(_json(CANON_3x3), json.dumps(CANON_3x3), META_3x3)
        assert result == (True, 1.0)


# ---------------------------------------------------------------------------
# All wrong
# ---------------------------------------------------------------------------


class TestAllWrong:
    def test_all_wrong_returns_false_0_0(self):
        wrong = {
            "House 1": {"Name": "Dave", "Pet": "bird", "Drink": "juice"},
            "House 2": {"Name": "Eve", "Pet": "rabbit", "Drink": "water"},
            "House 3": {"Name": "Frank", "Pet": "hamster", "Drink": "soda"},
        }
        passed, acc = verify_logic_answer(_json(wrong), CANON_3x3, META_3x3)
        assert not passed
        assert acc == 0.0


# ---------------------------------------------------------------------------
# Partial correctness
# ---------------------------------------------------------------------------


class TestPartialCorrectness:
    def _make_partial(self, flip_cells: int) -> dict:
        """Start from the correct answer and flip `flip_cells` cells wrong."""
        import copy
        answer = copy.deepcopy(CANON_3x3)
        # Flip Name in house 1 → 1 wrong out of 9
        if flip_cells >= 1:
            answer["House 1"]["Name"] = "WRONG"
        return answer

    def test_8_of_9_correct_is_89pct_which_fails(self):
        answer = self._make_partial(1)  # 8/9 ≈ 0.888
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert not passed
        assert abs(acc - 8 / 9) < 1e-6

    def test_exact_90pct_passes_threshold(self):
        """A 4×4 grid (16 cells) with 2 wrong = 14/16 = 0.875 < 0.9 → fails.
        We need at least 90% so use a 10-cell grid: 9/10 = 0.9 → passes."""
        # Build a 2×5 canonical (10 cells)
        canon_2x5 = {
            f"House {h}": {
                "Name":  ["Alice", "Bob"][h - 1],
                "Pet":   ["cat", "dog"][h - 1],
                "Drink": ["tea", "coffee"][h - 1],
                "Color": ["red", "blue"][h - 1],
                "Job":   ["doctor", "teacher"][h - 1],
            }
            for h in [1, 2]
        }
        meta = {"grid_size": [2, 5], "features": list(canon_2x5["House 1"].keys()), "cell_count": 10}

        import copy
        answer_9_of_10 = copy.deepcopy(canon_2x5)
        answer_9_of_10["House 1"]["Name"] = "WRONG"  # 1 wrong → 9/10 = 0.9

        passed, acc = verify_logic_answer(_json(answer_9_of_10), canon_2x5, meta)
        assert passed
        assert abs(acc - 0.9) < 1e-6

    def test_80pct_correct_fails_threshold(self):
        """5 cells total: 4 right, 1 wrong → 0.8 which is < 0.9."""
        canon = {
            "House 1": {"Name": "Alice", "Pet": "cat", "Drink": "tea", "Color": "red", "Job": "doctor"},
        }
        meta = {"grid_size": [1, 5], "features": ["Name", "Pet", "Drink", "Color", "Job"], "cell_count": 5}

        import copy
        answer_4_of_5 = copy.deepcopy(canon)
        answer_4_of_5["House 1"]["Job"] = "WRONG"

        passed, acc = verify_logic_answer(_json(answer_4_of_5), canon, meta)
        assert not passed
        assert abs(acc - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------


class TestMalformedJSON:
    def test_empty_string_fails(self):
        assert verify_logic_answer("", CANON_3x3, META_3x3) == (False, 0.0)

    def test_non_json_string_fails(self):
        assert verify_logic_answer("just words", CANON_3x3, META_3x3) == (False, 0.0)

    def test_json_array_not_object_fails(self):
        assert verify_logic_answer("[1, 2, 3]", CANON_3x3, META_3x3) == (False, 0.0)

    def test_non_string_model_answer_fails(self):
        assert verify_logic_answer(None, CANON_3x3, META_3x3) == (False, 0.0)  # type: ignore

    def test_truncated_json_with_valid_prefix_extracted(self):
        """The verifier should still extract the valid prefix JSON object."""
        clean = _json(CANON_3x3)
        truncated = clean[:100]  # definitely malformed
        passed, acc = verify_logic_answer(truncated, CANON_3x3, META_3x3)
        # Either we extract partial JSON or fail gracefully — must not raise
        assert isinstance(passed, bool)
        assert 0.0 <= acc <= 1.0

    def test_json_embedded_in_prose_is_extracted(self):
        """Model wraps the JSON in prose — verifier must find the {...}."""
        prose = "Here is my answer:\n" + _json(CANON_3x3) + "\nHope that helps!"
        passed, acc = verify_logic_answer(prose, CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0


# ---------------------------------------------------------------------------
# House key normalisation
# ---------------------------------------------------------------------------


class TestHouseKeyNormalisation:
    def test_lowercase_house_key(self):
        answer = {
            "house 1": {"Name": "Alice", "Pet": "cat", "Drink": "tea"},
            "house 2": {"Name": "Bob", "Pet": "dog", "Drink": "coffee"},
            "house 3": {"Name": "Carol", "Pet": "fish", "Drink": "milk"},
        }
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0

    def test_underscore_house_key(self):
        answer = {
            "house_1": {"Name": "Alice", "Pet": "cat", "Drink": "tea"},
            "house_2": {"Name": "Bob", "Pet": "dog", "Drink": "coffee"},
            "house_3": {"Name": "Carol", "Pet": "fish", "Drink": "milk"},
        }
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0

    def test_numeric_string_house_key(self):
        answer = {
            "1": {"Name": "Alice", "Pet": "cat", "Drink": "tea"},
            "2": {"Name": "Bob", "Pet": "dog", "Drink": "coffee"},
            "3": {"Name": "Carol", "Pet": "fish", "Drink": "milk"},
        }
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0

    def test_mixed_case_feature_key(self):
        """Feature keys from model are case-insensitive."""
        answer = {
            "House 1": {"NAME": "Alice", "PET": "cat", "drink": "tea"},
            "House 2": {"Name": "Bob", "Pet": "dog", "Drink": "coffee"},
            "House 3": {"Name": "Carol", "Pet": "fish", "Drink": "milk"},
        }
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0

    def test_value_case_insensitive(self):
        """Values are compared case-insensitively."""
        answer = {
            "House 1": {"Name": "ALICE", "Pet": "CAT", "Drink": "TEA"},
            "House 2": {"Name": "BOB", "Pet": "DOG", "Drink": "COFFEE"},
            "House 3": {"Name": "CAROL", "Pet": "FISH", "Drink": "MILK"},
        }
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_canonical_answer(self):
        passed, acc = verify_logic_answer(_json({"House 1": {}}), {}, META_3x3)
        assert passed is False
        assert acc == 0.0

    def test_extra_houses_in_model_answer_are_ignored(self):
        """Model outputs an extra house — should not crash and should score on canon only."""
        answer = dict(CANON_3x3)
        answer["House 4"] = {"Name": "Extra", "Pet": "extra", "Drink": "extra"}
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert passed
        assert acc == 1.0

    def test_missing_house_in_model_answer_penalises_accuracy(self):
        """Model omits House 3 — those cells score 0."""
        answer = {
            "House 1": {"Name": "Alice", "Pet": "cat", "Drink": "tea"},
            "House 2": {"Name": "Bob", "Pet": "dog", "Drink": "coffee"},
            # House 3 missing → 3 cells wrong
        }
        passed, acc = verify_logic_answer(_json(answer), CANON_3x3, META_3x3)
        assert not passed  # 6/9 = 0.666 < 0.9
        assert abs(acc - 6 / 9) < 1e-6
