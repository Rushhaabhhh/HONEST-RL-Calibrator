"""Unit tests for Pillar 1 — Hindsight Calibration Reward."""

import random

import pytest

from server.hindsight import (
    DEFAULT_HINDSIGHT_WEIGHT,
    HindsightCoordinator,
    compute_hindsight_reward,
    parse_hindsight,
    reward_hindsight,
)


# ---------------------------------------------------------------------------
# parse_hindsight
# ---------------------------------------------------------------------------


class TestParseHindsight:
    def test_well_formed_value(self):
        out = parse_hindsight("<hindsight>0.7</hindsight>")
        assert out == {"type": "hindsight", "retrospective": 0.7}

    def test_zero_and_one_are_valid(self):
        assert parse_hindsight("<hindsight>0</hindsight>")["retrospective"] == 0.0
        assert parse_hindsight("<hindsight>1</hindsight>")["retrospective"] == 1.0

    def test_out_of_range_is_malformed(self):
        assert parse_hindsight("<hindsight>1.5</hindsight>")["type"] == "malformed"
        assert parse_hindsight("<hindsight>-0.1</hindsight>")["type"] == "malformed"

    def test_no_tag(self):
        assert parse_hindsight("just text, no tags")["type"] == "malformed"

    def test_non_numeric(self):
        assert parse_hindsight("<hindsight>maybe</hindsight>")["type"] == "malformed"

    def test_none_input(self):
        assert parse_hindsight(None)["type"] == "malformed"

    def test_case_insensitive(self):
        assert parse_hindsight("<HINDSIGHT>0.5</Hindsight>")["retrospective"] == 0.5


# ---------------------------------------------------------------------------
# compute_hindsight_reward
# ---------------------------------------------------------------------------


class TestComputeHindsightReward:
    def test_perfect_correct(self):
        # r=1, y=1 (correct) → R = 0
        assert compute_hindsight_reward(1.0, True) == 0.0

    def test_perfect_wrong(self):
        # r=0, y=0 (wrong) → R = 0
        assert compute_hindsight_reward(0.0, False) == 0.0

    def test_worst_correct(self):
        # r=0, y=1 → R = -k * 1 = -0.3
        assert compute_hindsight_reward(0.0, True) == pytest.approx(-0.3)

    def test_worst_wrong(self):
        # r=1, y=0 → R = -k * 1 = -0.3
        assert compute_hindsight_reward(1.0, False) == pytest.approx(-0.3)

    def test_clamping(self):
        # Out-of-range r is clamped to [0,1] in compute (parse_hindsight
        # rejects, but compute is the lower-level utility and is more lenient).
        assert compute_hindsight_reward(2.0, True) == 0.0
        assert compute_hindsight_reward(-1.0, False) == 0.0

    def test_weight_scales_linearly(self):
        a = compute_hindsight_reward(0.0, True, weight=0.3)
        b = compute_hindsight_reward(0.0, True, weight=0.6)
        assert b == pytest.approx(2 * a)


# ---------------------------------------------------------------------------
# reward_hindsight (TRL-compatible)
# ---------------------------------------------------------------------------


class TestRewardHindsightTRL:
    def test_zero_for_non_hindsight_completion(self):
        out = reward_hindsight(
            ["<answer>4</answer><confidence>0.9</confidence>"],
            previous_correctness=[True],
        )
        assert out == [0.0]

    def test_well_formed_uses_previous_correctness(self):
        out = reward_hindsight(
            ["<hindsight>1.0</hindsight>"],
            previous_correctness=[True],
        )
        assert out == [pytest.approx(0.0)]

    def test_well_formed_with_wrong_y(self):
        # r=1, y=False → very wrong → -0.3
        out = reward_hindsight(
            ["<hindsight>1.0</hindsight>"],
            previous_correctness=[False],
        )
        assert out == [pytest.approx(-0.3)]

    def test_missing_previous_correctness_penalised(self):
        # Slot is invalid without a graded prior answer.
        out = reward_hindsight(
            ["<hindsight>0.7</hindsight>"],
            previous_correctness=[None],
        )
        assert out[0] < 0.0


# ---------------------------------------------------------------------------
# HindsightCoordinator
# ---------------------------------------------------------------------------


class TestHindsightCoordinator:
    def test_off_when_probability_zero(self):
        coord = HindsightCoordinator(probability=0.0)
        assert not coord.is_active()
        rng = random.Random(0)
        assert coord.maybe_request(True, 0.7, rng) is False
        assert not coord.pending()

    def test_always_on_when_probability_one(self):
        coord = HindsightCoordinator(probability=1.0)
        rng = random.Random(0)
        assert coord.maybe_request(True, 0.9, rng) is True
        assert coord.pending()

    def test_consume_clears_state(self):
        coord = HindsightCoordinator(probability=1.0)
        rng = random.Random(0)
        coord.maybe_request(True, 0.9, rng)
        active, y, c = coord.consume()
        assert active is True
        assert y is True
        assert c == 0.9
        assert not coord.pending()
        # Second consume returns inactive.
        a2, y2, c2 = coord.consume()
        assert a2 is False
        assert y2 is None and c2 is None

    def test_correctness_none_never_requests(self):
        coord = HindsightCoordinator(probability=1.0)
        rng = random.Random(0)
        # None correctness (abstain / malformed) — slot must NOT activate
        assert coord.maybe_request(None, 0.5, rng) is False

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            HindsightCoordinator(probability=1.5)
        with pytest.raises(ValueError):
            HindsightCoordinator(probability=-0.1)
