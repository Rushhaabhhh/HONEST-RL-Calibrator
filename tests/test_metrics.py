"""Unit tests for eval/metrics.py."""
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.metrics import (
    compute_ace,
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_mce,
    compute_nll,
)


class TestBrier:
    def test_perfect(self):
        assert compute_brier([1.0, 1.0], [1, 1]) == pytest.approx(0.0)

    def test_worst(self):
        assert compute_brier([1.0, 1.0], [0, 0]) == pytest.approx(1.0)

    def test_random_guess(self):
        assert compute_brier([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.25)

    def test_mixed(self):
        # ((0.9-1)^2 + (0.1-0)^2) / 2 = 0.01
        assert compute_brier([0.9, 0.1], [1, 0]) == pytest.approx(0.01)

    def test_empty_nan(self):
        assert math.isnan(compute_brier([], []))

    def test_single_correct(self):
        assert compute_brier([0.8], [1]) == pytest.approx(0.04)

    def test_single_wrong(self):
        assert compute_brier([0.8], [0]) == pytest.approx(0.64)


class TestECE:
    def test_empty_nan(self):
        assert math.isnan(compute_ece([], []))

    def test_in_range(self):
        rng = random.Random(42)
        c = [rng.random() for _ in range(100)]
        o = [rng.randint(0, 1) for _ in range(100)]
        assert 0.0 <= compute_ece(c, o) <= 1.0

    def test_badly_miscalibrated(self):
        assert compute_ece([0.95] * 20, [0] * 20) > 0.8

    def test_single_bin(self):
        # n_bins=1: acc=2/3, conf=0.7, gap = |0.667 - 0.7| ≈ 0.033
        confs = [0.7, 0.8, 0.6]
        correct = [1, 1, 0]
        ece = compute_ece(confs, correct, n_bins=1)
        assert ece == pytest.approx(abs(2/3 - 0.7), rel=1e-3)


class TestACE:
    def test_empty_nan(self):
        assert math.isnan(compute_ace([], []))

    def test_in_range(self):
        rng = random.Random(7)
        c = [rng.random() for _ in range(50)]
        o = [rng.randint(0, 1) for _ in range(50)]
        assert 0.0 <= compute_ace(c, o) <= 1.0

    def test_fewer_samples_than_bins(self):
        result = compute_ace([0.5, 0.3], [1, 0], n_bins=10)
        assert isinstance(result, float) and not math.isnan(result)

    def test_order_invariant(self):
        rng = random.Random(21)
        c = [rng.random() for _ in range(60)]
        o = [rng.randint(0, 1) for _ in range(60)]
        paired = list(zip(c, o))
        rng.shuffle(paired)
        c2, o2 = zip(*paired)
        assert compute_ace(c, o) == pytest.approx(compute_ace(list(c2), list(o2)), rel=1e-9)


class TestMCE:
    def test_empty_nan(self):
        assert math.isnan(compute_mce([], []))

    def test_in_range(self):
        rng = random.Random(55)
        c = [rng.random() for _ in range(100)]
        o = [rng.randint(0, 1) for _ in range(100)]
        assert 0.0 <= compute_mce(c, o) <= 1.0

    def test_mce_ge_ece(self):
        rng = random.Random(13)
        c = [rng.random() for _ in range(100)]
        o = [rng.randint(0, 1) for _ in range(100)]
        assert compute_mce(c, o) >= compute_ece(c, o) - 1e-9

    def test_perfect_one_bin(self):
        # acc=0.8, conf=0.8 → gap=0
        mce = compute_mce([0.8] * 5, [1, 1, 1, 1, 0], n_bins=1)
        assert mce == pytest.approx(0.0, abs=1e-9)

    def test_badly_calibrated_one_bin(self):
        # conf=0.9, acc=0 → MCE ≈ 0.9
        mce = compute_mce([0.9] * 3, [0, 0, 0], n_bins=1)
        assert mce == pytest.approx(0.9, rel=1e-4)


class TestNLL:
    def test_empty_nan(self):
        assert math.isnan(compute_nll([], []))

    def test_perfect_one(self):
        # conf=1 on correct → -log(1-eps) ≈ 1e-7 (clipped, near 0)
        assert compute_nll([1.0], [1]) < 1e-5

    def test_perfect_zero(self):
        # conf=0 on incorrect → -log(1-eps) ≈ 1e-7 too
        assert compute_nll([0.0], [0]) < 1e-5

    def test_worst_one(self):
        # conf=1 on wrong → -log(eps) ≈ 16.1 (clipped at 1e-7 → ln(1e-7))
        nll = compute_nll([1.0], [0])
        assert nll > 10.0  # large but finite

    def test_random_guess(self):
        # conf=0.5 on either label → -log(0.5) = 0.693 for each
        nll = compute_nll([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert nll == pytest.approx(0.693, abs=1e-3)

    def test_lower_when_more_confident_correct(self):
        worse = compute_nll([0.6], [1])
        better = compute_nll([0.9], [1])
        assert better < worse


class TestAUROC:
    def test_empty_nan(self):
        assert math.isnan(compute_auroc([], []))

    def test_one_class_nan(self):
        assert math.isnan(compute_auroc([0.5, 0.6], [1, 1]))
        assert math.isnan(compute_auroc([0.5, 0.6], [0, 0]))

    def test_perfect_separation(self):
        # All correct have higher confidence than all incorrect.
        c = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        o = [1,   1,   1,   0,   0,   0]
        assert compute_auroc(c, o) == pytest.approx(1.0)

    def test_inverted_separation(self):
        # All correct have *lower* confidence than all incorrect.
        c = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        o = [1,   1,   1,   0,   0,   0]
        assert compute_auroc(c, o) == pytest.approx(0.0)

    def test_random_pairing(self):
        # Random labels with same confidence distribution → AUROC ~ 0.5
        rng = random.Random(11)
        c = [rng.random() for _ in range(400)]
        o = [rng.randint(0, 1) for _ in range(400)]
        # Need both classes; reroll if degenerate
        if 0 not in o or 1 not in o:
            o[0], o[1] = 0, 1
        score = compute_auroc(c, o)
        assert 0.40 <= score <= 0.60

    def test_ties_give_half(self):
        # Half the positives and negatives share an identical confidence.
        # With average-tie ranking, AUROC should be 0.5.
        c = [0.5, 0.5, 0.5, 0.5]
        o = [1, 0, 1, 0]
        assert compute_auroc(c, o) == pytest.approx(0.5)
