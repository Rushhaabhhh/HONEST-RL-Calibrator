"""Unit tests for Pillar 2 — Calibration-Prioritized Replay."""

import random

import pytest

from server.replay_buffer import CalibrationPrioritizedReplay, ReplayEntry


def _add_calibrated(buf, n, **kw):
    """Add n perfectly-calibrated entries (|c-y| = 0)."""
    for i in range(n):
        buf.add(
            prompt=f"q{i}",
            ground_truth="42",
            domain="math",
            difficulty=1,
            problem_id=f"pid_{i}",
            confidence=1.0,
            correctness=True,
            **kw,
        )


def _add_miscalibrated(buf, n, **kw):
    """Add n maximally-miscalibrated entries (|c-y| = 1)."""
    for i in range(n):
        buf.add(
            prompt=f"qm{i}",
            ground_truth="42",
            domain="math",
            difficulty=1,
            problem_id=f"mid_{i}",
            confidence=1.0,
            correctness=False,
            **kw,
        )


class TestReplayEntry:
    def test_miscalibration_is_cached(self):
        e = ReplayEntry.make("q", "42", "math", 1, "pid", 0.7, True)
        assert e.miscalibration == pytest.approx(0.3)

    def test_clamps_confidence(self):
        e = ReplayEntry.make("q", "42", "math", 1, "pid", 1.5, True)
        assert e.confidence == 1.0

    def test_to_dict_is_jsonable(self):
        import json
        e = ReplayEntry.make("q", "42", "math", 1, "pid", 0.7, True)
        s = json.dumps(e.to_dict())
        assert "miscalibration" in s


class TestBufferLifecycle:
    def test_starts_empty(self):
        buf = CalibrationPrioritizedReplay(capacity=10)
        assert len(buf) == 0
        assert not buf.is_warm(1)
        assert buf.mean_miscalibration() is None
        assert buf.entropy_of_priorities() is None

    def test_add_and_size(self):
        buf = CalibrationPrioritizedReplay(capacity=10)
        _add_calibrated(buf, 3)
        assert len(buf) == 3

    def test_capacity_eviction(self):
        buf = CalibrationPrioritizedReplay(capacity=5)
        _add_calibrated(buf, 7)
        assert len(buf) == 5

    def test_clear(self):
        buf = CalibrationPrioritizedReplay(capacity=10)
        _add_calibrated(buf, 3)
        buf.clear()
        assert len(buf) == 0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            CalibrationPrioritizedReplay(capacity=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            CalibrationPrioritizedReplay(alpha=1.5)


class TestSampling:
    def test_empty_returns_empty(self):
        buf = CalibrationPrioritizedReplay()
        assert buf.sample(5) == []

    def test_n_le_zero_returns_empty(self):
        buf = CalibrationPrioritizedReplay()
        _add_miscalibrated(buf, 3)
        assert buf.sample(0) == []
        assert buf.sample(-1) == []

    def test_sampling_prefers_miscalibrated(self):
        # 1 miscalibrated entry, 1000 perfectly-calibrated entries.
        # With alpha=1, the priority of the miscalibrated entry is
        # ~1.0 vs ~eps for the others. Samples should heavily favor it.
        buf = CalibrationPrioritizedReplay(capacity=2000, alpha=1.0, eps=1e-3, seed=0)
        _add_calibrated(buf, 1000)
        _add_miscalibrated(buf, 1)
        rng = random.Random(0)
        samples = buf.sample(500, rng=rng)
        # The mid_0 entry should appear in *most* samples.
        miscal_hits = sum(1 for s in samples if s.problem_id == "mid_0")
        assert miscal_hits > 250  # >50% of samples

    def test_alpha_zero_is_uniform(self):
        # With alpha=0, all priorities collapse to 1, sampling is uniform.
        buf = CalibrationPrioritizedReplay(capacity=2000, alpha=0.0, seed=0)
        _add_calibrated(buf, 100)
        _add_miscalibrated(buf, 1)
        rng = random.Random(0)
        samples = buf.sample(1000, rng=rng)
        miscal_hits = sum(1 for s in samples if s.problem_id == "mid_0")
        # Roughly uniform: expected hits ~= 1000 / 101 ≈ 10 (well below 50%).
        assert miscal_hits < 100

    def test_with_replacement(self):
        # Single entry → all samples are the same entry.
        buf = CalibrationPrioritizedReplay(seed=0)
        _add_miscalibrated(buf, 1)
        rng = random.Random(0)
        samples = buf.sample(10, rng=rng)
        assert len(samples) == 10
        assert all(s.problem_id == "mid_0" for s in samples)


class TestSnapshot:
    def test_snapshot_keys(self):
        buf = CalibrationPrioritizedReplay(capacity=8, alpha=0.6, eps=1e-3)
        _add_miscalibrated(buf, 3)
        snap = buf.snapshot()
        assert set(snap.keys()) >= {"size", "capacity", "alpha", "eps",
                                    "mean_miscal", "entropy", "max_entropy"}
        assert snap["size"] == 3
        assert snap["mean_miscal"] == pytest.approx(1.0)

    def test_entropy_collapse_warning_signal(self):
        # One very-miscalibrated, many calibrated → low entropy.
        buf = CalibrationPrioritizedReplay(capacity=200, alpha=1.0, eps=1e-6)
        _add_calibrated(buf, 100)
        _add_miscalibrated(buf, 1)
        snap = buf.snapshot()
        # max_entropy = ln(101) ≈ 4.6; entropy with one dominant entry is
        # tiny → snapshot ratio is well below 0.5.
        assert snap["entropy"] / snap["max_entropy"] < 0.5
