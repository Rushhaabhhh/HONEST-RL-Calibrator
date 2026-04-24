"""Calibration and accuracy metrics for the HONEST-RL-Calibrator.

All functions accept parallel arrays:
    confidences : list/array of float in [0, 1]   — model's stated confidence
    correctness : list/array of bool/int (0 or 1) — whether answer was correct

None values (abstain/malformed) must be filtered out BEFORE calling these.
"""

import numpy as np


def compute_brier(confidences: list, correctness: list) -> float:
    """Mean Brier Score: lower is better (0 = perfect), 1 = worst possible.

    B = mean((confidence - outcome)^2)

    Returns NaN for empty input.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(correctness, dtype=float)
    if len(c) == 0:
        return float("nan")
    return float(np.mean((c - o) ** 2))


def compute_ece(
    confidences: list,
    correctness: list,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (uniform bins).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Returns NaN for empty input.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(correctness, dtype=float)
    n = len(c)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Include upper edge in the last bin
        if hi == 1.0:
            mask = (c >= lo) & (c <= hi)
        else:
            mask = (c >= lo) & (c < hi)
        if mask.sum() == 0:
            continue
        acc = o[mask].mean()
        conf = c[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_ace(
    confidences: list,
    correctness: list,
    n_bins: int = 10,
) -> float:
    """Adaptive Calibration Error (equal-sample bins).

    Like ECE but bins are sized so each has the same number of samples,
    making it less sensitive to sparse regions of the confidence distribution.

    Returns NaN for empty input.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(correctness, dtype=float)
    n = len(c)
    if n == 0:
        return float("nan")

    # Sort by confidence
    order = np.argsort(c)
    c_s = c[order]
    o_s = o[order]

    bin_edges = np.array_split(np.arange(n), n_bins)
    ace = 0.0
    for idxs in bin_edges:
        if len(idxs) == 0:
            continue
        acc = o_s[idxs].mean()
        conf = c_s[idxs].mean()
        ace += (len(idxs) / n) * abs(acc - conf)
    return float(ace)


def compute_mce(
    confidences: list,
    correctness: list,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error — worst-case bin calibration gap.

    MCE = max_b |acc(B_b) - conf(B_b)|  (uniform bins, non-empty bins only)

    Returns NaN for empty input.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(correctness, dtype=float)
    n = len(c)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (c >= lo) & (c <= hi)
        else:
            mask = (c >= lo) & (c < hi)
        if mask.sum() == 0:
            continue
        acc = o[mask].mean()
        conf = c[mask].mean()
        gaps.append(abs(acc - conf))
    if not gaps:
        return float("nan")
    return float(max(gaps))
