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


# ---------------------------------------------------------------------------
# Likelihood + discrimination metrics (NLL, AUROC, AUPRC)
# ---------------------------------------------------------------------------

# NLL is unbounded as confidences approach 0 or 1. We clip to a tiny epsilon
# band before taking the log so a single mis-clipped 0.0 doesn't blow the
# whole batch to +inf. 1e-7 is conservative — at conf=1e-7 the NLL term is
# ~16 nats, large but finite.
_NLL_EPS = 1e-7


def compute_nll(confidences: list, correctness: list) -> float:
    """Negative Log Likelihood — standard calibration paper metric.

        NLL = -mean( y * log(p) + (1-y) * log(1-p) )

    Lower is better. Unbounded above (clipped via _NLL_EPS so a single 0/1
    confidence cannot poison the average). Returns NaN for empty input.

    Note: NLL and Brier rank similarly on most distributions, but NLL is more
    sensitive to extreme over/under-confidence on a few examples — useful as
    a *worst-case* signal alongside ECE/MCE.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(correctness, dtype=float)
    if len(c) == 0:
        return float("nan")
    p = np.clip(c, _NLL_EPS, 1.0 - _NLL_EPS)
    return float(-np.mean(o * np.log(p) + (1.0 - o) * np.log(1.0 - p)))


def compute_auroc(confidences: list, correctness: list) -> float:
    """Area Under ROC Curve, using confidence as the score for distinguishing
    correct (positive) from incorrect (negative).

    Range [0, 1]. 0.5 = random, 1.0 = perfect discrimination.

    Distinct from calibration: a model can have AUROC=0.9 (it can rank correct
    > incorrect well) yet ECE=0.2 (the absolute confidences are in the wrong
    range). Both signals are informative.

    Implementation uses the rank-sum (Mann–Whitney U) identity to avoid pulling
    in scikit-learn:

        AUROC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    where R_pos is the sum of ranks of the positive class. Ties are split via
    average ranks (numpy argsort + average-tie ranking).

    Returns NaN if either class is empty (one-class AUROC is undefined).
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(correctness, dtype=int)
    if len(c) == 0:
        return float("nan")
    n_pos = int((o == 1).sum())
    n_neg = int((o == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average-tie ranking: rank with ties broken by averaging.
    order = np.argsort(c, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    n = len(c)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and c[order[j + 1]] == c[order[i]]:
            j += 1
        # ranks i..j (inclusive) are tied — assign average rank (1-based).
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    r_pos = float(ranks[o == 1].sum())
    auroc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)

