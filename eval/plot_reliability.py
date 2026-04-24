"""
eval/plot_reliability.py — Publication-quality reliability / calibration diagrams.

Usage:
    python eval/plot_reliability.py                          # uses baseline_results.json
    python eval/plot_reliability.py --results path/to.json  # custom results file
    python eval/plot_reliability.py --prefix after_rl        # changes output file prefix

Comparison helper (importable):
    from eval.plot_reliability import plot_comparison
    plot_comparison("eval/baseline_results.json", "eval/after_rl_results.json")
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = {
    "bar":      "#4C72B0",  # Seaborn blue
    "perfect":  "#DD8452",  # Warm orange diagonal
    "gap_pos":  "#55A868",  # Green — overconfidentundershoot
    "gap_neg":  "#C44E52",  # Red   — underconfident
    "bg":       "#F8F9FA",
}

BIN_EDGES = np.linspace(0.0, 1.0, 11)   # 10 bins: [0,.1), [.1,.2), …, [.9,1]
BIN_CENTRES = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
BIN_WIDTH = 0.09         # slightly narrower than the 0.1 spacing for readability


# ---------------------------------------------------------------------------
# Core: build calibration bins from a flat list of (confidence, correct) pairs
# ---------------------------------------------------------------------------

def build_bins(confidences: list, correctness: list) -> dict:
    """Return per-bin stats used by the reliability diagram."""
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)

    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (conf >= lo) & (conf < hi)
        # last bin is inclusive on the right
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        n = mask.sum()
        bin_count.append(int(n))
        bin_acc.append(float(corr[mask].mean()) if n > 0 else np.nan)
        bin_conf.append(float(conf[mask].mean()) if n > 0 else np.nan)

    return {
        "bin_acc":   np.array(bin_acc),
        "bin_conf":  np.array(bin_conf),
        "bin_count": np.array(bin_count),
    }


def compute_ece_from_bins(bins: dict) -> float:
    counts  = bins["bin_count"]
    accs    = bins["bin_acc"]
    confs   = bins["bin_conf"]
    total   = counts.sum()
    if total == 0:
        return float("nan")
    ece = 0.0
    for n, a, c in zip(counts, accs, confs):
        if n > 0 and not np.isnan(a):
            ece += (n / total) * abs(a - c)
    return ece


# ---------------------------------------------------------------------------
# Single reliability diagram
# ---------------------------------------------------------------------------

def _draw_reliability(
    ax: plt.Axes,
    bins: dict,
    ece: float,
    title: str,
    show_counts: bool = True,
):
    """Draw a reliability diagram onto ax."""
    acc   = bins["bin_acc"]
    count = bins["bin_count"]

    # ── background & reference diagonal ─────────────────────────────────────
    ax.set_facecolor(PALETTE["bg"])
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["perfect"],
            linewidth=1.8, label="Perfect calibration", zorder=3)

    # ── gap fill (miscalibration region) ────────────────────────────────────
    for i, (c, a, n) in enumerate(zip(BIN_CENTRES, acc, count)):
        if np.isnan(a) or n == 0:
            continue
        lo, hi = min(c, a), max(c, a)
        color = PALETTE["gap_neg"] if a < c else PALETTE["gap_pos"]
        ax.bar(c, hi - lo, bottom=lo, width=BIN_WIDTH * 0.98,
               color=color, alpha=0.25, zorder=1)

    # ── accuracy bars ────────────────────────────────────────────────────────
    valid = ~np.isnan(acc)
    bars = ax.bar(
        BIN_CENTRES[valid], acc[valid],
        width=BIN_WIDTH, color=PALETTE["bar"],
        edgecolor="white", linewidth=0.6,
        label="Observed accuracy", alpha=0.85, zorder=2,
    )

    # ── count annotations on bars ────────────────────────────────────────────
    if show_counts:
        for bar, n in zip(bars, count[valid]):
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h / 2, f"n={n}",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=5)

    # ── ECE badge ────────────────────────────────────────────────────────────
    ece_text = f"ECE = {ece:.4f}" if not np.isnan(ece) else "ECE = n/a"
    ax.text(0.97, 0.04, ece_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.9))

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel("Confidence (predicted probability)", fontsize=11)
    ax.set_ylabel("Accuracy (fraction correct)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(BIN_EDGES)
    ax.set_xticklabels([f"{e:.1f}" for e in BIN_EDGES], fontsize=8, rotation=45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


# ---------------------------------------------------------------------------
# High-level: single-results plotting
# ---------------------------------------------------------------------------

def extract_pairs(samples: list) -> tuple:
    """Return (confidences, correctness) from a list of sample dicts."""
    confs, corrs = [], []
    for s in samples:
        if s.get("confidence") is not None and s.get("correct") is not None:
            confs.append(float(s["confidence"]))
            corrs.append(1 if s["correct"] else 0)
    return confs, corrs


def plot_domain(
    domain: str,
    conditions: dict,
    out_dir: Path,
    prefix: str = "baseline",
):
    """One 2×3 sub-grid showing each difficulty level + aggregate for a domain."""
    domain_conditions = {
        k: v for k, v in conditions.items() if k.startswith(domain + "_")
    }
    if not domain_conditions:
        print(f"  ! No conditions found for domain '{domain}' — skipping.")
        return

    # collect aggregate pairs for the domain
    agg_conf, agg_corr = [], []
    per_diff = {}
    for key, cond in sorted(domain_conditions.items()):
        diff = int(key.split("_")[-1])
        cf, cr = extract_pairs(cond.get("samples", []))
        per_diff[diff] = (cf, cr)
        agg_conf.extend(cf)
        agg_corr.extend(cr)

    n_diffs = len(per_diff)
    # layout: difficulties in a row + 1 aggregate panel
    ncols = n_diffs + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))
    fig.patch.set_facecolor("white")

    for idx, (diff, (cf, cr)) in enumerate(sorted(per_diff.items())):
        bins = build_bins(cf, cr)
        ece  = compute_ece_from_bins(bins)
        _draw_reliability(
            axes[idx], bins, ece,
            title=f"{domain.capitalize()} — Difficulty {diff}",
        )

    # aggregate panel (last)
    agg_bins = build_bins(agg_conf, agg_corr)
    agg_ece  = compute_ece_from_bins(agg_bins)
    _draw_reliability(
        axes[-1], agg_bins, agg_ece,
        title=f"{domain.capitalize()} — All Difficulties",
    )

    fig.suptitle(
        f"Baseline Calibration - {domain.capitalize()}",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{domain}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


def plot_overall(
    conditions: dict,
    overall: dict,
    out_dir: Path,
    prefix: str = "baseline",
):
    """Aggregate diagram across ALL domains and a per-domain summary bar."""
    all_conf, all_corr = [], []
    domain_eces = {}
    domain_accs = {}

    for domain in ["math", "code", "logic"]:
        d_conf, d_corr = [], []
        for key, cond in conditions.items():
            if key.startswith(domain + "_"):
                cf, cr = extract_pairs(cond.get("samples", []))
                d_conf.extend(cf)
                d_corr.extend(cr)
                all_conf.extend(cf)
                all_corr.extend(cr)
        if d_conf:
            bins = build_bins(d_conf, d_corr)
            domain_eces[domain] = compute_ece_from_bins(bins)
            domain_accs[domain] = float(np.mean(d_corr)) if d_corr else 0.0

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # ── left: overall reliability diagram ───────────────────────────────────
    overall_bins = build_bins(all_conf, all_corr)
    overall_ece  = compute_ece_from_bins(overall_bins)
    _draw_reliability(ax_left, overall_bins, overall_ece,
                      title="Overall Reliability Diagram")

    # ── right: per-domain ECE + accuracy summary bar chart ──────────────────
    domains = list(domain_eces.keys())
    x = np.arange(len(domains))
    width = 0.35
    ece_vals = [domain_eces[d] for d in domains]
    acc_vals = [domain_accs[d] for d in domains]

    bars1 = ax_right.bar(x - width / 2, acc_vals, width,
                         label="Accuracy", color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax_right.bar(x + width / 2, ece_vals, width,
                         label="ECE (lower=better)", color="#C44E52", alpha=0.85, edgecolor="white")

    for bar, val in zip(list(bars1) + list(bars2), acc_vals + ece_vals):
        ax_right.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.01, f"{val:.2f}",
                      ha="center", fontsize=9, fontweight="bold")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels([d.capitalize() for d in domains], fontsize=11)
    ax_right.set_ylim(0, 1.15)
    ax_right.set_ylabel("Score", fontsize=11)
    ax_right.set_title("Per-Domain Summary: Accuracy vs ECE", fontsize=13,
                        fontweight="bold")
    ax_right.legend(fontsize=10)
    ax_right.set_facecolor(PALETTE["bg"])

    fig.suptitle("Baseline Calibration - Overall", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = out_dir / f"{prefix}_overall.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# Comparison: before vs after RL/SFT
# ---------------------------------------------------------------------------

def plot_comparison(
    before_path: str,
    after_path: str,
    out_dir: Optional[str] = None,
    label_before: str = "Before Training",
    label_after: str = "After Training",
):
    """
    Generate side-by-side overall reliability diagrams from two result JSON files.
    Typically called after RL training to visualise improvement.

    Example:
        from eval.plot_reliability import plot_comparison
        plot_comparison("eval/baseline_results.json", "eval/after_rl_results.json")
    """
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    def _collect(results):
        confs, corrs = [], []
        for cond in results["conditions"].values():
            cf, cr = extract_pairs(cond.get("samples", []))
            confs.extend(cf)
            corrs.extend(cr)
        return confs, corrs

    b_conf, b_corr = _collect(before)
    a_conf, a_corr = _collect(after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    b_bins = build_bins(b_conf, b_corr)
    a_bins = build_bins(a_conf, a_corr)

    _draw_reliability(ax1, b_bins, compute_ece_from_bins(b_bins),
                      title=label_before, show_counts=False)
    _draw_reliability(ax2, a_bins, compute_ece_from_bins(a_bins),
                      title=label_after, show_counts=False)

    b_ece = compute_ece_from_bins(b_bins)
    a_ece = compute_ece_from_bins(a_bins)
    delta = b_ece - a_ece
    sign  = "↓" if delta > 0 else "↑"
    fig.suptitle(
        f"Calibration Comparison  |  ECE: {b_ece:.4f} → {a_ece:.4f}  ({sign} {abs(delta):.4f})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    _out_dir = Path(out_dir) if out_dir else Path(before_path).parent / "plots"
    _out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _out_dir / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate reliability / calibration diagrams from eval results."
    )
    parser.add_argument(
        "--results", default="eval/baseline_results.json",
        help="Path to the results JSON (default: eval/baseline_results.json)",
    )
    parser.add_argument(
        "--out-dir", default="eval/plots",
        help="Output directory for PNG files (default: eval/plots)",
    )
    parser.add_argument(
        "--prefix", default="baseline",
        help="Filename prefix for output PNGs (default: baseline)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run baseline_eval.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    conditions = data["conditions"]
    overall    = data.get("overall", {})
    out_dir    = Path(args.out_dir)

    print(f"\nGenerating reliability diagrams from: {results_path}")
    print(f"Output directory: {out_dir}\n")

    domains = sorted({k.rsplit("_", 1)[0] for k in conditions})
    for domain in domains:
        print(f"Domain: {domain}")
        plot_domain(domain, conditions, out_dir, prefix=args.prefix)

    print("\nOverall:")
    plot_overall(conditions, overall, out_dir, prefix=args.prefix)

    print(f"\nDone! {len(domains) + 1} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
