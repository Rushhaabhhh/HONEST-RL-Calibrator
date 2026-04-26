"""Research-grade before/after comparison report.

Reads a baseline JSON (from baseline_eval.py) and an after-training JSON
(from full_eval.py), then produces:

    1. A consolidated markdown report (stdout + optional --output file).
    2. Per-domain Brier/ECE deltas with significance flags.
    3. OOD generalization summary (medical + legal).
    4. Calibration-curve + reliability bin breakdown.
    5. A single-line verdict suitable for slide screenshots.

Why a separate script?
    - Keeps `full_eval.py` focused on running inference.
    - Lets you re-render the report from saved JSONs without re-running
      the model (useful when iterating on demo materials).
    - Produces a single artifact (`comparison.md`) that can be copy-pasted
      into a pitch deck.

Usage:
    python eval/compare_runs.py \\
        --baseline eval/baseline_results.json \\
        --after    eval/full_results.json \\
        --output   eval/comparison.md
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (  # noqa: E402
    compute_ace,
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_mce,
    compute_nll,
)

# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bootstrap_brier_ci(
    confidences: List[float],
    correctness: List[int],
    n_iter: int = 500,
    seed: int = 7,
) -> Tuple[float, float]:
    """95% bootstrap confidence interval for Brier score.

    Cheap (≤ 500 resamples × O(N)) and avoids needing scipy. Used to flag
    whether a Brier delta is statistically meaningful at typical
    evaluation sample sizes (N ≈ 100 - 500).
    """
    import random
    n = len(confidences)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    samples = []
    for _ in range(n_iter):
        c, o = [], []
        for _ in range(n):
            i = rng.randrange(n)
            c.append(confidences[i])
            o.append(correctness[i])
        samples.append(compute_brier(c, o))
    samples.sort()
    lo = samples[int(0.025 * n_iter)]
    hi = samples[int(0.975 * n_iter)]
    return (lo, hi)


def _confidence_histogram(
    confidences: List[float],
    n_bins: int = 10,
) -> List[int]:
    """Equal-width bin counts for a quick text histogram."""
    counts = [0] * n_bins
    for c in confidences:
        if c is None:
            continue
        b = min(int(c * n_bins), n_bins - 1)
        counts[b] += 1
    return counts


def _hist_bar(counts: List[int], width: int = 30) -> List[str]:
    total = max(sum(counts), 1)
    bars = []
    for i, c in enumerate(counts):
        lo = i / len(counts)
        hi = (i + 1) / len(counts)
        n_blocks = round((c / total) * width)
        bars.append(f"  [{lo:.1f}–{hi:.1f}) {'█' * n_blocks}{' ' * (width - n_blocks)} {c:>4}")
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────────────────────────────────────


def _flatten_samples(payload: Dict[str, Any], sections: List[str]) -> List[Dict]:
    """Pull every per-sample record from a nested results JSON."""
    out: List[Dict] = []
    for sec in sections:
        section_data = payload.get(sec, {})
        if not isinstance(section_data, dict):
            continue
        for cond in section_data.values():
            for s in cond.get("samples", []):
                out.append(s)
    return out


def _conf_correct(samples: List[Dict]) -> Tuple[List[float], List[int]]:
    confs, corrects = [], []
    for s in samples:
        if s.get("confidence") is not None and s.get("correct") is not None:
            confs.append(float(s["confidence"]))
            corrects.append(1 if s["correct"] else 0)
    return confs, corrects


def _summary(samples: List[Dict]) -> Dict[str, float]:
    confs, corrects = _conf_correct(samples)
    n = len(samples)
    n_correct = sum(1 for s in samples if s.get("correct") is True)
    n_format = sum(1 for s in samples if s.get("format_valid"))
    n_abstain = sum(1 for s in samples if s.get("parsed_type") == "abstain")
    n_malformed = sum(1 for s in samples if s.get("parsed_type") == "malformed")
    rewards = [float(s.get("reward", 0.0)) for s in samples]

    brier_lo, brier_hi = _bootstrap_brier_ci(confs, corrects)

    return {
        "n":             n,
        "accuracy":      n_correct / n if n else 0.0,
        "format_rate":   n_format / n if n else 0.0,
        "abstain_rate":  n_abstain / n if n else 0.0,
        "malformed_rate": n_malformed / n if n else 0.0,
        "mean_conf":     (sum(confs) / len(confs)) if confs else 0.0,
        "mean_reward":   (sum(rewards) / len(rewards)) if rewards else 0.0,
        "brier":         compute_brier(confs, corrects),
        "ece":           compute_ece(confs, corrects),
        "ace":           compute_ace(confs, corrects),
        "mce":           compute_mce(confs, corrects),
        "nll":           compute_nll(confs, corrects),
        "auroc":         compute_auroc(confs, corrects),
        "brier_ci_lo":   brier_lo,
        "brier_ci_hi":   brier_hi,
    }


def _per_domain(samples: List[Dict], domain_keys: List[str]) -> Dict[str, Dict]:
    """Group by question-source domain."""
    by_dom: Dict[str, List[Dict]] = {d: [] for d in domain_keys}
    by_dom["other"] = []
    for s in samples:
        src = s.get("source") or s.get("domain") or "other"
        by_dom.setdefault(src, []).append(s)
    return {k: _summary(v) for k, v in by_dom.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────


def _delta_line(label: str, before: float, after: float, lower_better: bool = True,
                fmt: str = "{:.4f}") -> str:
    if math.isnan(before) or math.isnan(after):
        return f"  {label:<24} {fmt.format(before):>10}  ->  {fmt.format(after):>10}"
    delta = after - before
    if lower_better:
        sym = "✓ better" if delta < 0 else ("• same" if abs(delta) < 1e-4 else "✗ worse")
    else:
        sym = "✓ better" if delta > 0 else ("• same" if abs(delta) < 1e-4 else "✗ worse")
    return (
        f"  {label:<24} {fmt.format(before):>10}  ->  {fmt.format(after):>10}  "
        f"(Δ {delta:+.4f}  {sym})"
    )


def _verdict(before: Dict, after: Dict) -> str:
    """One-line verdict: PASS / PARTIAL / FAIL."""
    db = after["brier"] - before["brier"]
    de = after["ece"] - before["ece"]
    da = after["accuracy"] - before["accuracy"]

    # Brier should improve, ECE should improve, accuracy not regress > 5pp.
    if db < -0.005 and de < -0.005 and da > -0.05:
        return "PASS  (Brier ↓, ECE ↓, accuracy preserved)"
    if db < -0.005 or de < -0.005:
        return "PARTIAL  (one of {Brier, ECE} improved)"
    return "FAIL  (no clear calibration gain)"


def render_report(baseline: Dict[str, Any], after: Dict[str, Any]) -> str:
    sections = ["in_distribution", "ood"]
    base_samples  = _flatten_samples(baseline.get("conditions") and {"in_distribution": baseline["conditions"]} or baseline, sections)
    after_samples = _flatten_samples(after,    sections)

    # Baseline files store conditions at top level, full_eval stores nested.
    if not base_samples:
        # baseline_eval.py shape: {conditions: {...}}
        base_samples = []
        for cond in baseline.get("conditions", {}).values():
            base_samples.extend(cond.get("samples", []))

    bs = _summary(base_samples)
    af = _summary(after_samples)

    # Per-domain breakdown
    base_dom  = _per_domain(base_samples,  ["math", "code", "logic", "medical", "legal"])
    after_dom = _per_domain(after_samples, ["math", "code", "logic", "medical", "legal"])

    # In-distribution / OOD split for "after" (which carries section info)
    indist_samples = []
    for cond in after.get("in_distribution", {}).values():
        indist_samples.extend(cond.get("samples", []))
    ood_samples = []
    for cond in after.get("ood", {}).values():
        ood_samples.extend(cond.get("samples", []))

    indist_after = _summary(indist_samples) if indist_samples else None
    ood_after    = _summary(ood_samples)    if ood_samples    else None

    out: List[str] = []
    out.append("# HONEST-RL Calibration: Before vs After")
    out.append("")
    out.append(f"- **Baseline run**:   `{baseline.get('model_id')}`  "
               f"(preset={baseline.get('preset', '?')}, "
               f"reasoning={baseline.get('reasoning_mode', '?')})")
    out.append(f"- **Trained run**:    `{after.get('model_id')}`  "
               f"(preset={after.get('preset', '?')}, "
               f"reasoning={after.get('reasoning_mode', '?')})")
    out.append(f"- **Adapter**:        `{after.get('adapter_path')}`")
    out.append(f"- **Baseline N**:     {bs['n']}   |   **Trained N**: {af['n']}")
    out.append("")

    # ── 1. Headline ────────────────────────────────────────────────────────
    out.append("## 1. Headline")
    out.append("```")
    out.append(_delta_line("Brier      (↓)",     bs["brier"],     af["brier"]))
    out.append(_delta_line("ECE        (↓)",     bs["ece"],       af["ece"]))
    out.append(_delta_line("ACE        (↓)",     bs["ace"],       af["ace"]))
    out.append(_delta_line("MCE        (↓)",     bs["mce"],       af["mce"]))
    out.append(_delta_line("NLL        (↓)",     bs["nll"],       af["nll"]))
    out.append(_delta_line("AUROC      (↑)",     bs["auroc"],     af["auroc"], lower_better=False))
    out.append(_delta_line("Accuracy   (↑)",     bs["accuracy"],  af["accuracy"], lower_better=False, fmt="{:.1%}"))
    out.append(_delta_line("Format OK  (↑)",     bs["format_rate"], af["format_rate"], lower_better=False, fmt="{:.1%}"))
    out.append(_delta_line("MeanConf   (—)",     bs["mean_conf"], af["mean_conf"], lower_better=False))
    out.append(_delta_line("MeanReward (↑)",     bs["mean_reward"], af["mean_reward"], lower_better=False))
    out.append("```")
    out.append("")
    out.append(f"**Verdict:** {_verdict(bs, af)}")
    out.append("")
    out.append(f"_95% bootstrap Brier CI:_  baseline=[{bs['brier_ci_lo']:.4f}, "
               f"{bs['brier_ci_hi']:.4f}],  after=[{af['brier_ci_lo']:.4f}, "
               f"{af['brier_ci_hi']:.4f}]")
    out.append("")

    # ── 2. Per-domain breakdown ────────────────────────────────────────────
    out.append("## 2. Per-Domain Breakdown")
    out.append("")
    out.append("| Domain | N (before/after) | Brier before | Brier after | ΔBrier | "
               "ECE before | ECE after | ΔECE | Acc before | Acc after |")
    out.append("|--------|-----------------:|-------------:|------------:|-------:|"
               "-----------:|----------:|-----:|-----------:|----------:|")
    for d in ["math", "code", "logic", "medical", "legal"]:
        b = base_dom.get(d)
        a = after_dom.get(d)
        if not b and not a:
            continue
        b = b or {"n": 0, "brier": float("nan"), "ece": float("nan"), "accuracy": 0.0}
        a = a or {"n": 0, "brier": float("nan"), "ece": float("nan"), "accuracy": 0.0}
        db = (a["brier"] - b["brier"]) if not (math.isnan(a["brier"]) or math.isnan(b["brier"])) else float("nan")
        de = (a["ece"]   - b["ece"])   if not (math.isnan(a["ece"])   or math.isnan(b["ece"]))   else float("nan")
        out.append(
            f"| {d:<7} | {b['n']:>3}/{a['n']:<3} | "
            f"{b['brier']:.4f} | {a['brier']:.4f} | {db:+.4f} | "
            f"{b['ece']:.4f} | {a['ece']:.4f} | {de:+.4f} | "
            f"{b['accuracy']:.1%} | {a['accuracy']:.1%} |"
        )
    out.append("")

    # ── 3. In-dist vs OOD ──────────────────────────────────────────────────
    if indist_after or ood_after:
        out.append("## 3. In-Distribution vs OOD (after training)")
        out.append("")
        out.append("| Section          |  N  |  Brier  |   ECE   |   ACE   |   MCE   |   NLL   |  AUROC  |  Acc  |")
        out.append("|------------------|----:|--------:|--------:|--------:|--------:|--------:|--------:|------:|")
        if indist_after:
            o = indist_after
            out.append(
                f"| In-distribution  | {o['n']:>3} | {o['brier']:.4f} | {o['ece']:.4f} | "
                f"{o['ace']:.4f} | {o['mce']:.4f} | {o['nll']:.4f} | {o['auroc']:.4f} | "
                f"{o['accuracy']:.1%} |"
            )
        if ood_after:
            o = ood_after
            out.append(
                f"| OOD              | {o['n']:>3} | {o['brier']:.4f} | {o['ece']:.4f} | "
                f"{o['ace']:.4f} | {o['mce']:.4f} | {o['nll']:.4f} | {o['auroc']:.4f} | "
                f"{o['accuracy']:.1%} |"
            )
        if indist_after and ood_after:
            gap = ood_after["brier"] - indist_after["brier"]
            note = "good (small gap)" if gap < 0.05 else \
                   "moderate gap" if gap < 0.10 else "large gap (overfitting risk)"
            out.append("")
            out.append(f"**Generalization gap (Brier OOD − Brier indist):** {gap:+.4f}  ({note})")
        out.append("")

    # ── 4. Confidence histogram (after) ────────────────────────────────────
    confs_b, _ = _conf_correct(base_samples)
    confs_a, _ = _conf_correct(after_samples)

    out.append("## 4. Confidence Distribution (text histogram)")
    out.append("")
    out.append("```")
    out.append("BEFORE")
    out.extend(_hist_bar(_confidence_histogram(confs_b)))
    out.append("")
    out.append("AFTER")
    out.extend(_hist_bar(_confidence_histogram(confs_a)))
    out.append("```")
    out.append("")
    out.append("_Read: well-calibrated models show mass spread across [0.1, 0.9]; "
               "a single tall spike near 0.5 indicates hedging collapse, "
               "and a tall spike at 0.9+ indicates overconfidence._")
    out.append("")

    # ── 5. Operating mode shifts ──────────────────────────────────────────
    out.append("## 5. Operating-Mode Shifts")
    out.append("")
    out.append("| Mode      | Before | After  | Δ      |")
    out.append("|-----------|-------:|-------:|-------:|")
    for k, label in [("format_rate", "Format OK"), ("abstain_rate", "Abstain"),
                     ("malformed_rate", "Malformed")]:
        b = bs[k]
        a = af[k]
        out.append(f"| {label:<9} | {b:.1%} | {a:.1%} | {a-b:+.1%} |")
    out.append("")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Generate before/after calibration report.")
    ap.add_argument("--baseline", type=str, default="eval/baseline_results.json")
    ap.add_argument("--after",    type=str, default="eval/full_results.json")
    ap.add_argument("--output",   type=str, default=None,
                    help="Optional markdown file path. Defaults to stdout only.")
    ap.add_argument("--plot",     action="store_true",
                    help="Also render side-by-side reliability diagrams via "
                         "eval/plot_reliability.py.")
    ap.add_argument("--plot-output", type=str, default=None,
                    help="PNG path for the reliability diagram (used with --plot).")
    args = ap.parse_args()

    bp = Path(args.baseline)
    ap_ = Path(args.after)
    if not bp.exists():
        sys.exit(f"Baseline JSON not found: {bp}")
    if not ap_.exists():
        sys.exit(f"After JSON not found: {ap_}")

    baseline = json.loads(bp.read_text())
    after    = json.loads(ap_.read_text())

    report = render_report(baseline, after)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nReport written -> {out_path}", file=sys.stderr)

    if args.plot:
        try:
            from eval.plot_reliability import plot_comparison
            png_path = plot_comparison(
                str(bp),
                str(ap_),
                output_path=args.plot_output,
                label_before="Before GRPO",
                label_after="After GRPO",
            )
            print(f"Reliability diagram written -> {png_path}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — plot failure is non-fatal
            print(f"(reliability plot skipped: {exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
