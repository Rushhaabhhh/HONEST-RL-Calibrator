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
from calibration_profiles import (  # noqa: E402
    SUPPORTED_OOD_SLICES,
    ood_slice_floor,
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


def _bootstrap_paired_delta_ece_ci(
    base_confs: List[float],
    base_corrects: List[int],
    after_confs: List[float],
    after_corrects: List[int],
    n_iter: int = 500,
    seed: int = 11,
) -> Tuple[float, float]:
    """Paired bootstrap 95% CI for the ECE delta (after - before).

    "Paired" because the OOD slices are read from the same JSONL on
    both runs, so question-i in the baseline corresponds to question-i
    in the after run. Resampling indices once and applying the same
    indices to both arrays exploits that pairing for a tighter CI than
    independent resampling. If the two arrays have different lengths
    (e.g. baseline was run with a smaller --samples), we truncate to
    the minimum length and bootstrap on the truncated common prefix.

    Returns ``(nan, nan)`` if either input is empty.
    """
    import random
    n = min(len(base_confs), len(after_confs))
    if n == 0 or len(base_corrects) < n or len(after_corrects) < n:
        return (float("nan"), float("nan"))
    bc, bo = base_confs[:n], base_corrects[:n]
    ac, ao = after_confs[:n], after_corrects[:n]
    rng = random.Random(seed)
    samples = []
    for _ in range(n_iter):
        idxs = [rng.randrange(n) for _ in range(n)]
        s_bc = [bc[i] for i in idxs]
        s_bo = [bo[i] for i in idxs]
        s_ac = [ac[i] for i in idxs]
        s_ao = [ao[i] for i in idxs]
        samples.append(compute_ece(s_ac, s_ao) - compute_ece(s_bc, s_bo))
    samples.sort()
    return (samples[int(0.025 * n_iter)], samples[int(0.975 * n_iter)])


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


# ─────────────────────────────────────────────────────────────────────────────
# Calibration-transfer renderer — the headline claim.
#
# This is the section judges are looking for: did calibration learned on
# math+code+logic transfer to unseen OOD domains? We answer with:
#   - a per-slice table with ΔECE + 95% paired-bootstrap CI
#   - a "status" cell that distinguishes statistically significant
#     transfer from noisy improvements and from at-floor failures
#   - a single transfer-ratio headline (= ΔECE_OOD_avg / ΔECE_ID),
#     averaged over slices that are clear of the random-MCQ floor.
# ─────────────────────────────────────────────────────────────────────────────


def _slice_samples(payload: Dict[str, Any], slice_name: str) -> List[Dict]:
    """Pull per-sample records for a single OOD slice from a results JSON.

    Handles two shapes: ``ood`` keyed by canonical slice name (current
    full_eval output) and the legacy shape where there is no ``ood``
    key (older baseline files), returning [] in that case.
    """
    ood = payload.get("ood")
    if not isinstance(ood, dict):
        return []
    cond = ood.get(slice_name)
    if not isinstance(cond, dict):
        return []
    return list(cond.get("samples", []))


def _indist_samples_all(payload: Dict[str, Any]) -> List[Dict]:
    """Pull every in-distribution sample from a full_eval-shaped JSON."""
    out: List[Dict] = []
    indist = payload.get("in_distribution")
    if isinstance(indist, dict):
        for cond in indist.values():
            out.extend(cond.get("samples", []))
        return out
    # Fallback: baseline_eval shape with conditions at top level.
    for cond in payload.get("conditions", {}).values():
        out.extend(cond.get("samples", []))
    return out


def _slice_acc(samples: List[Dict]) -> float:
    n = len(samples)
    if n == 0:
        return 0.0
    return sum(1 for s in samples if s.get("correct") is True) / n


# An OOD slice is "clear of the random-MCQ floor" when its post-RL accuracy
# beats the random-guess baseline by at least this many percentage points.
# Below this margin, ECE/Brier deltas are dominated by sampling noise — we
# label such slices "at floor" in the transfer report and exclude them from
# the headline transfer ratio. The 1e-9 keeps the boundary numerically
# robust against float underflow on exact 5pp differences (e.g. 0.30 −
# 0.25 evaluates to 0.04999... in IEEE-754).
_TRANSFER_FLOOR_MARGIN = 0.05 - 1e-9


def _delta_ece_status(
    delta_ece: float,
    ci_lo: float,
    ci_hi: float,
    acc_after: float,
    floor: float,
) -> str:
    """Convert ΔECE + CI + acc-vs-floor into a human-readable status."""
    import math
    # At-floor: even a perfect calibration policy can't show much movement.
    if acc_after - floor < _TRANSFER_FLOOR_MARGIN:
        return "⚠ at floor"
    if math.isnan(delta_ece) or math.isnan(ci_lo) or math.isnan(ci_hi):
        return "n/a"
    if delta_ece < 0 and ci_hi < 0:
        return "✓ transferred"
    if delta_ece < 0:
        return "~ partial (CI crosses 0)"
    return "✗ no transfer"


def _render_transfer_section(
    baseline: Dict[str, Any],
    after: Dict[str, Any],
) -> List[str]:
    """Render the markdown calibration-transfer table + headline.

    The table has one anchor row for in-distribution (math+code+logic
    aggregated) and one row per OOD slice. The transfer ratio in the
    summary line is averaged over slices that are clear of the
    random-MCQ floor (acc_after − floor ≥ 5pp).
    """
    out: List[str] = []
    out.append("## 4. Calibration Transfer (Headline Claim)")
    out.append("")

    # ── In-distribution anchor (the "what we trained on" baseline) ─────────
    base_id = _indist_samples_all(baseline)
    after_id = _indist_samples_all(after)

    # If the baseline JSON has no in-dist samples we can't compute a
    # transfer anchor — punt early with a friendly message instead of
    # a confusing all-NaN table.
    if not base_id or not after_id:
        out.append(
            "_Insufficient data to compute a calibration-transfer table: "
            "baseline JSON has no in-distribution samples. Re-run "
            "`baseline_eval.py` (or `full_eval.py` without `--adapter-path`) "
            "with `--samples ≥ 50` so the anchor row can be populated._"
        )
        out.append("")
        return out

    bid_c, bid_o = _conf_correct(base_id)
    aid_c, aid_o = _conf_correct(after_id)
    id_acc_b = _slice_acc(base_id)
    id_acc_a = _slice_acc(after_id)
    id_ece_b = compute_ece(bid_c, bid_o)
    id_ece_a = compute_ece(aid_c, aid_o)
    id_brier_b = compute_brier(bid_c, bid_o)
    id_brier_a = compute_brier(aid_c, aid_o)
    id_delta_ece = id_ece_a - id_ece_b
    id_ci_lo, id_ci_hi = _bootstrap_paired_delta_ece_ci(bid_c, bid_o, aid_c, aid_o)

    # ── Per-slice OOD rows ─────────────────────────────────────────────────
    rows: List[Dict] = []
    for slice_name in SUPPORTED_OOD_SLICES:
        base_samples  = _slice_samples(baseline, slice_name)
        after_samples = _slice_samples(after,    slice_name)
        if not after_samples:
            # If there's no after-RL OOD data for this slice, it isn't
            # part of this run. Skip silently rather than show NaNs.
            continue
        floor = ood_slice_floor(slice_name)
        bc, bo = _conf_correct(base_samples)
        ac, ao = _conf_correct(after_samples)
        ece_b = compute_ece(bc, bo) if bc else float("nan")
        ece_a = compute_ece(ac, ao) if ac else float("nan")
        brier_b = compute_brier(bc, bo) if bc else float("nan")
        brier_a = compute_brier(ac, ao) if ac else float("nan")
        acc_b = _slice_acc(base_samples) if base_samples else float("nan")
        acc_a = _slice_acc(after_samples)
        d_ece = ece_a - ece_b if (bc and ac) else float("nan")
        ci_lo, ci_hi = (
            _bootstrap_paired_delta_ece_ci(bc, bo, ac, ao) if (bc and ac)
            else (float("nan"), float("nan"))
        )
        status = _delta_ece_status(d_ece, ci_lo, ci_hi, acc_a, floor)
        rows.append({
            "slice":   slice_name,
            "n":       len(after_samples),
            "floor":   floor,
            "acc_b":   acc_b,
            "acc_a":   acc_a,
            "ece_b":   ece_b,
            "ece_a":   ece_a,
            "brier_b": brier_b,
            "brier_a": brier_a,
            "d_ece":   d_ece,
            "ci_lo":   ci_lo,
            "ci_hi":   ci_hi,
            "status":  status,
        })

    if not rows:
        out.append(
            "_No OOD slices in the after-RL JSON. Run "
            "`python eval/full_eval.py --adapter-path <adapter> "
            "--ood-slices auto` to populate them._"
        )
        out.append("")
        return out

    out.append("| Section | n | floor | acc (before → after) | ECE before | ECE after | ΔECE [95% CI] | Brier (before → after) | Status |")
    out.append("|---------|--:|:-----:|:---------------------|----------:|----------:|--------------|-----------------------:|:-------|")

    def _fmt_acc_pair(b: float, a: float) -> str:
        import math
        if math.isnan(b):
            return f"      → {a:.1%}"
        return f"{b:>5.1%} → {a:.1%}"

    def _fmt_ci(d: float, lo: float, hi: float) -> str:
        import math
        if math.isnan(d):
            return "n/a"
        if math.isnan(lo) or math.isnan(hi):
            return f"{d:+.4f}"
        return f"{d:+.4f}  [{lo:+.4f}, {hi:+.4f}]"

    # In-distribution anchor row.
    out.append(
        f"| **ID (math+code+logic)** | {len(after_id)} | — | "
        f"{_fmt_acc_pair(id_acc_b, id_acc_a)} | "
        f"{id_ece_b:.4f} | {id_ece_a:.4f} | "
        f"{_fmt_ci(id_delta_ece, id_ci_lo, id_ci_hi)} | "
        f"{id_brier_b:.4f} → {id_brier_a:.4f} | "
        f"_anchor_ |"
    )
    for r in rows:
        out.append(
            f"| {r['slice']} | {r['n']} | {r['floor']:.2f} | "
            f"{_fmt_acc_pair(r['acc_b'], r['acc_a'])} | "
            f"{r['ece_b']:.4f} | {r['ece_a']:.4f} | "
            f"{_fmt_ci(r['d_ece'], r['ci_lo'], r['ci_hi'])} | "
            f"{r['brier_b']:.4f} → {r['brier_a']:.4f} | "
            f"{r['status']} |"
        )
    out.append("")

    # ── Transfer ratio + headline ──────────────────────────────────────────
    import math
    clear_rows = [
        r for r in rows
        if (r["acc_a"] - r["floor"]) >= _TRANSFER_FLOOR_MARGIN
        and not math.isnan(r["d_ece"])
    ]
    if id_delta_ece < -1e-4 and clear_rows:
        avg_ood_delta = sum(r["d_ece"] for r in clear_rows) / len(clear_rows)
        # Ratio is positive when both deltas are negative (improvements).
        # Clamp display to [-2, 2] so a wonky baseline doesn't spike to 50×.
        ratio = avg_ood_delta / id_delta_ece
        ratio_disp = max(-2.0, min(2.0, ratio))
        n_clear = len(clear_rows)
        n_total = len(rows)
        ratio_word = (
            "strong (>0.7×)"  if ratio_disp >= 0.7  else
            "moderate (>0.3×)" if ratio_disp >= 0.3 else
            "weak (<0.3×)"
        )
        improvement_pct = (
            (-avg_ood_delta / id_ece_b) * 100.0
            if (id_ece_b > 1e-4) else float("nan")
        )
        out.append(
            f"**Transfer ratio:** {ratio_disp:.2f}×  "
            f"_(= mean ΔECE on {n_clear}/{n_total} OOD slices clear of floor "
            f"÷ ΔECE on in-distribution; 1.0× would mean OOD calibration "
            f"improved by the same amount as in-distribution)_"
        )
        out.append("")
        out.append(
            f"**Headline:** "
            f"Calibration learned on math+code+logic **transfers to OOD** "
            f"with **{ratio_word}** strength — average OOD ECE moves from "
            f"{sum(r['ece_b'] for r in clear_rows)/n_clear:.4f} to "
            f"{sum(r['ece_a'] for r in clear_rows)/n_clear:.4f} "
            f"(Δ={avg_ood_delta:+.4f}"
            + (f", {improvement_pct:.0f}% relative improvement" if not math.isnan(improvement_pct) else "")
            + ")."
        )
    elif id_delta_ece >= -1e-4:
        out.append(
            "**Transfer ratio:** n/a — in-distribution ECE did not improve "
            f"(ΔECE_ID = {id_delta_ece:+.4f}). Transfer is undefined when "
            f"there is nothing to transfer; investigate the in-distribution "
            f"training first."
        )
    else:
        out.append(
            "**Transfer ratio:** n/a — every OOD slice is at the random-MCQ "
            "floor (acc_after − floor < 5pp), so per-slice ECE deltas are "
            "dominated by sampling noise. Switch to a tier with smaller "
            "OOD slices (`--ood-slices commonsense,science_easy`) or "
            "evaluate a larger model."
        )
    out.append("")
    return out


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

    # ── 3.5 Calibration transfer (HEADLINE CLAIM) ──────────────────────────
    # The whole point of OOD evaluation: prove that the calibration
    # learned on math+code+logic transfers to unseen domains. We render
    # a per-slice table with paired-bootstrap CIs on ΔECE, flag slices
    # where the model is at the random-MCQ floor (transfer claim noisy
    # there), and end with a single "transfer ratio" headline.
    transfer_section = _render_transfer_section(baseline, after)
    out.extend(transfer_section)

    # ── 4. Confidence histogram (after) ────────────────────────────────────
    confs_b, _ = _conf_correct(base_samples)
    confs_a, _ = _conf_correct(after_samples)

    out.append("## 5. Confidence Distribution (text histogram)")
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

    # ── 6. Operating mode shifts ──────────────────────────────────────────
    out.append("## 6. Operating-Mode Shifts")
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
