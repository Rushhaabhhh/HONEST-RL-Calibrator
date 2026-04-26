#!/usr/bin/env python
"""Diagnose whether the Hindsight Calibration Reward (HCR) is actually firing.

Reads a TRL ``trainer_state.json`` and reports, for the
``reward_hindsight_train_*`` reward channel:

  * Total recorded steps.
  * Number / fraction of steps where the reward was *exactly* 0.0
    (= no rollout in that step's group emitted a parseable <hindsight> tag).
  * Number / fraction of steps where it was non-zero (= signal entered
    the gradient).
  * The min / max / mean of the non-zero reward window — useful for
    sanity-checking the magnitude relative to the primary Brier reward.

Why this matters
----------------

The ``server.hindsight`` module returns 0.0 for any completion that
does not contain a parseable ``<hindsight>`` block. The base model has
no prior to ever emit that tag unless the system prompt explicitly
describes it. If the prompt template never mentions <hindsight>
(see ``calibration_profiles.prompt_templates``), the reward channel
is structurally silent: the gradient through that head is exactly zero
at every step, and the auxiliary head is doing nothing.

This script gives you a one-line answer: "Did hindsight contribute any
signal in this run?" so you can decide whether to (a) accept it as a
non-functional control and document the finding, or (b) re-run with
the patched ``--hindsight-mode refined`` path.

Usage
-----

    python bin/audit_hindsight.py \\
        --trainer-state ./honest-qwen-1-5b-grpo/trainer_state.json

    # JSON output for embedding in a writeup table:
    python bin/audit_hindsight.py \\
        --trainer-state runs/qwen0.5b/trainer_state.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_trainer_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        sys.exit(f"trainer_state.json not found at {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"Failed to parse {path}: {e}")


def _find_hindsight_key(log_history: List[Dict[str, Any]]) -> Optional[str]:
    """Find the reward channel name TRL recorded for the hindsight head.

    TRL names per-reward-function logs as ``rewards/<func.__name__>/mean``,
    where ``__name__`` is whatever the reward function set (we use
    ``reward_hindsight_train_x{weight:g}`` in
    ``training.train_grpo.make_train_time_hindsight_reward``).
    """
    for entry in log_history:
        for k in entry:
            if "reward_hindsight" in k and k.endswith("/mean"):
                return k
    return None


def _summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    nz = [v for v in values if abs(v) > 1e-9]
    return {
        "n_steps":        len(values),
        "n_zero":         len(values) - len(nz),
        "n_nonzero":      len(nz),
        "frac_zero":      (len(values) - len(nz)) / len(values),
        "frac_nonzero":   len(nz) / len(values),
        "min":            min(values),
        "max":            max(values),
        "mean":           sum(values) / len(values),
        "nonzero_min":    min(nz) if nz else 0.0,
        "nonzero_max":    max(nz) if nz else 0.0,
        "nonzero_mean":   (sum(nz) / len(nz)) if nz else 0.0,
    }


def _ascii_bar(frac: float, width: int = 30) -> str:
    fill = int(round(frac * width))
    return "[" + "█" * fill + "·" * (width - fill) + f"] {frac * 100:5.1f}%"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--trainer-state", required=True, type=Path)
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human report.")
    args = p.parse_args()

    state = _load_trainer_state(args.trainer_state)
    history = state.get("log_history") or []
    if not history:
        sys.exit("trainer_state has no log_history (run hasn't logged anything yet).")

    key = _find_hindsight_key(history)
    if not key:
        msg = (
            "No hindsight reward channel found in log_history.\n"
            "  → Either the run was launched WITHOUT --hindsight, or the run\n"
            "    is older than the hindsight head. Re-run with --hindsight to enable it."
        )
        if args.json:
            print(json.dumps({"status": "no_hindsight_channel", "message": msg}))
        else:
            print(msg)
        return 1

    values = [float(e[key]) for e in history if key in e]
    summary = _summarise(values)
    summary["channel"] = key
    summary["trainer_state"] = str(args.trainer_state)

    # Lookup for context: what was the primary brier reward magnitude?
    brier_key = next(
        (k for entry in history for k in entry
         if "reward_brier" in k and k.endswith("/mean")),
        None,
    )
    if brier_key:
        brier_vals = [float(e[brier_key]) for e in history if brier_key in e]
        if brier_vals:
            summary["brier_mean"] = sum(brier_vals) / len(brier_vals)
            summary["hindsight_to_brier_ratio"] = (
                abs(summary["nonzero_mean"]) / max(abs(summary["brier_mean"]), 1e-9)
                if summary["n_nonzero"] > 0 else 0.0
            )

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0 if summary["n_nonzero"] > 0 else 2

    # Human report.
    print("=" * 60)
    print("Hindsight Calibration Reward — Audit")
    print("=" * 60)
    print(f"  trainer_state : {args.trainer_state}")
    print(f"  channel       : {key}")
    print(f"  total steps   : {summary['n_steps']}")
    print()
    print(f"  Zero reward (hindsight tag missing or unparseable):")
    print(f"    {_ascii_bar(summary['frac_zero'])}  ({summary['n_zero']} / {summary['n_steps']})")
    print(f"  Non-zero reward (hindsight head fired):")
    print(f"    {_ascii_bar(summary['frac_nonzero'])}  ({summary['n_nonzero']} / {summary['n_steps']})")
    print()
    print(f"  Reward magnitude (overall)  : "
          f"min={summary['min']:.4f}  max={summary['max']:.4f}  mean={summary['mean']:.4f}")
    if summary["n_nonzero"] > 0:
        print(f"  Reward magnitude (non-zero) : "
              f"min={summary['nonzero_min']:.4f}  max={summary['nonzero_max']:.4f}  "
              f"mean={summary['nonzero_mean']:.4f}")
        if "brier_mean" in summary:
            print(f"  Hindsight / Brier ratio     : "
                  f"{summary['hindsight_to_brier_ratio']:.3f}  "
                  f"({'sufficient' if summary['hindsight_to_brier_ratio'] > 0.05 else 'too weak'})")
    print()
    print("─" * 60)

    if summary["frac_zero"] > 0.95:
        print("VERDICT: hindsight is structurally silent.")
        print()
        print("  Likely cause: the system prompt does not describe the")
        print("  <hindsight> tag, so the model never emits it.")
        print()
        print("  Fix: launch the next run with `--hindsight-mode refined`,")
        print("       which switches to the Calibration-Aware Self-Refinement")
        print("       protocol. See docs/SELF_LEARNING.md §2.5 for the design.")
        return 2
    if summary["frac_zero"] > 0.5:
        print("VERDICT: hindsight is firing intermittently.")
        print(f"  Only {summary['frac_nonzero'] * 100:.1f}% of steps see signal.")
        print(f"  Consider raising --hindsight-weight or switching to refined mode.")
        return 0
    print("VERDICT: hindsight is firing on most steps. Healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
