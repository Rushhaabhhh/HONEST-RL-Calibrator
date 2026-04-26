"""training/format_sft.py — DEPRECATED shim.

The original ``format_sft.py`` was a Qwen-3B-only script that taught the
model the 3-tag XML format. It has been superseded by the model-agnostic
``training/calibration_sft.py``, which:

* works for every supported preset (Qwen-0.5B / 1.5B / 3B, Llama-1B / 3B,
  Phi-4-mini), not just Qwen-3B;
* falls back cleanly to HF transformers + bnb when Unsloth isn't
  available or when running on CPU for ``--dry-run``;
* teaches three priors per example (format, correctness-conditioned
  confidence, and the optional ``<hindsight>`` tag) so the resulting
  adapter unlocks the legacy hindsight reward channel during GRPO.

This shim preserves the historical entry point for any external scripts
that called ``python training/format_sft.py`` directly. New users should
instead invoke::

    python training/calibration_sft.py \\
        --model-id Qwen/Qwen2.5-3B-Instruct \\
        --output-dir ./training/format_sft_adapters

or use the one-command orchestrator::

    ./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


_LEGACY_DEFAULTS = [
    "--model-id",  "Qwen/Qwen2.5-3B-Instruct",
    "--output-dir", str(PROJECT_ROOT / "training" / "format_sft_adapters"),
]


def main() -> None:
    print(
        "[format_sft] DEPRECATED — forwarding to training/calibration_sft.py.\n"
        "             See docs/SELF_LEARNING.md §2.6 for the new recipe.\n",
        flush=True,
    )

    # Forward any user-supplied args (which take precedence over the legacy
    # defaults) to the new CLI.
    user_args = list(sys.argv[1:])
    if not any(a == "--model-id" for a in user_args):
        user_args = _LEGACY_DEFAULTS + user_args

    sys.argv = [sys.argv[0], *user_args]

    from training.calibration_sft import main as _new_main
    _new_main()


if __name__ == "__main__":
    main()
