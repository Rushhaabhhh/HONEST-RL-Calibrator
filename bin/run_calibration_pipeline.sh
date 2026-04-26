#!/usr/bin/env bash
#
# bin/run_calibration_pipeline.sh — one-command SFT-then-GRPO recipe.
#
# Why this script exists
# ----------------------
# Tiny models (Qwen-0.5B, Llama-1B) cannot reliably emit the strict 3-tag
# XML format from the system prompt alone. Skipping the SFT phase wastes
# the entire GRPO compute budget on the malformed-penalty floor — every
# rollout returns the same -1.0 reward, the GRPO advantage signal collapses
# to zero, and the model never learns calibration. The fix is a short
# format+calibration SFT pass before the RL phase.
#
# This script chains the two phases together so a tiny-model run is one
# command. For small/medium tiers the SFT phase is optional but still
# helpful (it activates the legacy hindsight head and accelerates early
# convergence by ~15%).
#
# Usage
# -----
#   ./bin/run_calibration_pipeline.sh <model-id> [--skip-sft] [extra GRPO args...]
#
# Examples:
#   # Tiny tier — full pipeline (SFT then GRPO with legacy hindsight)
#   ./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct
#   ./bin/run_calibration_pipeline.sh meta-llama/Llama-3.2-1B-Instruct
#
#   # Medium tier — uses CASR by default
#   ./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-3B-Instruct
#
#   # Skip SFT (only sensible for medium tier or research baselines)
#   ./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-3B-Instruct --skip-sft
#
#   # Pass extra GRPO args (everything after the model-id is forwarded)
#   ./bin/run_calibration_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct --max-steps 100
#
# Output
# ------
#   ./sft-<slug>/                      LoRA from the SFT phase
#   ./honest-<slug>-grpo/              Working dir for the GRPO phase
#   ./honest-<slug>-grpo/final_adapters  Final LoRA + controller state

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-id> [--skip-sft] [extra GRPO args...]" >&2
    exit 2
fi

MODEL_ID="$1"; shift

SKIP_SFT=0
EXTRA_GRPO_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-sft" ]]; then
        SKIP_SFT=1
    else
        EXTRA_GRPO_ARGS+=("$arg")
    fi
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Slug for output directories — derived from model-id last path component.
SLUG=$(echo "$MODEL_ID" | awk -F'/' '{print tolower($NF)}' | tr '.' '-')

SFT_DIR="${SFT_DIR:-./sft-${SLUG}}"
GRPO_DIR="${GRPO_DIR:-./honest-${SLUG}-grpo}"

PYTHON="${PYTHON:-python}"

echo "============================================================"
echo "Calibration pipeline"
echo "  model:       ${MODEL_ID}"
echo "  sft_dir:     ${SFT_DIR}"
echo "  grpo_dir:    ${GRPO_DIR}"
echo "  skip_sft:    ${SKIP_SFT}"
echo "  extra args:  ${EXTRA_GRPO_ARGS[*]:-}"
echo "============================================================"

# Resolve the tier-aware default --hindsight-mode for this preset. The
# Python helper returns "legacy" for tiny models and "refined" for the
# rest; users can override via EXTRA_GRPO_ARGS.
DEFAULT_HINDSIGHT_MODE=$("$PYTHON" -c "
from calibration_profiles import recommend_hindsight_mode, get_preset
p = get_preset('${MODEL_ID}', 'auto')
print(recommend_hindsight_mode(p.name))
")
TIER=$("$PYTHON" -c "
from calibration_profiles import get_preset
print(get_preset('${MODEL_ID}', 'auto').tier)
")
echo "  preset_tier: ${TIER}"
echo "  hindsight:   ${DEFAULT_HINDSIGHT_MODE} (override with --hindsight-mode ...)"
echo "============================================================"

# ── Phase 1: SFT ─────────────────────────────────────────────────────────
if [[ ${SKIP_SFT} -eq 0 ]]; then
    echo "[Phase 1/2] Calibration SFT → ${SFT_DIR}"
    "$PYTHON" training/calibration_sft.py \
        --model-id "${MODEL_ID}" \
        --output-dir "${SFT_DIR}"
    INIT_ADAPTER_FLAG=(--init-adapter "${SFT_DIR}")
else
    echo "[Phase 1/2] SKIPPED (--skip-sft requested)"
    INIT_ADAPTER_FLAG=()
fi

# ── Phase 2: GRPO ────────────────────────────────────────────────────────
echo "[Phase 2/2] GRPO → ${GRPO_DIR}"

# If the user did not explicitly provide --hindsight-mode in extras, append
# the tier-appropriate default. We always pass --hindsight (the head is
# silent on its own when the format is missing, and it activates the
# diagnostic instrumentation in bin/audit_hindsight.py post-run).
USER_OVERRIDE_HS_MODE=0
for arg in "${EXTRA_GRPO_ARGS[@]:-}"; do
    if [[ "$arg" == "--hindsight-mode" ]]; then
        USER_OVERRIDE_HS_MODE=1
    fi
done

HS_FLAGS=(--hindsight)
if [[ ${USER_OVERRIDE_HS_MODE} -eq 0 ]]; then
    HS_FLAGS+=(--hindsight-mode "${DEFAULT_HINDSIGHT_MODE}")
fi

"$PYTHON" training/train_grpo.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${GRPO_DIR}" \
    "${INIT_ADAPTER_FLAG[@]:-}" \
    "${HS_FLAGS[@]}" \
    "${EXTRA_GRPO_ARGS[@]:-}"

echo "============================================================"
echo "Pipeline complete."
echo "  SFT adapter:           ${SFT_DIR}"
echo "  GRPO final adapter:    ${GRPO_DIR}/final_adapters"
echo ""
echo "Next steps:"
echo "  bin/audit_hindsight.py ${GRPO_DIR}/trainer_state.json"
echo "  bin/plot_training_curves.py ${GRPO_DIR}/trainer_state.json --out-dir ${GRPO_DIR}/plots"
echo "============================================================"
