"""Tests for Calibration-Aware Self-Refinement (CASR) — server.hindsight_v2.

These tests pin down the contract the training loop relies on:

  1. Parser is forgiving: missing slots → `has_*` flags False, never raises.
  2. Reward is silent (0.0) when no refinement structure is attempted.
  3. ΔBrier term is positive iff the refinement *improved* calibration.
  4. Anti-copy penalty fires for r ≈ c.
  5. Partial-structure penalty fires for half-emitted refinement.
  6. Final reward is clipped to ±clip on pathological inputs.
  7. TRL-compatible factory matches the expected signature and does not
     leak crashes when ground-truth verification fails.
"""

from __future__ import annotations

import pytest

from server.hindsight_v2 import (
    ALPHA_DELTA_BRIER,
    BETA_CRITIQUE_FORMAT,
    DELTA_PARTIAL_STRUCTURE,
    GAMMA_TRIVIAL_COPY,
    REWARD_CLIP,
    compute_refinement_reward,
    make_refinement_reward,
    parse_refinement,
)


# A canonical "happy path" completion: all four slots present, the model
# is wrong but lowered its confidence after critique.
HAPPY = (
    "<reasoning>3 + 4 = 8</reasoning>"
    "<answer>8</answer>"
    "<confidence>0.9</confidence>"
    "<critique>actually 3 + 4 is 7, not 8 — I made an arithmetic slip</critique>"
    "<refined_confidence>0.1</refined_confidence>"
)

# Same shape but the model is right and bumped confidence up after critique.
HAPPY_CORRECT = (
    "<reasoning>3 + 4 = 7</reasoning>"
    "<answer>7</answer>"
    "<confidence>0.6</confidence>"
    "<critique>I rechecked: 3 + 4 = 7, no errors in the arithmetic chain</critique>"
    "<refined_confidence>0.95</refined_confidence>"
)

# Answer-only baseline — no critique structure at all.
BARE = (
    "<reasoning>3 + 4 = 7</reasoning>"
    "<answer>7</answer>"
    "<confidence>0.8</confidence>"
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseRefinement:
    def test_full_structure(self):
        p = parse_refinement(HAPPY)
        assert p.has_answer
        assert p.has_confidence
        assert p.has_critique
        assert p.has_refined_confidence
        assert p.has_full_structure
        assert not p.has_partial_structure
        assert p.confidence == pytest.approx(0.9)
        assert p.refined_confidence == pytest.approx(0.1)

    def test_bare_answer(self):
        p = parse_refinement(BARE)
        assert p.has_answer and p.has_confidence
        assert not p.has_critique
        assert not p.has_refined_confidence
        assert not p.has_partial_structure  # nothing emitted, not partial

    def test_partial_only_critique(self):
        # critique without refined_confidence — partial structure
        text = BARE + "<critique>I think this is fine actually here goes</critique>"
        p = parse_refinement(text)
        assert p.has_critique and not p.has_refined_confidence
        assert p.has_partial_structure
        assert not p.has_full_structure

    def test_partial_only_refined(self):
        text = BARE + "<refined_confidence>0.5</refined_confidence>"
        p = parse_refinement(text)
        assert p.has_refined_confidence and not p.has_critique
        assert p.has_partial_structure

    def test_critique_too_short_is_not_critique(self):
        # Below MIN_CRITIQUE_CHARS we treat the slot as empty
        text = BARE + "<critique>oops</critique><refined_confidence>0.5</refined_confidence>"
        p = parse_refinement(text)
        assert not p.has_critique
        assert p.has_refined_confidence
        # → counts as partial structure, since refined alone was emitted
        assert p.has_partial_structure

    def test_refined_out_of_range_is_unparseable(self):
        text = BARE + (
            "<critique>this needs more thought, definitely</critique>"
            "<refined_confidence>1.5</refined_confidence>"
        )
        p = parse_refinement(text)
        assert p.has_critique
        assert not p.has_refined_confidence

    def test_empty_or_none_input(self):
        for raw in [None, "", "   "]:
            p = parse_refinement(raw)
            assert not p.has_answer
            assert not p.has_critique
            assert not p.has_refined_confidence

    def test_case_insensitive_tags(self):
        text = (
            "<reasoning>r</reasoning><answer>7</answer><confidence>0.6</confidence>"
            "<CRITIQUE>I should double check this answer carefully</CRITIQUE>"
            "<Refined_Confidence>0.7</Refined_Confidence>"
        )
        p = parse_refinement(text)
        assert p.has_critique
        assert p.has_refined_confidence


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class TestComputeRefinementReward:
    def test_silent_zero_for_bare_answer(self):
        p = parse_refinement(BARE)
        assert compute_refinement_reward(p, correctness=True) == 0.0
        assert compute_refinement_reward(p, correctness=False) == 0.0
        assert compute_refinement_reward(p, correctness=None) == 0.0

    def test_refining_wrong_answer_down_is_rewarded(self):
        # c=0.9 → r=0.1 on a wrong answer:
        # ΔBrier = (0.9 - 0)^2 - (0.1 - 0)^2 = 0.81 - 0.01 = 0.80
        # + β (critique format) - 0 (not a copy) = 1.0 * 0.80 + 0.05 = 0.85
        # → clipped to REWARD_CLIP = 0.30
        p = parse_refinement(HAPPY)
        r = compute_refinement_reward(p, correctness=False)
        assert r == pytest.approx(REWARD_CLIP)  # clip dominates

    def test_refining_correct_answer_up_is_rewarded(self):
        # c=0.6 → r=0.95 on correct answer:
        # ΔBrier = (0.6 - 1)^2 - (0.95 - 1)^2 = 0.16 - 0.0025 = 0.1575
        # + 0.05 critique - 0 (not a copy) = 0.2075
        p = parse_refinement(HAPPY_CORRECT)
        r = compute_refinement_reward(p, correctness=True)
        assert r > 0
        assert r == pytest.approx(0.1575 * ALPHA_DELTA_BRIER + BETA_CRITIQUE_FORMAT, abs=1e-6)

    def test_refining_in_wrong_direction_is_punished(self):
        # Model is wrong (y=0) but raised confidence after critiquing:
        # c=0.1 → r=0.9 on an incorrect answer.
        # ΔBrier = (0.1 - 0)^2 - (0.9 - 0)^2 = 0.01 - 0.81 = -0.80 (worsened)
        # + β = -0.80 + 0.05 = -0.75 → clipped to -REWARD_CLIP
        text = (
            "<reasoning>3+4=8</reasoning><answer>8</answer><confidence>0.1</confidence>"
            "<critique>actually I am very sure this is right after all on review</critique>"
            "<refined_confidence>0.9</refined_confidence>"
        )
        p = parse_refinement(text)
        r = compute_refinement_reward(p, correctness=False)
        assert r == pytest.approx(-REWARD_CLIP)

    def test_trivial_copy_penalty_fires(self):
        # c = r = 0.7 → ΔBrier = 0 + β - γ = 0
        text = (
            "<reasoning>r</reasoning><answer>7</answer><confidence>0.7</confidence>"
            "<critique>looks fine I think the answer holds up nicely</critique>"
            "<refined_confidence>0.7</refined_confidence>"
        )
        p = parse_refinement(text)
        r = compute_refinement_reward(p, correctness=True)
        assert r == pytest.approx(BETA_CRITIQUE_FORMAT - GAMMA_TRIVIAL_COPY, abs=1e-6)
        assert abs(r) < 1e-6  # net zero given equal β and γ

    def test_partial_structure_is_penalised(self):
        # Critique present but no refined_confidence
        text = BARE + "<critique>this answer feels reasonable enough to me</critique>"
        p = parse_refinement(text)
        r = compute_refinement_reward(p, correctness=True)
        # +β format - δ partial = 0.05 - 0.05 = 0
        assert r == pytest.approx(BETA_CRITIQUE_FORMAT - DELTA_PARTIAL_STRUCTURE, abs=1e-6)

    def test_refined_only_with_partial_no_format_bonus(self):
        # refined_confidence present but no critique → partial penalty,
        # no critique-format bonus, ΔBrier missing because we have c and r
        # but the spec says we still need critique to count as full
        # structure. ΔBrier IS computable here though — confidence and
        # refined are both present — so we DO add it. Critically, the
        # partial penalty still fires.
        text = (
            "<reasoning>r</reasoning><answer>7</answer><confidence>0.6</confidence>"
            "<refined_confidence>0.9</refined_confidence>"
        )
        p = parse_refinement(text)
        r = compute_refinement_reward(p, correctness=True)
        # ΔBrier = (0.6-1)^2 - (0.9-1)^2 = 0.16 - 0.01 = 0.15
        # - δ partial = 0.15 - 0.05 = 0.10
        assert r == pytest.approx(0.15 * ALPHA_DELTA_BRIER - DELTA_PARTIAL_STRUCTURE, abs=1e-6)

    def test_correctness_none_skips_delta_brier(self):
        # When correctness can't be determined (e.g. abstain or unverifiable),
        # ΔBrier is silent but format bonuses / penalties still apply.
        p = parse_refinement(HAPPY)
        r = compute_refinement_reward(p, correctness=None)
        # Only β fires (not a copy, has critique, has refined, full structure)
        assert r == pytest.approx(BETA_CRITIQUE_FORMAT, abs=1e-6)

    def test_clip_bounds_extreme_inputs(self):
        # Refining from c=0 to r=1 on a wrong answer (worst case)
        text = (
            "<reasoning>r</reasoning><answer>X</answer><confidence>0.0</confidence>"
            "<critique>nope this is definitely correct on reflection</critique>"
            "<refined_confidence>1.0</refined_confidence>"
        )
        p = parse_refinement(text)
        r = compute_refinement_reward(p, correctness=False)
        assert r == pytest.approx(-REWARD_CLIP)
        # Symmetric on the other extreme:
        text2 = (
            "<reasoning>r</reasoning><answer>X</answer><confidence>1.0</confidence>"
            "<critique>nope this is definitely wrong on reflection</critique>"
            "<refined_confidence>0.0</refined_confidence>"
        )
        p2 = parse_refinement(text2)
        r2 = compute_refinement_reward(p2, correctness=False)
        assert r2 == pytest.approx(REWARD_CLIP)


# ---------------------------------------------------------------------------
# TRL reward function factory
# ---------------------------------------------------------------------------


class TestMakeRefinementReward:
    def test_silent_for_bare_answers(self):
        fn = make_refinement_reward(weight=1.0)
        out = fn([BARE, BARE], prompts=["x", "x"], ground_truth=["7", "7"])
        assert out == [0.0, 0.0]

    def test_full_structure_emits_signal(self):
        fn = make_refinement_reward(weight=1.0)
        out = fn(
            [HAPPY_CORRECT],
            prompts=["x"],
            ground_truth=["7"],
            domain=["math"],
            problem_id=["test"],
        )
        assert out[0] > 0

    def test_weight_scales_output(self):
        fn1 = make_refinement_reward(weight=1.0)
        fn2 = make_refinement_reward(weight=2.0)
        out1 = fn1([HAPPY_CORRECT], prompts=["x"], ground_truth=["7"])
        out2 = fn2([HAPPY_CORRECT], prompts=["x"], ground_truth=["7"])
        assert out2[0] == pytest.approx(2 * out1[0])

    def test_function_name_includes_weight(self):
        fn = make_refinement_reward(weight=0.7)
        assert "reward_refinement" in fn.__name__
        assert "0.7" in fn.__name__

    def test_handles_missing_ground_truth_gracefully(self):
        # If gt is empty/missing the verifier may return False; the head
        # must still produce a finite reward, never raise.
        fn = make_refinement_reward(weight=1.0)
        out = fn([HAPPY], prompts=[""], ground_truth=[""])
        assert isinstance(out, list) and len(out) == 1
        assert -REWARD_CLIP <= out[0] <= REWARD_CLIP

    def test_mixed_batch(self):
        fn = make_refinement_reward(weight=1.0)
        completions = [BARE, HAPPY, HAPPY_CORRECT]
        out = fn(
            completions,
            prompts=["x"] * 3,
            ground_truth=["7", "7", "7"],
            domain=["math"] * 3,
            problem_id=["p"] * 3,
        )
        assert len(out) == 3
        assert out[0] == 0.0  # bare → silent
        assert -REWARD_CLIP <= out[1] <= REWARD_CLIP
        assert -REWARD_CLIP <= out[2] <= REWARD_CLIP
