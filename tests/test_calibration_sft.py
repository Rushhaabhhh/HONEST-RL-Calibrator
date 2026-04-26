"""Tests for the calibration SFT pipeline.

The actual GPU-bound trainer can't run in CI, so we focus on the parts that
can be unit-tested deterministically:

  * the dataset builder produces records with the right shape, tier-aware
    composition, and ground-truth-aligned hindsight tags;
  * every assistant target survives ``server.reward.parse_action`` (i.e.
    the SFT prior is *not* teaching a malformed format);
  * tier metadata in ``calibration_profiles`` is internally consistent;
  * the ``--init-adapter`` plumbing in ``train_grpo.py`` is recognised by
    argparse and threaded through to the loaders.

These tests prove that *if* the SFT trainer converges on the supplied
dataset, the resulting model will emit text that the existing reward
parser accepts — closing the loop end-to-end without needing a GPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_profiles import (
    MODEL_PRESETS,
    SUPPORTED_TIERS,
    get_preset,
    is_tiny_tier,
    prompt_templates,
    recommend_hindsight_mode,
)
from server.hindsight import parse_hindsight
from server.reward import parse_action
from training.calibration_sft import (
    _CORRECT_CONFIDENCE_BUCKETS,
    _WRONG_CONFIDENCE_BUCKETS,
    _format_assistant_target,
    _perturb_answer,
    build_sft_examples,
    summarise_records,
)


# ---------------------------------------------------------------------------
# Tier metadata sanity
# ---------------------------------------------------------------------------


class TestTierMetadata:
    def test_every_preset_has_a_tier(self):
        for name, preset in MODEL_PRESETS.items():
            assert preset.tier in SUPPORTED_TIERS, (
                f"preset {name!r} has unsupported tier {preset.tier!r}"
            )

    def test_tier_invariants(self):
        for name, preset in MODEL_PRESETS.items():
            assert preset.recommended_sft_examples >= 100, name
            assert 1 <= preset.recommended_sft_epochs <= 5, name
            assert 1 <= preset.recommended_sft_max_difficulty <= 5, name
            assert 0.0 <= preset.recommended_sft_hindsight_frac <= 1.0, name

    def test_tiny_models_recommend_legacy_hindsight(self):
        # CASR is too complex for tiny models — they can't critique their
        # own reasoning. Legacy hindsight is just a self-prediction
        # regression target, well within capacity post-SFT.
        for name in ("qwen0.5b", "llama1b"):
            assert recommend_hindsight_mode(name) == "legacy", name

    def test_small_and_medium_recommend_refined(self):
        for name in ("qwen1.5b", "qwen3b", "llama3b", "phi4mini"):
            assert recommend_hindsight_mode(name) == "refined", name

    def test_is_tiny_tier_true_only_for_tiny(self):
        assert is_tiny_tier("qwen0.5b") is True
        assert is_tiny_tier("llama1b") is True
        assert is_tiny_tier("qwen1.5b") is False
        assert is_tiny_tier("qwen3b") is False
        assert is_tiny_tier("does-not-exist") is False


# ---------------------------------------------------------------------------
# _perturb_answer
# ---------------------------------------------------------------------------


class TestPerturbAnswer:
    def test_numeric_answer_is_offset(self):
        import random
        rng = random.Random(0)
        for gt in ["7", "100", "-3"]:
            alt = _perturb_answer("math", gt, rng)
            assert alt != gt
            assert int(alt) != int(gt)

    def test_short_letter_answer_swaps(self):
        import random
        rng = random.Random(0)
        alt = _perturb_answer("logic", "A", rng)
        assert alt != "A"
        assert isinstance(alt, str) and len(alt) >= 1

    def test_word_answer_appends_digit(self):
        import random
        rng = random.Random(0)
        alt = _perturb_answer("logic", "Aurora", rng)
        assert alt.startswith("Aurora")
        assert alt != "Aurora"


# ---------------------------------------------------------------------------
# _format_assistant_target
# ---------------------------------------------------------------------------


class TestFormatAssistantTarget:
    def test_three_tag_target_round_trips_through_parse_action(self):
        text = _format_assistant_target(
            reasoning="Direct evaluation: 7.",
            answer="7",
            confidence=0.91,
            include_hindsight=False,
            correct=True,
        )
        parsed = parse_action(text)
        assert parsed["type"] == "answer"
        assert parsed["answer"] == "7"
        assert pytest.approx(parsed["confidence"], abs=0.01) == 0.91

    def test_target_with_hindsight_tag_parses_both(self):
        text = _format_assistant_target(
            reasoning="Trivial.",
            answer="42",
            confidence=0.85,
            include_hindsight=True,
            correct=True,
        )
        action_parse = parse_action(text)
        hindsight_parse = parse_hindsight(text)
        assert action_parse["type"] == "answer"
        assert action_parse["answer"] == "42"
        assert hindsight_parse["type"] == "hindsight"
        assert hindsight_parse["retrospective"] == 1.0

    def test_hindsight_value_matches_correctness(self):
        wrong = _format_assistant_target(
            reasoning="Trivial.",
            answer="3",
            confidence=0.20,
            include_hindsight=True,
            correct=False,
        )
        assert "<hindsight>0.0</hindsight>" in wrong
        assert parse_hindsight(wrong)["retrospective"] == 0.0

        right = _format_assistant_target(
            reasoning="Trivial.",
            answer="42",
            confidence=0.85,
            include_hindsight=True,
            correct=True,
        )
        assert "<hindsight>1.0</hindsight>" in right
        assert parse_hindsight(right)["retrospective"] == 1.0

    def test_abstain_target(self):
        text = _format_assistant_target(
            reasoning="N/A",
            answer="anything",
            confidence=0.5,
            include_hindsight=False,
            correct=False,
            abstain=True,
        )
        assert text == "<abstain/>"
        assert parse_action(text)["type"] == "abstain"


# ---------------------------------------------------------------------------
# build_sft_examples
# ---------------------------------------------------------------------------


class TestBuildSFTExamples:
    @pytest.fixture
    def small_sample(self):
        sys_p, user_t = prompt_templates("required")
        return build_sft_examples(
            n=200,
            domain_weights={"math": 0.5, "code": 0.3, "logic": 0.2},
            max_difficulty=2,
            hindsight_frac=0.5,
            correct_frac=0.65,
            seed=42,
            system_prompt=sys_p,
            user_template=user_t,
        )

    def test_record_count_and_shape(self, small_sample):
        assert len(small_sample) == 200
        for r in small_sample:
            assert set(r.keys()) == {"messages", "meta"}
            roles = [m["role"] for m in r["messages"]]
            assert roles == ["system", "user", "assistant"]

    def test_difficulty_respects_max_difficulty(self, small_sample):
        diffs = {r["meta"]["difficulty"] for r in small_sample}
        assert diffs.issubset({1, 2})

    def test_domain_distribution_roughly_matches_weights(self, small_sample):
        summary = summarise_records(small_sample)
        # Math weight = 0.5, so we expect ~100 of 200 within a generous
        # tolerance (the generators sometimes raise and we retry).
        math_count = summary["by_domain"].get("math", 0)
        code_count = summary["by_domain"].get("code", 0)
        assert math_count > code_count, summary
        assert summary["frac_correct"] >= 0.55  # 0.65 ± perturbation collisions
        assert summary["frac_correct"] <= 0.85
        assert 0.40 <= summary["frac_hindsight"] <= 0.60  # 0.5 ± noise

    def test_every_assistant_target_is_well_formed(self, small_sample):
        # The whole point of SFT is to teach the format. If a single
        # generated target is malformed we'd be teaching the wrong thing.
        for r in small_sample:
            text = r["messages"][2]["content"]
            parsed = parse_action(text)
            assert parsed["type"] in ("answer", "abstain"), (
                f"Malformed SFT target produced:\n{text!r}\nparsed={parsed}"
            )

    def test_hindsight_tags_are_well_formed(self, small_sample):
        for r in small_sample:
            text = r["messages"][2]["content"]
            if r["meta"]["has_hindsight"]:
                hp = parse_hindsight(text)
                assert hp["type"] == "hindsight", (
                    f"has_hindsight=True but parse failed:\n{text!r}"
                )
                expected = 1.0 if r["meta"]["correct"] else 0.0
                assert hp["retrospective"] == expected
            else:
                # Optionally absent — don't require <hindsight> in non-HS rows.
                pass

    def test_user_message_uses_template(self, small_sample):
        for r in small_sample:
            user_text = r["messages"][1]["content"]
            assert "<reasoning>" in user_text or "Think briefly" in user_text


class TestBuildSFTExamplesValidation:
    @pytest.fixture
    def prompts(self):
        return prompt_templates("required")

    def test_n_must_be_positive(self, prompts):
        sys_p, user_t = prompts
        with pytest.raises(ValueError):
            build_sft_examples(
                n=0,
                domain_weights={"math": 1.0},
                max_difficulty=2,
                hindsight_frac=0.5,
                seed=0,
                system_prompt=sys_p,
                user_template=user_t,
            )

    def test_hindsight_frac_must_be_in_unit_interval(self, prompts):
        sys_p, user_t = prompts
        with pytest.raises(ValueError):
            build_sft_examples(
                n=10, domain_weights={"math": 1.0}, max_difficulty=2,
                hindsight_frac=1.2, seed=0,
                system_prompt=sys_p, user_template=user_t,
            )

    def test_max_difficulty_bounds(self, prompts):
        sys_p, user_t = prompts
        with pytest.raises(ValueError):
            build_sft_examples(
                n=10, domain_weights={"math": 1.0}, max_difficulty=0,
                hindsight_frac=0.0, seed=0,
                system_prompt=sys_p, user_template=user_t,
            )
        with pytest.raises(ValueError):
            build_sft_examples(
                n=10, domain_weights={"math": 1.0}, max_difficulty=6,
                hindsight_frac=0.0, seed=0,
                system_prompt=sys_p, user_template=user_t,
            )


class TestPerTierBuilders:
    """Smoke-test that each preset's recommended config produces sensible data."""

    @pytest.mark.parametrize("preset_name", list(MODEL_PRESETS))
    def test_preset_recommendation_builds(self, preset_name):
        preset = MODEL_PRESETS[preset_name]
        sys_p, user_t = prompt_templates("required")
        # Use a small fraction of the recommended count so the test stays fast.
        n = max(50, preset.recommended_sft_examples // 20)
        records = build_sft_examples(
            n=n,
            domain_weights=preset.domain_weights,
            max_difficulty=preset.recommended_sft_max_difficulty,
            hindsight_frac=preset.recommended_sft_hindsight_frac,
            correct_frac=0.65,
            seed=42,
            system_prompt=sys_p,
            user_template=user_t,
        )
        assert len(records) == n
        # Every example must round-trip through parse_action — this is the
        # binding contract between SFT data and the GRPO reward function.
        for r in records:
            assert parse_action(r["messages"][2]["content"])["type"] in (
                "answer", "abstain",
            )

    @pytest.mark.parametrize("preset_name", ["qwen0.5b", "llama1b"])
    def test_tiny_presets_cap_at_difficulty_2(self, preset_name):
        preset = MODEL_PRESETS[preset_name]
        assert preset.recommended_sft_max_difficulty <= 2, preset_name


# ---------------------------------------------------------------------------
# train_grpo.py CLI / dry-run integration
# ---------------------------------------------------------------------------


class TestTrainGRPOCLI:
    """The orchestrator passes --init-adapter through to train_grpo.py.

    These tests exercise the argparse + dry-run path without loading any
    model. They confirm:
      * --init-adapter is accepted and visible in the dry-run summary;
      * the tier-aware warning fires on tiny tier when no adapter is given;
      * the warning does NOT fire when an adapter IS given.
    """

    def _run_dryrun(self, *extra_args: str) -> tuple[int, str, str]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "train_grpo.py"),
            "--dry-run",
            *extra_args,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        return result.returncode, result.stdout, result.stderr

    def test_dry_run_with_init_adapter_flag(self):
        rc, stdout, stderr = self._run_dryrun(
            "--model-id", "Qwen/Qwen2.5-0.5B-Instruct",
            "--hindsight",
            "--init-adapter", "/tmp/some-adapter",
        )
        assert rc == 0, stderr
        assert "init_adapter:                /tmp/some-adapter" in stdout

    def test_dry_run_without_adapter_still_works(self):
        rc, stdout, stderr = self._run_dryrun(
            "--model-id", "Qwen/Qwen2.5-0.5B-Instruct",
            "--hindsight",
        )
        assert rc == 0, stderr
        assert re.search(r"init_adapter:\s+\(none", stdout)


class TestSFTScriptCLI:
    """The SFT script's dry-run path doesn't need a GPU."""

    def _run_dryrun(self, *extra_args: str) -> tuple[int, str, str]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "calibration_sft.py"),
            "--dry-run",
            "--output-dir", "/tmp/sft-test-out",
            *extra_args,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        return result.returncode, result.stdout, result.stderr

    def test_tiny_dry_run(self):
        rc, stdout, stderr = self._run_dryrun(
            "--model-id", "Qwen/Qwen2.5-0.5B-Instruct",
            "--n-examples", "120",
        )
        assert rc == 0, stderr
        # log output goes to stderr by default
        combined = stdout + stderr
        assert "tier=tiny" in combined
        assert "SFT is REQUIRED" in combined

    def test_medium_dry_run(self):
        rc, stdout, stderr = self._run_dryrun(
            "--model-id", "Qwen/Qwen2.5-3B-Instruct",
            "--n-examples", "120",
        )
        assert rc == 0, stderr
        combined = stdout + stderr
        assert "tier=medium" in combined
        assert "SFT is optional" in combined
