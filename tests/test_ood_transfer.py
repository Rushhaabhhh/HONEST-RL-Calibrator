"""Tests for the OOD calibration-transfer pipeline.

Covers the three pieces that together make the headline transfer claim
provable for small models:

  1. ``calibration_profiles`` slice registry, tier-aware defaults, and
     filename/floor lookups.
  2. ``eval/full_eval._discover_ood_slices`` — auto-discovery of JSONL
     files, plus filtering / missing-file handling for explicit
     ``--ood-slices`` requests.
  3. ``eval/compare_runs._render_transfer_section`` — the markdown
     calibration-transfer table + headline, including all three
     status branches (transferred / partial / at-floor) and the
     ``ΔECE_ID ≥ 0`` short-circuit.

The tests synthesize per-sample records rather than running a model,
so they're fast (<1 s end-to-end) and don't require a GPU or HF
network access.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_profiles import (  # noqa: E402
    OOD_SLICE_REGISTRY,
    SUPPORTED_OOD_SLICES,
    SUPPORTED_TIERS,
    MODEL_PRESETS,
    ood_slice_filename,
    ood_slice_floor,
    recommend_ood_slices,
    tier_ood_slices,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Slice registry + tier defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestSliceRegistry:
    """Catch typos and tier-default drift before they hit the eval CLI."""

    def test_registry_keys_match_supported(self):
        assert set(OOD_SLICE_REGISTRY.keys()) == set(SUPPORTED_OOD_SLICES)

    def test_every_slice_has_filename_source_floor(self):
        for slice_name, meta in OOD_SLICE_REGISTRY.items():
            assert "filename" in meta and meta["filename"], slice_name
            assert "source"   in meta and meta["source"],   slice_name
            assert "floor"    in meta, slice_name
            assert 0.0 < float(meta["floor"]) < 1.0

    def test_filenames_unique_and_jsonl(self):
        files = [meta["filename"] for meta in OOD_SLICE_REGISTRY.values()]
        assert len(set(files)) == len(files), "filenames must be unique"
        for f in files:
            assert f.endswith(".jsonl"), f

    @pytest.mark.parametrize("slice_name", SUPPORTED_OOD_SLICES)
    def test_helper_lookups_match_registry(self, slice_name):
        assert ood_slice_filename(slice_name) == OOD_SLICE_REGISTRY[slice_name]["filename"]
        assert ood_slice_floor(slice_name)    == pytest.approx(
            float(OOD_SLICE_REGISTRY[slice_name]["floor"])
        )

    def test_unknown_slice_raises(self):
        with pytest.raises(ValueError):
            ood_slice_filename("not-a-real-slice")

    def test_unknown_slice_floor_falls_back(self):
        # Sensible default rather than raising — full_eval reads this
        # for every sample and must not blow up on a stray slice name.
        assert ood_slice_floor("not-a-real-slice") == pytest.approx(0.25)


class TestTierDefaults:
    @pytest.mark.parametrize("tier", SUPPORTED_TIERS)
    def test_tier_default_is_subset_of_registry(self, tier):
        slices = tier_ood_slices(tier)
        assert all(s in SUPPORTED_OOD_SLICES for s in slices)

    def test_tier_defaults_are_monotonic(self):
        # Small must include everything tiny has; medium must include small.
        tiny   = set(tier_ood_slices("tiny"))
        small  = set(tier_ood_slices("small"))
        medium = set(tier_ood_slices("medium"))
        assert tiny.issubset(small),  f"tiny={tiny} ⊄ small={small}"
        assert small.issubset(medium), f"small={small} ⊄ medium={medium}"

    def test_unknown_tier_returns_medium(self):
        assert tier_ood_slices("nonsense") == tier_ood_slices("medium")

    def test_tiny_tier_excludes_at_floor_slices(self):
        # Tiny models cannot score above the random-MCQ floor on
        # professional_medicine or LSAT-LR — the whole point of the
        # tier system is to keep those slices out of the tiny report.
        slices = tier_ood_slices("tiny")
        assert "medical" not in slices
        assert "legal"   not in slices

    @pytest.mark.parametrize("preset_name", list(MODEL_PRESETS.keys()))
    def test_preset_recommendation_consistent_with_tier(self, preset_name):
        recommended = recommend_ood_slices(preset_name)
        # Empty recommendations should fall back to the tier default;
        # a populated recommendation must be a subset of the registry.
        assert all(s in SUPPORTED_OOD_SLICES for s in recommended)
        assert len(recommended) > 0

    def test_unknown_preset_falls_back_to_medium(self):
        assert recommend_ood_slices("phantom-model") == tier_ood_slices("medium")


# ─────────────────────────────────────────────────────────────────────────────
# 2. full_eval._discover_ood_slices
# ─────────────────────────────────────────────────────────────────────────────


from eval.full_eval import _discover_ood_slices  # noqa: E402


def _write_blank_jsonl(path: Path) -> None:
    """Create an empty (but valid) JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


class TestOODDiscovery:
    def test_auto_discovery_finds_all_existing(self, tmp_path):
        # Create three of the five canonical slice files.
        for slice_name in ("commonsense", "science_easy", "science_hard"):
            _write_blank_jsonl(tmp_path / ood_slice_filename(slice_name))

        pairs = _discover_ood_slices(tmp_path, requested=None)
        names = [p[0] for p in pairs]
        assert names == ["commonsense", "science_easy", "science_hard"], names
        # Order matches the canonical registry order, not directory walk order.

    def test_auto_discovery_empty_dir_returns_empty(self, tmp_path):
        assert _discover_ood_slices(tmp_path, requested=None) == []

    def test_explicit_request_preserves_order(self, tmp_path):
        # Even though the registry order is commonsense first, a caller
        # asking for [legal, commonsense] should get them in *that* order
        # so the rendered report rows come out in caller-controlled order.
        for s in ("commonsense", "legal"):
            _write_blank_jsonl(tmp_path / ood_slice_filename(s))
        pairs = _discover_ood_slices(tmp_path, requested=["legal", "commonsense"])
        assert [p[0] for p in pairs] == ["legal", "commonsense"]

    def test_explicit_request_skips_missing_with_warning(self, tmp_path, capsys):
        # Only commonsense exists on disk.
        _write_blank_jsonl(tmp_path / ood_slice_filename("commonsense"))
        pairs = _discover_ood_slices(
            tmp_path, requested=["commonsense", "medical", "legal"]
        )
        assert [p[0] for p in pairs] == ["commonsense"]
        captured = capsys.readouterr().out
        assert "missing" in captured.lower()
        # The message should tell the user how to fix it.
        assert "fetch_ood_data" in captured

    def test_explicit_request_warns_on_unknown_slice(self, tmp_path, capsys):
        _write_blank_jsonl(tmp_path / ood_slice_filename("commonsense"))
        pairs = _discover_ood_slices(
            tmp_path, requested=["commonsense", "made-up"]
        )
        assert [p[0] for p in pairs] == ["commonsense"]
        captured = capsys.readouterr().out
        assert "unknown" in captured.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. compare_runs transfer-report rendering
# ─────────────────────────────────────────────────────────────────────────────


from eval.compare_runs import (  # noqa: E402
    _render_transfer_section,
    render_report,
)


def _mk_sample(conf: float, correct: bool, slice_name: str) -> Dict:
    return {
        "confidence":   conf,
        "correct":      correct,
        "reward":       0.0,
        "parsed_type":  "answer",
        "format_valid": True,
        "domain":       slice_name,
        "ood_slice":    slice_name,
        "source":       slice_name,
    }


def _mk_indist_sample(conf: float, correct: bool, domain: str, difficulty: int) -> Dict:
    s = _mk_sample(conf, correct, domain)
    s["difficulty"] = difficulty
    s.pop("ood_slice")  # in-dist samples don't carry ood_slice
    return s


def _build_indist_section(
    *, overconfident: bool, n_per_cond: int = 30, seed: int = 0
) -> Dict:
    """Build a 15-condition in-dist section. If overconfident, every
    sample emits 0.85 regardless of correctness; otherwise confidence
    matches correctness (which gives ~ ECE 0.05).
    """
    rng = random.Random(seed)
    out = {}
    for d in ("math", "code", "logic"):
        for diff in (1, 2, 3, 4, 5):
            samples = []
            for _ in range(n_per_cond):
                acc_p = max(0.05, 0.85 - 0.10 * (diff - 1))
                correct = rng.random() < acc_p
                if overconfident:
                    conf = 0.85
                else:
                    conf = 0.85 if correct else 0.20
                samples.append(_mk_indist_sample(conf, correct, d, diff))
            out[f"{d}_{diff}"] = {"samples": samples, "n_samples": n_per_cond}
    return out


def _build_ood_slice(
    slice_name: str, n: int, accuracy: float, *, calibrated: bool, seed_offset: int = 0
) -> Dict:
    """Synthesize an OOD slice with a deliberately exaggerated calibration
    gap so the bootstrap CI is stably clean of zero at moderate sample
    sizes. Baseline is overconfident at 0.95 regardless of correctness;
    calibrated mode emits 0.90 / 0.05 — a 0.85-point spread that keeps
    the ΔECE comfortably outside the 95% CI of pure noise.
    """
    samples = []
    rng = random.Random(hash(slice_name) % (2**31) + seed_offset)
    for _ in range(n):
        correct = rng.random() < accuracy
        if calibrated:
            conf = 0.90 if correct else 0.05
        else:
            conf = 0.95
        samples.append(_mk_sample(conf, correct, slice_name))
    return {"samples": samples, "random_floor": ood_slice_floor(slice_name)}


class TestTransferRendering:
    def test_clean_transfer_renders_strong_headline(self):
        """Synthetic baseline (overconfident everywhere) vs after (calibrated
        everywhere) → headline should report transfer with ✓ status on
        slices clear of floor."""
        baseline = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "preset":          "qwen0.5b",
            "in_distribution": _build_indist_section(overconfident=True, seed=1),
            "ood": {
                "commonsense":  _build_ood_slice("commonsense",  200, 0.45, calibrated=False, seed_offset=1),
                "science_easy": _build_ood_slice("science_easy", 200, 0.55, calibrated=False, seed_offset=1),
            },
        }
        after = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "preset":          "qwen0.5b",
            "in_distribution": _build_indist_section(overconfident=False, seed=2),
            "ood": {
                "commonsense":  _build_ood_slice("commonsense",  200, 0.48, calibrated=True, seed_offset=2),
                "science_easy": _build_ood_slice("science_easy", 200, 0.60, calibrated=True, seed_offset=2),
            },
        }
        lines = _render_transfer_section(baseline, after)
        body = "\n".join(lines)

        # Section header present.
        assert "## 4. Calibration Transfer" in body

        # Both OOD slices show up in the table.
        assert "| commonsense |"   in body
        assert "| science_easy |"  in body

        # Both OOD slices should be flagged as transferred (CI excludes 0).
        assert body.count("✓ transferred") >= 2

        # Headline summarises positively.
        assert "**Transfer ratio:**" in body
        assert "transfers to OOD" in body

    def test_at_floor_slice_marked_and_excluded_from_ratio(self):
        """A slice where acc is at floor should be flagged ⚠ and not
        contribute to the transfer-ratio average."""
        baseline = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "in_distribution": _build_indist_section(overconfident=True, seed=10),
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 200, 0.45, calibrated=False, seed_offset=11),
                "medical":     _build_ood_slice("medical",     149, 0.27, calibrated=False, seed_offset=12),
                "legal":       _build_ood_slice("legal",       200, 0.21, calibrated=False, seed_offset=13),
            },
        }
        after = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "in_distribution": _build_indist_section(overconfident=False, seed=11),
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 200, 0.48, calibrated=True, seed_offset=21),
                "medical":     _build_ood_slice("medical",     149, 0.28, calibrated=True, seed_offset=22),
                "legal":       _build_ood_slice("legal",       200, 0.22, calibrated=True, seed_offset=23),
            },
        }
        body = "\n".join(_render_transfer_section(baseline, after))

        # Two at-floor slices flagged in the status column.
        assert body.count("⚠ at floor") >= 2

        # Commonsense (acc 0.48 vs floor 0.20) is NOT at floor — it should
        # be the only slice contributing to the transfer ratio numerator.
        assert "1/3 OOD slices clear of floor" in body, body

    def test_no_id_improvement_short_circuits_ratio(self):
        """When ΔECE_ID >= 0, the transfer ratio should be reported as
        n/a — there's nothing to transfer."""
        # Build a baseline where the after-RL run is *worse* than baseline
        # in-dist (e.g. an SFT step over-corrected). OOD is irrelevant
        # for the headline but still needs samples to render.
        baseline = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "in_distribution": _build_indist_section(overconfident=False, seed=20),  # already calibrated
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 100, 0.45, calibrated=False, seed_offset=30),
            },
        }
        after = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "in_distribution": _build_indist_section(overconfident=True,  seed=21),  # got worse
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 100, 0.45, calibrated=True, seed_offset=31),
            },
        }
        body = "\n".join(_render_transfer_section(baseline, after))
        assert "Transfer ratio:** n/a" in body
        assert "in-distribution ECE did not improve" in body

    def test_render_report_includes_section_4(self):
        """End-to-end: the full report renders all six sections in order."""
        baseline = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "in_distribution": _build_indist_section(overconfident=True, seed=40),
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 100, 0.45, calibrated=False, seed_offset=40),
            },
        }
        after = {
            "model_id":        "Qwen/Qwen2.5-0.5B-Instruct",
            "in_distribution": _build_indist_section(overconfident=False, seed=41),
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 100, 0.48, calibrated=True, seed_offset=41),
            },
        }
        report = render_report(baseline, after)
        for hdr in (
            "## 1. Headline",
            "## 2. Per-Domain Breakdown",
            "## 3. In-Distribution vs OOD",
            "## 4. Calibration Transfer",
            "## 5. Confidence Distribution",
            "## 6. Operating-Mode Shifts",
        ):
            assert hdr in report, f"missing section: {hdr}"

    def test_baseline_with_no_ood_still_renders_section(self):
        """Old-style baseline JSON (no OOD samples) should not crash —
        the section should explain what to do."""
        baseline = {
            "model_id":  "Qwen/Qwen2.5-3B-Instruct",
            "conditions": _build_indist_section(overconfident=True, seed=50),
        }
        after = {
            "model_id":        "Qwen/Qwen2.5-3B-Instruct",
            "in_distribution": _build_indist_section(overconfident=False, seed=51),
            "ood": {
                "commonsense": _build_ood_slice("commonsense", 80, 0.50, calibrated=True, seed_offset=51),
            },
        }
        # render_report flattens baseline.conditions into samples — make sure
        # we don't blow up if baseline.ood is absent entirely.
        body = render_report(baseline, after)
        assert "## 4. Calibration Transfer" in body
        # No baseline OOD → ECE before column should be NaN-rendered (the
        # cell still appears, just with 'nan'); the section as a whole
        # must not raise.
        assert "commonsense" in body


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end smoke: full_eval --dry-run with auto-discovered slices
# ─────────────────────────────────────────────────────────────────────────────


class TestFullEvalDryRun:
    def test_full_eval_dry_run_uses_auto_slices(self, tmp_path):
        """Running ``full_eval --dry-run --skip-indist --ood-slices auto``
        on a tiny preset should evaluate exactly the three tier-default
        slices when their JSONL files exist on disk.
        """
        # Materialise the three tier-default slice files with one trivial
        # sample each. The dry-run stub answers 'A' with conf 0.7 to
        # exercise the strict parser.
        for slice_name in ("commonsense", "science_easy", "science_hard"):
            path = tmp_path / ood_slice_filename(slice_name)
            row = {
                "question": "test?\n\nOptions:\n(A) yes\n(B) no",
                "answer":   "A",
                "domain":   slice_name,
                "source":   slice_name,
            }
            path.write_text(json.dumps(row) + "\n")

        # Add medical+legal too — auto with tiny preset must skip them.
        for s in ("medical", "legal"):
            (tmp_path / ood_slice_filename(s)).write_text(json.dumps({
                "question": "harder?\n\nOptions:\n(A) a\n(B) b",
                "answer":   "A",
                "domain":   s,
                "source":   s,
            }) + "\n")

        # Run full_eval as a subprocess so CLI parsing is exercised.
        import subprocess
        out_json = tmp_path / "full.json"
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "eval" / "full_eval.py"),
                "--dry-run",
                "--skip-indist",
                "--ood-dir", str(tmp_path),
                "--ood-slices", "auto",
                "--model-id", "Qwen/Qwen2.5-0.5B-Instruct",
                "--output", str(out_json),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        payload = json.loads(out_json.read_text())
        # Tiny preset → only the three tier-default slices got evaluated.
        assert set(payload["ood"].keys()) == {
            "commonsense", "science_easy", "science_hard"
        }, payload["ood"].keys()
        # ood_slices field is preserved for downstream consumers.
        assert payload["ood_slices"] == [
            "commonsense", "science_easy", "science_hard"
        ]
        # Each slice carries the random_floor we wrote into the registry.
        for s in ("commonsense", "science_easy", "science_hard"):
            assert payload["ood"][s]["random_floor"] == pytest.approx(
                ood_slice_floor(s)
            )
