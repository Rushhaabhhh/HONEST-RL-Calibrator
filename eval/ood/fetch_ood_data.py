"""Fetch OOD evaluation data from HuggingFace datasets.

Materialises one JSONL per OOD slice in ``eval/ood/``. ``full_eval.py``
auto-discovers those files at evaluation time (any new slice added here
is picked up with no further plumbing).

Slice catalogue (kept in sync with
``calibration_profiles.OOD_SLICE_REGISTRY``):

  ┌──────────────┬─────────────────────────────────────────────┬───────┐
  │ slice        │ source                                     │ floor │
  ├──────────────┼─────────────────────────────────────────────┼───────┤
  │ commonsense  │ tau/commonsense_qa :: validation            │ 0.20  │
  │ science_easy │ allenai/ai2_arc :: ARC-Easy :: test         │ 0.25  │
  │ science_hard │ cais/mmlu :: astronomy :: test              │ 0.25  │
  │ medical      │ cais/mmlu :: professional_medicine :: val   │ 0.25  │
  │ legal        │ AGIEval LSAT-LR (with MMLU law fallback)    │ 0.20  │
  └──────────────┴─────────────────────────────────────────────┴───────┘

Why these five slices? The transferability claim
("RL-trained calibration generalises to OOD") only holds when the model
can score *measurably above* the random-MCQ floor on the slice — at the
floor, ECE/Brier deltas collapse into bootstrap noise and the claim is
empirically unprovable. ``commonsense`` / ``science_easy`` /
``science_hard`` give tiny models (Qwen-0.5B, Llama-1B) genuine
calibration headroom; ``medical`` / ``legal`` keep the original
hard-MCQ probes for medium-tier models.

Tier-aware fetching:

    python eval/ood/fetch_ood_data.py --tier tiny     # 3 small-model slices
    python eval/ood/fetch_ood_data.py --tier medium   # all 5 slices  (default)
    python eval/ood/fetch_ood_data.py --slices commonsense,medical
    python eval/ood/fetch_ood_data.py --n 200 --seed 42

Each output line:
    {"question": str, "answer": str, "domain": str, "source": str}

The ``answer`` field is always a single capital letter (A-E); the
``domain`` field equals the slice name and is consumed by the
calibration-transfer report.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_profiles import (  # noqa: E402
    OOD_SLICE_REGISTRY,
    SUPPORTED_OOD_SLICES,
    SUPPORTED_TIERS,
    ood_slice_filename,
    tier_ood_slices,
)


# MCQ answer index -> letter
_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def _safe_load_dataset(path: str, *args, **kwargs):
    """Wrapper around ``datasets.load_dataset`` with a clean import error."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install datasets to fetch OOD data:  pip install datasets"
        ) from exc
    return load_dataset(path, *args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Slice fetchers — one function per slice.
# Contract: each returns a list[dict] with the public schema, capped at ``n``.
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_mmlu_medicine(n: int, seed: int) -> List[Dict]:
    """MMLU professional_medicine (validation, ~149 questions).

    Used as a redistributable substitute for MedQA — same physician-
    licensing exam genre, no licensing snag.
    """
    print(f"Fetching MMLU professional_medicine ({n} samples)...")
    ds = _safe_load_dataset("cais/mmlu", "professional_medicine", split="validation")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break
        choices    = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        if not choices or answer_idx not in range(len(choices)):
            continue
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {c}" for i, c in enumerate(choices)
        )
        question = f"{row['question']}\n\nOptions:\n{options_str}"
        records.append({
            "question": question,
            "answer":   _IDX_TO_LETTER.get(answer_idx, str(answer_idx)),
            "domain":   "medical",
            "source":   "mmlu/professional_medicine",
        })
    print(f"  -> {len(records)} medical questions fetched.")
    return records


def _fetch_agieval_lsat(n: int, seed: int) -> List[Dict]:
    """AGIEval LSAT-LR (test) — with MMLU professional_law fallback."""
    print(f"Fetching AGIEval lsat-lr ({n} samples)...")
    try:
        ds = _safe_load_dataset("dmayhem93/agieval-lsat-lr", split="test")
    except Exception as e:
        print(f"  Primary fetch failed ({e}); trying mirror...")
        try:
            ds = _safe_load_dataset("hails/agieval-lsat-lr", split="test")
        except Exception as e2:
            print(f"  Mirror failed ({e2}); using MMLU professional_law as fallback.")
            return _fetch_mmlu_lsat_fallback(n, seed)

    ds = ds.shuffle(seed=seed)
    records = []
    for row in ds:
        if len(records) >= n:
            break
        question_text = row.get("query") or row.get("question", "")
        options       = row.get("choices") or row.get("options", [])
        label         = row.get("gold") or row.get("answer", "")
        if not question_text or not options:
            continue
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {opt}" for i, opt in enumerate(options)
        )
        question_full = f"{question_text}\n\nOptions:\n{options_str}"
        if isinstance(label, int):
            label = _IDX_TO_LETTER.get(label, str(label))
        elif isinstance(label, list) and label:
            label = str(label[0])
        records.append({
            "question": question_full,
            "answer":   str(label).strip().upper(),
            "domain":   "legal",
            "source":   "agieval/lsat-lr",
        })
    print(f"  -> {len(records)} legal questions fetched.")
    return records


def _fetch_mmlu_lsat_fallback(n: int, seed: int) -> List[Dict]:
    """Fallback when AGIEval LSAT-LR isn't reachable."""
    print("  Using MMLU professional_law as legal fallback...")
    ds = _safe_load_dataset("cais/mmlu", "professional_law", split="test")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break
        choices    = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        if not choices or answer_idx not in range(len(choices)):
            continue
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {c}" for i, c in enumerate(choices)
        )
        question = f"{row['question']}\n\nOptions:\n{options_str}"
        records.append({
            "question": question,
            "answer":   _IDX_TO_LETTER.get(answer_idx, str(answer_idx)),
            "domain":   "legal",
            "source":   "mmlu/professional_law",
        })
    print(f"  -> {len(records)} legal (fallback) questions fetched.")
    return records


def _fetch_commonsense_qa(n: int, seed: int) -> List[Dict]:
    """CommonsenseQA validation slice — 5-way MCQ, broad commonsense.

    Tiny-friendly: Qwen-0.5B / Llama-1B reach 30-45 % accuracy here,
    which is well above the 20 % random floor and gives the
    calibration-transfer report measurable headroom.
    """
    print(f"Fetching CommonsenseQA validation ({n} samples)...")
    try:
        ds = _safe_load_dataset("tau/commonsense_qa", split="validation")
    except Exception as e:
        print(f"  Primary fetch failed ({e}); trying alt mirror...")
        ds = _safe_load_dataset("commonsense_qa", split="validation")

    ds = ds.shuffle(seed=seed)
    records = []
    for row in ds:
        if len(records) >= n:
            break
        question_stem = row.get("question") or ""
        choices_obj   = row.get("choices") or {}
        labels = list(choices_obj.get("label", []))
        texts  = list(choices_obj.get("text", []))
        answer_key = (row.get("answerKey") or "").strip().upper()
        if not question_stem or not labels or not texts or len(labels) != len(texts):
            continue
        if answer_key not in labels:
            continue
        options_str = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
        question_full = f"{question_stem}\n\nOptions:\n{options_str}"
        records.append({
            "question": question_full,
            "answer":   answer_key,
            "domain":   "commonsense",
            "source":   "tau/commonsense_qa",
        })
    print(f"  -> {len(records)} commonsense questions fetched.")
    return records


def _fetch_arc_easy(n: int, seed: int) -> List[Dict]:
    """ARC-Easy test slice — grade-school science MCQ.

    Tiny-friendly: Qwen-0.5B reaches ~45-60 %, Llama-1B ~50-65 %. Acts
    as the easy-science probe in the transfer report and is the slice
    most likely to *show* baseline overconfidence (small models tend
    to slam 0.9 on easy MCQ).
    """
    print(f"Fetching ARC-Easy test ({n} samples)...")
    ds = _safe_load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break
        question_stem = row.get("question") or ""
        choices_obj   = row.get("choices") or {}
        labels = list(choices_obj.get("label", []))
        texts  = list(choices_obj.get("text", []))
        answer_key = (row.get("answerKey") or "").strip().upper()
        if not question_stem or not labels or not texts or len(labels) != len(texts):
            continue
        # ARC sometimes uses "1/2/3/4" labels; remap to letters so the
        # parser contract (single A-E letter) holds.
        if all(lbl.isdigit() for lbl in labels):
            labels = [_IDX_TO_LETTER.get(int(lbl) - 1, lbl) for lbl in labels]
            if answer_key.isdigit():
                answer_key = _IDX_TO_LETTER.get(int(answer_key) - 1, answer_key)
        if answer_key not in labels:
            continue
        options_str = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
        question_full = f"{question_stem}\n\nOptions:\n{options_str}"
        records.append({
            "question": question_full,
            "answer":   answer_key,
            "domain":   "science_easy",
            "source":   "allenai/ai2_arc/ARC-Easy",
        })
    print(f"  -> {len(records)} science_easy (ARC-Easy) questions fetched.")
    return records


def _fetch_mmlu_astronomy(n: int, seed: int) -> List[Dict]:
    """MMLU astronomy test slice — moderate STEM probe.

    Tiny models (~25-35 %) sit closer to the floor here than on
    CommonsenseQA but still above it; medium models (~50-70 %) can
    show a clean transfer signal. We deliberately use ``astronomy``
    rather than ``high_school_physics`` because astronomy MCQs are
    less arithmetic-heavy and so disentangle calibration from
    in-distribution math skill.
    """
    print(f"Fetching MMLU astronomy test ({n} samples)...")
    ds = _safe_load_dataset("cais/mmlu", "astronomy", split="test")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        if len(records) >= n:
            break
        choices    = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        if not choices or answer_idx not in range(len(choices)):
            continue
        options_str = "\n".join(
            f"({_IDX_TO_LETTER.get(i, str(i))}) {c}" for i, c in enumerate(choices)
        )
        question = f"{row['question']}\n\nOptions:\n{options_str}"
        records.append({
            "question": question,
            "answer":   _IDX_TO_LETTER.get(answer_idx, str(answer_idx)),
            "domain":   "science_hard",
            "source":   "mmlu/astronomy",
        })
    print(f"  -> {len(records)} science_hard (MMLU astronomy) questions fetched.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table — slice-name -> fetcher. Keys MUST match
# OOD_SLICE_REGISTRY in calibration_profiles.py.
# ─────────────────────────────────────────────────────────────────────────────


SLICE_FETCHERS: Dict[str, Callable[[int, int], List[Dict]]] = {
    "commonsense":  _fetch_commonsense_qa,
    "science_easy": _fetch_arc_easy,
    "science_hard": _fetch_mmlu_astronomy,
    "medical":      _fetch_mmlu_medicine,
    "legal":        _fetch_agieval_lsat,
}


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_slices(
    slices_arg: Optional[str],
    tier_arg: Optional[str],
) -> List[str]:
    """Decide which slices to fetch from CLI flags.

    Precedence: explicit --slices > --tier > full medium suite.
    """
    if slices_arg:
        slices = [s.strip() for s in slices_arg.split(",") if s.strip()]
        unknown = [s for s in slices if s not in SLICE_FETCHERS]
        if unknown:
            raise SystemExit(
                f"Unknown OOD slice(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(sorted(SLICE_FETCHERS))}"
            )
        return slices
    if tier_arg:
        if tier_arg == "all":
            return list(SUPPORTED_OOD_SLICES)
        if tier_arg not in SUPPORTED_TIERS:
            raise SystemExit(
                f"Unknown --tier '{tier_arg}'. "
                f"Valid: {', '.join(SUPPORTED_TIERS)}, all"
            )
        return list(tier_ood_slices(tier_arg))
    # Default: fetch the full medium suite — keeps backward compat
    # with the pre-tier behaviour (fetched all available slices).
    return list(SUPPORTED_OOD_SLICES)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OOD evaluation data for the calibration-transfer report."
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Samples per OOD slice (default 200; some slices may be capped "
             "by the upstream split size — e.g. MMLU professional_medicine "
             "has ~149)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--out-dir", type=str, default=str(Path(__file__).parent),
        help="Output directory for jsonl files",
    )
    parser.add_argument(
        "--slices", type=str, default=None,
        help=(
            "Comma-separated subset of slices to fetch "
            f"(any of: {', '.join(sorted(SLICE_FETCHERS))}). "
            "Overrides --tier."
        ),
    )
    parser.add_argument(
        "--tier", type=str, default=None,
        choices=[*SUPPORTED_TIERS, "all"],
        help=(
            "Tier-aware fetch shortcut. tiny=3 small-model-friendly slices "
            "(commonsense + ARC-Easy + MMLU astronomy); small=tiny + medical; "
            "medium/all=full suite (medical + legal added)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slices = _resolve_slices(args.slices, args.tier)
    print(f"Fetching slices: {', '.join(slices)}\n")

    failures: List[str] = []
    for slice_name in slices:
        fetcher = SLICE_FETCHERS[slice_name]
        target_path = out_dir / ood_slice_filename(slice_name)
        try:
            records = fetcher(args.n, args.seed)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — keep going on per-slice failures
            print(f"  ✗ {slice_name} failed: {exc}")
            failures.append(slice_name)
            continue
        write_jsonl(records, target_path)

    print()
    if failures:
        print(f"⚠  Some slices failed: {', '.join(failures)}")
        print("   Re-run with --slices to retry only the failed ones.")
    print(f"Done. Wrote slices to {out_dir}")
    print("Next: python eval/full_eval.py --baseline-results eval/baseline_results.json")


if __name__ == "__main__":
    main()
