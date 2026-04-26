"""Calibration SFT — model-agnostic warmup for tiny / small / medium models.

Why a dedicated SFT phase?
==========================

The RLVE pipeline assumes the base model can already emit the strict
3-tag XML contract::

    <reasoning>...</reasoning>
    <answer>...</answer>
    <confidence>0.X</confidence>

For Qwen-3B and larger, that assumption holds within ~30 GRPO steps.
For Qwen-0.5B and Llama-1B it does **not**: empirically, ~97 % of early
GRPO rollouts are malformed and the run wastes its compute budget on
the malformed-penalty floor. ``frac_reward_zero_std`` stays at 1.0 and
the GRPO advantage signal is identically zero — the model never gets
any calibration gradient at all.

This script teaches three things in a single SFT pass:

1. **Format compliance**: every assistant target uses the exact
   3-tag contract (or ``<abstain/>`` for the hardest examples).
2. **Correctness-conditioned confidence prior**: when the answer
   shown to the model in the SFT target is actually correct, the
   confidence is sampled from a high-band (≈ 0.85 ± 0.10); when it
   is intentionally perturbed (i.e. wrong), confidence is sampled
   from a low-band (≈ 0.25 ± 0.15). The model thus learns *before*
   any GRPO that "I should be less confident when I'm wrong".
3. **Optional ``<hindsight>r</hindsight>`` tag** matching ground-
   truth correctness. This is what the legacy hindsight reward
   grades — without SFT, the base model has zero prior on the tag,
   so the legacy hindsight reward stays silent for the entire GRPO
   run. The SFT pass writes the prior *into* the model so the
   reward channel actually fires.

The same script is shared across all model sizes; the per-tier
defaults from ``calibration_profiles.py`` automatically scale the
example count, max difficulty and hindsight fraction so a tiny
preset gets aggressive warmup while a medium preset gets a light
nudge.

Usage
-----

Tier-appropriate defaults (recommended)::

    python training/calibration_sft.py \\
        --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --output-dir ./sft-qwen-0.5b

Override per-run::

    python training/calibration_sft.py \\
        --model-id meta-llama/Llama-3.2-1B-Instruct \\
        --output-dir ./sft-llama-1b \\
        --n-examples 2000 --epochs 2 --hindsight-frac 0.6

After SFT, resume into GRPO with ``--init-adapter``::

    python training/train_grpo.py \\
        --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --init-adapter ./sft-qwen-0.5b \\
        --hindsight --hindsight-mode legacy

Outputs
-------

  ``<output-dir>/``                LoRA adapter weights + tokenizer
  ``<output-dir>/sft_report.txt``  before/after format-rate diagnostic
"""

from __future__ import annotations

UNSLOTH_AVAILABLE = False
# Unsloth raises SystemExit (not Exception) on machines with no CUDA device,
# so we have to use BaseException — and we additionally gate on torch.cuda
# being available before we even attempt the import. This keeps --dry-run
# usable on CPU-only environments (CI, local laptops).
try:
    import torch as _torch  # noqa: WPS433
    if _torch.cuda.is_available():
        from unsloth import FastLanguageModel, is_bfloat16_supported  # type: ignore
        from unsloth.chat_templates import get_chat_template  # type: ignore
        UNSLOTH_AVAILABLE = True
except BaseException:
    UNSLOTH_AVAILABLE = False

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_profiles import (
    MODEL_PRESETS,
    SUPPORTED_PRESETS,
    get_preset,
    is_tiny_tier,
    prompt_templates,
    recommend_hindsight_mode,
)
from server.generators import code_gen, logic_gen, math_gen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sft")


# ---------------------------------------------------------------------------
# Dataset construction — pure functions, importable for tests
# ---------------------------------------------------------------------------


GENERATORS: Dict[str, Callable[..., Tuple[str, str]]] = {
    "math":  math_gen.generate,
    "code":  code_gen.generate,
    "logic": logic_gen.generate,
}


# Per-difficulty confidence buckets used for *correct* SFT targets.
# Easier difficulties → higher confidence. The model sees these enough
# times that the implicit prior "more difficult ⇒ more uncertain"
# transfers to held-out problems at GRPO time.
_CORRECT_CONFIDENCE_BUCKETS: Dict[int, List[float]] = {
    1: [0.95, 0.90, 0.90, 0.85],
    2: [0.90, 0.85, 0.85, 0.75],
    3: [0.80, 0.70, 0.70, 0.60],
    4: [0.65, 0.55, 0.55, 0.45],
    5: [0.50, 0.40, 0.30, 0.30],
}

# When we deliberately perturb the answer we cap confidence in [0.05, 0.40]
# so the model learns "wrong answer ⇒ low confidence". The lower band is
# tighter than the upper band on purpose — early GRPO empirically tends
# to over-confidently emit wrong answers, so the SFT prior should pull
# in that direction first.
_WRONG_CONFIDENCE_BUCKETS: Dict[int, List[float]] = {
    1: [0.35, 0.30, 0.25, 0.20, 0.15],
    2: [0.30, 0.25, 0.20, 0.15, 0.10],
    3: [0.25, 0.20, 0.15, 0.10, 0.10],
    4: [0.20, 0.15, 0.10, 0.10, 0.05],
    5: [0.15, 0.10, 0.10, 0.05, 0.05],
}


# Reasoning traces for procedural problems. Tiny models can't be expected
# to learn long chains of thought from a few hundred examples; we keep
# the SFT reasoning trace deliberately short so the model imitates a
# *concise* style. Verbose CoT during SFT bloats max_completion_length
# at GRPO time and increases the variance of the rollout-length prior.
_MAX_REASONING_CHARS: int = 96


def _short_reasoning(domain: str, question: str, answer: str, rng: random.Random) -> str:
    """Generate a brief 1-line reasoning trace appropriate to the domain.

    The procedural generators don't expose worked solutions, so we
    synthesize plausible 1-line traces. The exact wording doesn't matter
    for SFT — what matters is that the model learns to *emit something
    short, then close the tag*.
    """
    if domain == "math":
        templates = [
            f"Computing {question.replace(chr(10), ' ').strip()} step by step.",
            f"Direct evaluation: {answer}.",
            "Standard arithmetic; result follows immediately.",
        ]
    elif domain == "code":
        templates = [
            f"Tracing the function on the given input yields {answer}.",
            "Executing the code produces this value.",
            "Substitute and simplify.",
        ]
    else:
        templates = [
            f"Applying the given constraints, the unique solution is {answer}.",
            "Eliminating impossible cases leaves one consistent assignment.",
            "The constraints determine the answer uniquely.",
        ]
    text = rng.choice(templates)
    return text[:_MAX_REASONING_CHARS]


def _perturb_answer(domain: str, ground_truth: str, rng: random.Random) -> str:
    """Produce a plausible-looking but incorrect answer.

    For numeric answers (math + code): nudge by a small offset.
    For word/symbolic answers (logic): swap with a similar-looking token.
    Falls back to "unknown" if nothing reasonable is available.
    """
    try:
        n = int(ground_truth)
        offset = rng.choice([-3, -2, -1, 1, 2, 3, 5, -5, 10, -10])
        candidate = n + offset
        if candidate == n:
            candidate += 1
        return str(candidate)
    except ValueError:
        pass
    if len(ground_truth) <= 2 and ground_truth.isalpha():
        # Likely a logic name like "A" or "Alpha". Pick a different short
        # name from the same lexical bucket.
        alt_pool = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
                    "A", "B", "C", "D", "E"]
        alt_pool = [a for a in alt_pool if a != ground_truth]
        return rng.choice(alt_pool)
    if len(ground_truth) > 0:
        # Append a digit to nudge the string off the canonical answer.
        return ground_truth + str(rng.randint(1, 9))
    return "unknown"


def _format_assistant_target(
    *,
    reasoning: str,
    answer: str,
    confidence: float,
    include_hindsight: bool,
    correct: bool,
    abstain: bool = False,
) -> str:
    """Render the assistant target string for one SFT example.

    Two shapes:
      ``<abstain/>``                                   – skip path
      ``<reasoning>...</reasoning><answer>X</answer>
        <confidence>c</confidence>[<hindsight>r</hindsight>]``

    The hindsight tag, when included, is set to ``1.0`` if the (already
    selected) answer was correct and ``0.0`` if not. This is the
    legacy-hindsight ground truth — exactly what
    ``server.hindsight.compute_hindsight_reward`` grades against.
    """
    if abstain:
        return "<abstain/>"

    conf_str = f"{confidence:.2f}".rstrip("0").rstrip(".")
    if not conf_str:
        conf_str = "0"

    base = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>{answer}</answer>"
        f"<confidence>{conf_str}</confidence>"
    )
    if include_hindsight:
        # Bind hindsight to the ground-truth correctness of the displayed
        # answer. This is the SFT prior that activates the legacy hindsight
        # reward channel during GRPO.
        retro = "1.0" if correct else "0.0"
        base += f"<hindsight>{retro}</hindsight>"
    return base


def build_sft_examples(
    *,
    n: int,
    domain_weights: Dict[str, float],
    max_difficulty: int,
    hindsight_frac: float,
    correct_frac: float = 0.65,
    abstain_frac_at_max: float = 0.10,
    seed: int = 42,
    system_prompt: str,
    user_template: str,
    max_attempts_per_record: int = 5,
) -> List[Dict[str, Any]]:
    """Build ``n`` SFT chat-message examples.

    Returns a list of records with shape::

        {"messages": [{"role": "system", ...},
                      {"role": "user", ...},
                      {"role": "assistant", ...}],
         "meta": {"domain", "difficulty", "correct", "has_hindsight"}}

    The ``meta`` block isn't consumed by SFTTrainer but it lets unit
    tests assert the *distribution* of generated examples is balanced.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= hindsight_frac <= 1.0:
        raise ValueError(f"hindsight_frac must be in [0, 1], got {hindsight_frac}")
    if not 0.0 <= correct_frac <= 1.0:
        raise ValueError(f"correct_frac must be in [0, 1], got {correct_frac}")
    if max_difficulty < 1 or max_difficulty > 5:
        raise ValueError(f"max_difficulty must be in 1..5, got {max_difficulty}")

    rng = random.Random(seed)

    # Domain pool from weights — restrict to non-zero entries.
    domains = [d for d, w in domain_weights.items() if w > 0.0 and d in GENERATORS]
    if not domains:
        domains = list(GENERATORS.keys())
    weights = [domain_weights.get(d, 1.0) for d in domains]

    # Difficulty bands per tier are encoded by ``max_difficulty``. We pick
    # difficulty uniformly in [1, max_difficulty] but bias slightly toward
    # easier problems (the SFT loss converges faster on examples the
    # model can actually solve).
    difficulty_pool = list(range(1, max_difficulty + 1))
    diff_weights = [max(1.0, max_difficulty - d + 1.5) for d in difficulty_pool]

    records: List[Dict[str, Any]] = []
    attempts = 0
    while len(records) < n and attempts < n * max_attempts_per_record:
        attempts += 1
        domain = rng.choices(domains, weights=weights, k=1)[0]
        difficulty = rng.choices(difficulty_pool, weights=diff_weights, k=1)[0]
        gen = GENERATORS[domain]

        try:
            question, ground_truth = gen(difficulty, seed=rng.randint(0, 2**31 - 1))
        except Exception:
            continue

        use_correct = rng.random() < correct_frac
        if use_correct:
            answer = ground_truth
            conf_pool = _CORRECT_CONFIDENCE_BUCKETS[difficulty]
        else:
            answer = _perturb_answer(domain, ground_truth, rng)
            if answer == ground_truth:
                # Perturbation collided with the truth — treat as correct
                # rather than corrupting the prior.
                use_correct = True
                conf_pool = _CORRECT_CONFIDENCE_BUCKETS[difficulty]
            else:
                conf_pool = _WRONG_CONFIDENCE_BUCKETS[difficulty]

        confidence = rng.choice(conf_pool)
        # Add 0.0-0.05 jitter so the confidence distribution isn't a
        # discrete histogram (the model would otherwise memorise the
        # exact bucket values).
        confidence = max(0.0, min(1.0, confidence + rng.uniform(-0.03, 0.03)))

        include_hindsight = rng.random() < hindsight_frac
        abstain = (
            difficulty == max_difficulty
            and difficulty == 5
            and rng.random() < abstain_frac_at_max
        )

        reasoning = _short_reasoning(domain, question, str(answer), rng)
        assistant_text = _format_assistant_target(
            reasoning=reasoning,
            answer=str(answer),
            confidence=confidence,
            include_hindsight=include_hindsight and not abstain,
            correct=use_correct,
            abstain=abstain,
        )

        records.append({
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_template.format(question=question)},
                {"role": "assistant", "content": assistant_text},
            ],
            "meta": {
                "domain": domain,
                "difficulty": difficulty,
                "correct": bool(use_correct),
                "has_hindsight": bool(include_hindsight) and not abstain,
                "abstain": bool(abstain),
            },
        })

    if len(records) < n:
        log.warning(
            "Generator retries exhausted: produced %d / %d examples after %d attempts.",
            len(records), n, attempts,
        )
    return records


def summarise_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the per-{domain, difficulty, correct, has_hindsight} stats
    that the report and tests both consume."""
    total = len(records)
    if total == 0:
        return {"total": 0}
    by_domain: Dict[str, int] = {}
    by_difficulty: Dict[int, int] = {}
    n_correct = 0
    n_hindsight = 0
    n_abstain = 0
    for r in records:
        m = r["meta"]
        by_domain[m["domain"]] = by_domain.get(m["domain"], 0) + 1
        by_difficulty[m["difficulty"]] = by_difficulty.get(m["difficulty"], 0) + 1
        n_correct += int(m["correct"])
        n_hindsight += int(m["has_hindsight"])
        n_abstain += int(m["abstain"])
    return {
        "total":          total,
        "by_domain":      by_domain,
        "by_difficulty":  by_difficulty,
        "frac_correct":   n_correct / total,
        "frac_hindsight": n_hindsight / total,
        "frac_abstain":   n_abstain / total,
    }


# ---------------------------------------------------------------------------
# Format compliance probing
# ---------------------------------------------------------------------------

# We accept both shapes the parser supports: full 3-tag (with optional
# hindsight) or pure abstain. Comparing model outputs against this regex
# lets us measure compliance without a full TRL pipeline.
_FORMAT_PROBE_RE = re.compile(
    r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*<confidence>"
    r"\s*[01](?:\.\d+)?\s*</confidence>"
    r"|<abstain\s*/?>",
    re.DOTALL | re.IGNORECASE,
)


def quick_format_eval(
    model,
    tokenizer,
    *,
    system_prompt: str,
    user_template: str,
    n_probe: int = 30,
    max_difficulty: int = 2,
    seed: int = 99,
) -> float:
    """Run forward passes on n_probe random easy questions and return
    the fraction whose raw output matches the format contract."""
    import torch

    if UNSLOTH_AVAILABLE:
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

    rng = random.Random(seed)
    ok = 0
    n_done = 0
    for i in range(n_probe):
        domain = rng.choice(list(GENERATORS.keys()))
        difficulty = rng.randint(1, max(1, max_difficulty))
        try:
            question, _ = GENERATORS[domain](difficulty, seed=700_000 + i)
        except Exception:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_template.format(question=question)},
        ]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(getattr(model, "device", "cuda"))
        except Exception as exc:
            log.debug("apply_chat_template failed on probe %d: %s", i, exc)
            continue

        with torch.no_grad():
            try:
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except Exception as exc:
                log.debug("model.generate failed on probe %d: %s", i, exc)
                continue
        new_tokens = out[0][ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if _FORMAT_PROBE_RE.search(text):
            ok += 1
        n_done += 1

    if UNSLOTH_AVAILABLE:
        try:
            FastLanguageModel.for_training(model)
        except Exception:
            pass

    if n_done == 0:
        return 0.0
    return ok / n_done


# ---------------------------------------------------------------------------
# Model loading — unsloth-first, HF + bnb fallback
# ---------------------------------------------------------------------------


def _is_bf16_supported() -> bool:
    try:
        import torch
        if UNSLOTH_AVAILABLE:
            return is_bfloat16_supported()
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _load_model_for_sft(
    *,
    model_id: str,
    hf_token: Optional[str],
    lora_r: int,
    lora_alpha: int,
    max_seq_length: int,
):
    """Load the base model with a LoRA adapter attached, ready for SFT."""
    if UNSLOTH_AVAILABLE:
        log.info("Loading %s via Unsloth (4-bit) | LoRA r=%d α=%d", model_id, lora_r, lora_alpha)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=hf_token,
        )
        if "qwen" in model_id.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        tokenizer.padding_side = "right"  # SFT uses right-padding (TRL default)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading %s via HF transformers (4-bit bnb) | LoRA r=%d α=%d",
             model_id, lora_r, lora_alpha)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if _is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Calibration SFT — teach format + confidence prior + hindsight tag "
            "to any supported model. Use the resulting adapter as "
            "--init-adapter for training/train_grpo.py."
        ),
    )
    p.add_argument("--model-id", type=str, required=True,
                   help="HF model id, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'.")
    p.add_argument("--model-preset",
                   choices=["auto", *SUPPORTED_PRESETS],
                   default="auto",
                   help="Calibration preset; 'auto' infers from --model-id.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to save the SFT LoRA adapter.")

    p.add_argument("--n-examples", type=int, default=None,
                   help="Override preset's recommended_sft_examples.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override preset's recommended_sft_epochs.")
    p.add_argument("--max-difficulty", type=int, default=None,
                   help="Override preset's recommended_sft_max_difficulty (1..5).")
    p.add_argument("--hindsight-frac", type=float, default=None,
                   help="Fraction of SFT examples that include a <hindsight> tag.")
    p.add_argument("--correct-frac", type=float, default=0.65,
                   help="Fraction of SFT examples whose answer is the ground truth.")

    p.add_argument("--lora-r", type=int, default=None,
                   help="LoRA rank. Defaults to preset.default_lora_r.")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha. Defaults to 2× lora-r.")
    p.add_argument("--learning-rate", type=float, default=2.0e-5,
                   help="SFT learning rate (gentle nudge).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Effective batch = batch-size × grad-accum.")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--reasoning-mode", default="required",
                   choices=("required", "refined"),
                   help="Prompt mode the SFT examples are constructed against. "
                        "Stick with 'required' unless you specifically want to "
                        "warm a CASR-trained run.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-eval-probe", action="store_true",
                   help="Skip the before/after format-rate probe (faster).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the dataset and print a summary; no model load.")
    p.add_argument("--probe-n", type=int, default=24,
                   help="Number of probes for before/after format-rate eval.")
    return p


def _resolve_args(args, preset) -> None:
    """Fill any unset CLI arg from the preset's tier-aware recommendations."""
    if args.n_examples is None:
        args.n_examples = preset.recommended_sft_examples
    if args.epochs is None:
        args.epochs = preset.recommended_sft_epochs
    if args.max_difficulty is None:
        args.max_difficulty = preset.recommended_sft_max_difficulty
    if args.hindsight_frac is None:
        args.hindsight_frac = preset.recommended_sft_hindsight_frac
    if args.lora_r is None:
        args.lora_r = preset.default_lora_r
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    preset = get_preset(args.model_id, args.model_preset)
    _resolve_args(args, preset)

    system_prompt, user_template = prompt_templates(args.reasoning_mode)

    log.info("=" * 60)
    log.info(
        "Calibration SFT  | model=%s  preset=%s  tier=%s",
        args.model_id, preset.name, preset.tier,
    )
    log.info(
        "  data: n=%d epochs=%d max_d=%d hindsight_frac=%.2f correct_frac=%.2f",
        args.n_examples, args.epochs, args.max_difficulty,
        args.hindsight_frac, args.correct_frac,
    )
    log.info(
        "  optim: lr=%g bs=%d ga=%d (eff=%d) lora_r=%d α=%d",
        args.learning_rate, args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum, args.lora_r, args.lora_alpha,
    )
    if is_tiny_tier(preset.name):
        log.info("  tier='tiny' — SFT is REQUIRED before GRPO; "
                 "recommended hindsight-mode 'legacy' on this tier.")
    else:
        log.info(
            "  tier=%r — SFT is optional but accelerates calibration; "
            "recommended hindsight-mode '%s'.",
            preset.tier, recommend_hindsight_mode(preset.name),
        )
    log.info("=" * 60)

    log.info("Generating %d SFT examples...", args.n_examples)
    records = build_sft_examples(
        n=args.n_examples,
        domain_weights=preset.domain_weights,
        max_difficulty=args.max_difficulty,
        hindsight_frac=args.hindsight_frac,
        correct_frac=args.correct_frac,
        seed=args.seed,
        system_prompt=system_prompt,
        user_template=user_template,
    )
    summary = summarise_records(records)
    log.info("SFT data summary: %s", json.dumps(summary, indent=2, default=str))

    if args.dry_run:
        log.info("--dry-run: stopping after dataset construction.")
        return

    # Heavy imports deferred so --dry-run works without GPU stack.
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTConfig, SFTTrainer  # type: ignore

    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU detected. SFT requires CUDA. Use --dry-run to validate "
            "the dataset builder without a GPU."
        )

    hf_token = os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_for_sft(
        model_id=args.model_id,
        hf_token=hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
    )

    # ── Format-rate probe BEFORE SFT ─────────────────────────────────────
    before_rate = None
    if not args.no_eval_probe:
        log.info("Probing BEFORE-SFT format compliance (%d samples)...", args.probe_n)
        try:
            before_rate = quick_format_eval(
                model, tokenizer,
                system_prompt=system_prompt,
                user_template=user_template,
                n_probe=args.probe_n,
                max_difficulty=min(args.max_difficulty, 2),
            )
            log.info("Before SFT format rate: %.1f%%", before_rate * 100)
        except Exception as exc:
            log.warning("Before-SFT probe failed (%s); skipping.", exc)

    # ── Build TRL-compatible Dataset ─────────────────────────────────────
    raw_messages = [{"messages": r["messages"]} for r in records]
    raw_dataset = Dataset.from_list(raw_messages)

    def _apply_template(examples):
        return {"text": [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            for msgs in examples["messages"]
        ]}
    dataset = raw_dataset.map(_apply_template, batched=True)

    # ── Train ────────────────────────────────────────────────────────────
    total_steps = max(
        1,
        (len(records) // max(1, args.batch_size * args.grad_accum)) * args.epochs,
    )
    log.info("Starting SFT: %d epochs ≈ %d optimizer steps", args.epochs, total_steps)

    bf16 = _is_bf16_supported()
    try:
        train_args = SFTConfig(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            learning_rate=args.learning_rate,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(out_dir),
            report_to="none",
            save_strategy="no",
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )
    except (TypeError, NameError):
        # Older TRL: SFTConfig without max_seq_length argument.
        train_args = TrainingArguments(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            learning_rate=args.learning_rate,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(out_dir),
            report_to="none",
            save_strategy="no",
        )

    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=train_args,
        )
    except TypeError:
        # Older TRL signature.
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=train_args,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        )

    trainer.train()

    # ── Format-rate probe AFTER SFT ──────────────────────────────────────
    after_rate = None
    if not args.no_eval_probe:
        log.info("Probing AFTER-SFT format compliance (%d samples)...", args.probe_n)
        try:
            after_rate = quick_format_eval(
                model, tokenizer,
                system_prompt=system_prompt,
                user_template=user_template,
                n_probe=args.probe_n,
                max_difficulty=min(args.max_difficulty, 2),
            )
            log.info("After SFT format rate: %.1f%%", after_rate * 100)
        except Exception as exc:
            log.warning("After-SFT probe failed (%s); skipping.", exc)

    # ── Save adapter ─────────────────────────────────────────────────────
    log.info("Saving LoRA adapter to %s", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # ── Write report ─────────────────────────────────────────────────────
    report_lines = [
        "Calibration SFT Report",
        "=" * 40,
        f"Model:            {args.model_id}",
        f"Preset:           {preset.name} (tier={preset.tier})",
        f"Examples:         {len(records)}",
        f"Epochs:           {args.epochs}",
        f"Optimizer steps:  {total_steps}",
        f"LoRA rank:        {args.lora_r} (α={args.lora_alpha})",
        f"Max difficulty:   {args.max_difficulty}",
        f"Hindsight frac:   {args.hindsight_frac:.2f}",
        f"Correct frac:     {args.correct_frac:.2f}",
        "",
        "Data summary:",
        json.dumps(summary, indent=2, default=str),
        "",
    ]
    if before_rate is not None and after_rate is not None:
        report_lines.extend([
            f"Before SFT format rate: {before_rate * 100:.1f}%",
            f"After  SFT format rate: {after_rate * 100:.1f}%",
            f"Improvement:            {(after_rate - before_rate) * 100:+.1f}%",
            "",
            "PASS" if after_rate >= 0.85 else "WARN: target ≥ 85% format compliance",
        ])
    report_path = out_dir / "sft_report.txt"
    report_path.write_text("\n".join(str(x) for x in report_lines))
    log.info("SFT report -> %s", report_path)

    log.info(
        "Done. Resume into GRPO with:\n"
        "    python training/train_grpo.py --model-id %s --init-adapter %s",
        args.model_id, out_dir,
    )


if __name__ == "__main__":
    main()
