UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
    print("Unsloth available — using optimised path.")
except Exception:
    print("Unsloth not available, using HF fallback.")

import argparse
import asyncio
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.generators import code_gen, logic_gen, math_gen
from server.reward import (
    parse_action,
    compute_reward,
    reward_brier,
    reward_format,
    reward_accuracy,
    reward_anti_hedge,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")

MODEL_ID         = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTERS_DIR = str(PROJECT_ROOT / "training" / "format_sft_adapters")
OUTPUT_DIR       = "./honest-qwen-3b-grpo"
MAX_SEQ_LEN      = 2048
N_PROMPT_DATASET = 3000 

GENERATORS = {
    "math":  math_gen.generate,
    "code":  code_gen.generate,
    "logic": logic_gen.generate,
}

SYSTEM_PROMPT = """You are a precise and well-calibrated AI assistant.

When answering questions, you MUST respond in EXACTLY this format:
<reasoning>
Briefly think step-by-step to solve the problem.
</reasoning>
<answer>YOUR_ANSWER_HERE</answer>
<confidence>0.X</confidence>

Where:
- YOUR_ANSWER_HERE is your best answer to the question
- 0.X is a number between 0.0 and 1.0 representing how confident you are

Rules:
- Confidence 1.0 = completely certain
- Confidence 0.5 = 50/50 guess
- Confidence 0.0 = completely uncertain
- If you are very unsure, use <abstain/> instead
- Never include explanations outside the XML tags
- For numeric answers, give the number only (no units unless asked)
- For string answers, give the exact value only"""

USER_TEMPLATE = "{question}\n\nThink step-by-step in the <reasoning> block, then provide your final answer and confidence."


def build_prompt_dataset(n: int, tokenizer) -> list:
    log.info(f"Building prompt dataset ({n} prompts)...")
    rng = random.Random(1337)
    domain_list = list(GENERATORS.keys())
    records = []
    attempts = 0
    
    diff_weights = [0.40, 0.30, 0.15, 0.10, 0.05] 
    diff_choices = [1, 2, 3, 4, 5]

    while len(records) < n and attempts < n * 5:
        attempts += 1
        domain = rng.choice(domain_list)
        difficulty = rng.choices(diff_choices, weights=diff_weights, k=1)[0]
        seed = 500_000 + attempts
        
        try:
            question, ground_truth = GENERATORS[domain](difficulty, seed=seed)
        except Exception:
            continue
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(question=question)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        records.append({
            "prompt":       prompt_text,
            "ground_truth": str(ground_truth),
            "difficulty":   difficulty,
            "domain":       domain,
        })
        
    log.info(f"  -> {len(records)} prompts ready ({attempts} attempts).")
    return records

# Async env-server reward (fallback to local Brier)

async def _env_reward_async(prompt, completion, ground_truth, difficulty, env_url):
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{env_url}/reset") as resp:
                reset_data = await resp.json()
                
            step_payload = {"session_id": reset_data.get("session_id", ""), "raw_text": completion}
            async with session.post(f"{env_url}/step", json=step_payload) as resp:
                step_data = await resp.json()
            return float(step_data.get("reward", -0.5))
            
    except Exception as e:
        log.warning(f"Server reward failed ({e}), using local Brier score.")
        parsed = parse_action(completion)
        r, _ = compute_reward(parsed, ground_truth, difficulty)
        return float(r)


def make_env_reward_fn(env_url):
    def _fn(completions, prompts, ground_truth, difficulty, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            tasks = [
                _env_reward_async(p, c, gt, diff, env_url)
                for p, c, gt, diff in zip(prompts, completions, ground_truth, difficulty)
            ]
            return list(loop.run_until_complete(asyncio.gather(*tasks)))
        finally:
            loop.close()
    return _fn

# Reward distribution logging

_reward_history: deque = deque(maxlen=500)

def _log_reward_dist(rewards, step):
    _reward_history.extend(rewards)
    if step % 10 == 0 and len(_reward_history) > 0:
        arr = np.array(_reward_history)
        log.info(
            f"Step {step:04d} | mean={arr.mean():.4f}  std={arr.std():.4f}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}"
        )


def wrap_with_logging(fn, step_ref):
    def _logged(completions, prompts, **kwargs):
        rewards = fn(completions, prompts, **kwargs)
        step_ref[0] += 1
        _log_reward_dist(rewards, step_ref[0])
        return rewards
    return _logged

# Model loading

def _is_bfloat16_supported():
    if UNSLOTH_AVAILABLE:
        return is_bfloat16_supported()
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_unsloth(hf_token):
    log.info(f"Loading {MODEL_ID} via Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.padding_side = "left"
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def load_model_standard(hf_token):
    log.info(f"Loading {MODEL_ID} via HF transformers (4-bit bnb)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if _is_bfloat16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=hf_token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# KL early-stopping callback 

class KLEarlyStopCallback(TrainerCallback):
    """Stop training if KL divergence exceeds threshold for too many consecutive steps."""

    def __init__(self, kl_threshold: float = 0.5, patience: int = 20):
        self.kl_threshold = kl_threshold
        self.patience = patience
        self._counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        kl = logs.get("kl") or logs.get("objective/kl")
        if kl is not None:
            if kl > self.kl_threshold:
                self._counter += 1
                log.warning(
                    f"KL={kl:.4f} > {self.kl_threshold} "
                    f"({self._counter}/{self.patience} consecutive steps)"
                )
                if self._counter >= self.patience:
                    log.error("KL divergence too high for too long — stopping training.")
                    control.should_training_stop = True
            else:
                self._counter = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-wandb",  action="store_true")
    args = parser.parse_args()

    hf_token  = os.environ.get("HF_TOKEN")
    env_url   = os.environ.get("HONEST_ENV_URL", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    report_to = "none" if (args.no_wandb or not wandb_key) else "wandb"

    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("No GPU detected.")

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Torch: {torch.__version__}")

    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(hf_token)
    else:
        model, tokenizer = load_model_standard(hf_token)

    raw_records   = build_prompt_dataset(N_PROMPT_DATASET, tokenizer)
    train_dataset = Dataset.from_list(raw_records)
    bf16 = _is_bfloat16_supported()

    if env_url:
        _primary = make_env_reward_fn(env_url)
    else:
        _primary = reward_brier

    _step_ref = [0]
    logged_brier = wrap_with_logging(_primary, _step_ref)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=16,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_completion_length=512,
        learning_rate=1.5e-6,
        beta=0.1,
        max_grad_norm=0.1,
        scale_rewards=True,
        num_iterations=1,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        save_steps=50,
        logging_steps=1,
        fp16=not bf16,
        bf16=bf16,
        optim="adamw_8bit",
        report_to=report_to,
        seed=42,
        **({  "max_steps": args.max_steps} if args.max_steps else {}),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[logged_brier, reward_format, reward_accuracy, reward_anti_hedge],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[KLEarlyStopCallback(kl_threshold=0.5, patience=20)],
    )

    log.info("=" * 60)
    log.info(f"Model:   {MODEL_ID}")
    log.info(f"Backend: {'Unsloth' if UNSLOTH_AVAILABLE else 'HF transformers'}")
    log.info(f"GPU:     {torch.cuda.get_device_name(0)} | bf16 supported: {bf16}")
    log.info(f"Reward:  {'live @ ' + env_url if env_url else 'local multi-reward'}")
    log.info("KL beta: 0.1  |  max_grad_norm: 0.1  |  lr: 1.5e-6")
    log.info("=" * 60)

    t0 = time.time()
    trainer.train()
    log.info(f"Training complete in {(time.time()-t0)/60:.1f} min.")

    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path / "final_adapters"))
    tokenizer.save_pretrained(str(out_path / "final_adapters"))
    log.info(f"Saved to {out_path / 'final_adapters'}")


if __name__ == "__main__":
    main()