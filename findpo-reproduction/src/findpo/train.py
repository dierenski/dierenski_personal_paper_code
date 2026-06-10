"""[GPU] DPO + (Q)LoRA training of Llama-3-8B-Instruct on the preference dataset.

Reproduces FinDPO's training stage (§4.1.2): DPO alignment, LoRA (r=16, alpha=16,
dropout=0.05), AdamW, 5 epochs. Defaults here use 4-bit QLoRA so it fits a 16GB consumer
GPU (the paper used a single A100-40GB with bf16 LoRA). Reference model = None: with a
PEFT policy, TRL uses the base model with adapters disabled as the reference, which avoids
holding a second 8B model in memory — essential at 16GB.

    python -m findpo.train --config configs/qlora_16gb.yaml --pairs data/pref --out outputs/findpo

Hyperparameters the paper does NOT specify (DPO beta, learning rate, warmup, weight decay)
use documented sensible defaults — see configs/*.yaml and notes/PAPER_NOTES.md. Tune if F1
lands short of ~0.846.

TRL API note: written for trl>=0.11,<0.13 (DPOConfig + DPOTrainer, `tokenizer=`). Newer trl
renames `tokenizer` -> `processing_class`; if you upgrade, change that one kwarg.
"""
from __future__ import annotations

import argparse
import yaml
import torch
from datasets import load_from_disk


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora_16gb.yaml")
    ap.add_argument("--pairs", default="data/pref")
    ap.add_argument("--out", default="outputs/findpo")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = {"device_map": "auto"}
    if cfg.get("four_bit", True):
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        load_kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    if cfg.get("four_bit", True):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05), bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]))

    dpo_args = DPOConfig(
        output_dir=args.out,
        num_train_epochs=cfg.get("epochs", 5),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        learning_rate=cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        weight_decay=cfg.get("weight_decay", 0.0),
        beta=cfg.get("beta", 0.1),                       # DPO temperature (paper unspecified)
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 384),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=cfg.get("logging_steps", 20),
        save_strategy="epoch", report_to="none", seed=cfg.get("seed", 42))

    train_ds = load_from_disk(args.pairs)
    trainer = DPOTrainer(model=model, ref_model=None, args=dpo_args,
                         train_dataset=train_ds, tokenizer=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
