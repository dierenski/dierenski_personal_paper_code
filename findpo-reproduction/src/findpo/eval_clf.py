"""Reproduce FinDPO's classification benchmark (Table 2): weighted-F1 on the held-out
test splits of FPB / TFNS / NWGI.

Prediction = argmax over the model's 3 label-token logits (temperature-free; T only
affects the continuous score, not argmax). FinDPO reports a 3-dataset average weighted-F1
of 0.846 (beating FinGPT v3.3 by 11%); a faithful QLoRA reproduction should land close
(exact number depends on LR/beta/seed, which the paper does not fully specify).

Usage (on your GPU, after training):
    python -m findpo.eval_clf --model outputs/findpo --base meta-llama/Meta-Llama-3-8B-Instruct
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from .scoring import collect_label_logits


def evaluate(model, tokenizer, datasets: dict) -> dict:
    """datasets: {name: DatasetDict with 'test'}. Returns per-dataset + average weighted-F1."""
    rows, f1s = {}, []
    for name, dd in datasets.items():
        te = dd["test"]
        logits = collect_label_logits(model, tokenizer, list(te["text"]))
        pred = logits.argmax(axis=1)
        gold = np.array(te["label_id"])
        f1 = f1_score(gold, pred, average="weighted")
        acc = accuracy_score(gold, pred)
        rows[name] = {"weighted_f1": float(f1), "accuracy": float(acc), "n": len(gold)}
        f1s.append(f1)
        print(f"  {name:5s}  weighted-F1 {f1:.3f}  acc {acc:.3f}  (n={len(gold)})")
    avg = float(np.mean(f1s))
    rows["average"] = {"weighted_f1": avg}
    print(f"  ----  average weighted-F1 {avg:.3f}   (FinDPO paper: 0.846)")
    return rows


def _load_model(base: str, adapter: str | None, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                      # last-token logits need left padding
    kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--model", default=None, help="LoRA adapter dir (omit to eval the base model)")
    ap.add_argument("--four-bit", action="store_true", default=True)
    ap.add_argument("--full-precision", dest="four_bit", action="store_false")
    args = ap.parse_args()
    from .data import load_all
    model, tok = _load_model(args.base, args.model, args.four_bit)
    print(f"Evaluating {'adapter '+args.model if args.model else 'BASE '+args.base}:")
    evaluate(model, tok, load_all())


if __name__ == "__main__":
    main()
