"""[GPU preprocessing] Build the DPO preference dataset.

For every labelled sample we need the REFERENCE model's own prediction, because FinDPO's
dispreferred completion is the model's mistake (or, if it was right, a random wrong label).
So this one-time step runs Llama-3-8B-Instruct over all ~33k samples, parses each prediction
to {negative, neutral, positive}, and writes a preference dataset with columns
[prompt, chosen, rejected] ready for train.py.

    python -m findpo.build_pairs --base meta-llama/Meta-Llama-3-8B-Instruct \
        --out data/pref --four-bit

Resumable-ish: pass --max-samples for a quick smoke run first. Output is saved both as a
HF dataset (save_to_disk) and as preference.jsonl for portability.
"""
from __future__ import annotations

import os
import json
import random
import argparse

import torch
from datasets import Dataset

from . import LABELS
from .data import combined_train
from .preference import build_prompt, to_dpo_record


def _parse_label(text: str) -> str | None:
    t = text.strip().lower()
    for lab in LABELS:                       # first label word that appears
        if lab in t:
            return lab
    if "bull" in t: return "positive"
    if "bear" in t: return "negative"
    return None


def predict_labels(model, tokenizer, texts, batch_size=16, max_length=512) -> list[str | None]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds: list[str | None] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [build_prompt(tokenizer, t) for t in texts[i:i + batch_size]]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length).to(device)
            gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            for j in range(gen.size(0)):
                new = gen[j, enc["input_ids"].size(1):]
                preds.append(_parse_label(tokenizer.decode(new, skip_special_tokens=True)))
            if (i // batch_size) % 20 == 0:
                print(f"  predicted {i+len(chunk)}/{len(texts)}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out", default="data/pref")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--four-bit", action="store_true", default=True)
    ap.add_argument("--full-precision", dest="four_bit", action="store_false")
    ap.add_argument("--max-samples", type=int, default=0, help="cap for a quick smoke run (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **kw)

    ds = combined_train()
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    texts, golds = list(ds["text"]), list(ds["label"])
    print(f"Predicting reference labels for {len(texts):,} samples ...")
    preds = predict_labels(model, tok, texts, batch_size=args.batch)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    print(f"  reference-model accuracy on train: {acc:.3f}")

    rng = random.Random(args.seed)
    records = [to_dpo_record(tok, t, g, p, rng) for t, g, p in zip(texts, golds, preds)]
    os.makedirs(args.out, exist_ok=True)
    Dataset.from_list(records).save_to_disk(args.out)
    with open(os.path.join(args.out, "preference.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):,} preference pairs -> {args.out}")


if __name__ == "__main__":
    main()
