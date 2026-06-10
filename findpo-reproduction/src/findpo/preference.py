"""Build DPO preference pairs exactly as FinDPO describes (§4.1.1).

For each labeled sample (text x, gold label y):
  - preferred   y_w = the ground-truth label
  - dispreferred y_l:
        if the reference model's prediction == gold  -> a RANDOM different (wrong) label
        if the reference model's prediction != gold  -> the model's (wrong) prediction
This "guide the model away from its own mistakes" recipe is the paper's key design.

We render the prompt with the model's chat template so the policy keeps its
instruction-tuned behaviour. `chosen`/`rejected` are the single-word label completions;
TRL's DPOTrainer tokenizes prompt+completion and appends EOS.

This module is pure/deterministic given a prediction; the reference-model inference that
produces those predictions lives in build_pairs.py (the one GPU preprocessing step).
"""
from __future__ import annotations

import random
from . import LABELS

SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. Given a piece of financial text, decide "
    "whether its sentiment toward the asset/market is positive, negative, or neutral. "
    "Answer with exactly one word: positive, negative, or neutral."
)
USER_TEMPLATE = "Text: {text}\nSentiment:"


def build_prompt(tokenizer, text: str) -> str:
    """Chat-templated prompt string ending at the assistant generation point."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text.strip())},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def rejected_label(gold: str, ref_pred: str | None, rng: random.Random) -> str:
    """The dispreferred completion per FinDPO's recipe."""
    pred = (ref_pred or "").strip().lower()
    if pred in LABELS and pred != gold:
        return pred                                   # model was wrong -> use its mistake
    others = [l for l in LABELS if l != gold]         # model right (or unparsable) -> random wrong
    return rng.choice(others)


def to_dpo_record(tokenizer, text: str, gold: str, ref_pred: str | None,
                  rng: random.Random) -> dict:
    return {
        "prompt": build_prompt(tokenizer, text),
        "chosen": " " + gold,
        "rejected": " " + rejected_label(gold, ref_pred, rng),
    }


if __name__ == "__main__":
    # pure-logic self-test (no model/tokenizer needed)
    rng = random.Random(0)
    assert rejected_label("positive", "negative", rng) == "negative"   # uses the mistake
    assert rejected_label("positive", "positive", rng) in ("negative", "neutral")  # random wrong
    assert rejected_label("neutral", None, rng) in ("negative", "positive")        # unparsable -> random
    assert rejected_label("negative", "negative", rng) in ("neutral", "positive")
    print("preference.rejected_label logic OK")
