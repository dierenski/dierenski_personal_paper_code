"""Logit-to-score conversion + temperature scaling — FinDPO's mechanism for turning a
causal LLM's discrete output into a continuous, rankable sentiment score (Remark 2 + §4.2).

Pipeline:
  1. collect_label_logits(model, tok, texts)  [GPU]  -> array [N, 3] of the logits the model
     assigns to the FIRST token of " negative"/" neutral"/" positive" at the generation point.
  2. fit_temperature(logits, label_ids)       [CPU]  -> scalar T minimizing NLL on a labelled
     set (DPO-aligned LLMs are overconfident; T>1 softens the distribution). Calibrate on the
     TRAIN set only — no leakage into the financial test corpus.
  3. logits_to_score(logits, T)               [CPU]  -> per-sample dict: prob vector over the
     3 classes (temperature-scaled softmax) and a scalar score = P(pos) - P(neg) in [-1, 1].

The CPU math (2,3) is unit-tested below; (1) is written to spec (run on your GPU).
"""
from __future__ import annotations

import numpy as np
from . import LABELS


def label_token_ids(tokenizer) -> list[int]:
    """First-token id of each ' <label>' completion (matches the ' '+label used in training)."""
    ids = []
    for lab in LABELS:
        toks = tokenizer.encode(" " + lab, add_special_tokens=False)
        if not toks:
            toks = tokenizer.encode(lab, add_special_tokens=False)
        ids.append(toks[0])
    return ids


def collect_label_logits(model, tokenizer, texts, batch_size: int = 16,
                         max_length: int = 512, device: str | None = None) -> np.ndarray:
    """[GPU] For each text, return the model's next-token logits restricted to the 3 label
    tokens, shape [N, 3] (order = LABELS = [neg, neu, pos])."""
    import torch
    from .preference import build_prompt
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lab_ids = label_token_ids(tokenizer)
    out = np.empty((len(texts), 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [build_prompt(tokenizer, t) for t in texts[i:i + batch_size]]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length).to(device)
            logits = model(**enc).logits                     # [B, T, V]
            last = enc["attention_mask"].sum(1) - 1          # index of last real token per row
            rows = logits[torch.arange(logits.size(0)), last]  # [B, V]
            out[i:i + len(chunk)] = rows[:, lab_ids].float().cpu().numpy()
    return out


def fit_temperature(logits: np.ndarray, label_ids: np.ndarray,
                    grid=None) -> float:
    """[CPU] 1-D temperature that minimizes NLL of the true class. Post-hoc calibration."""
    from scipy.optimize import minimize_scalar

    def nll(T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(p[np.arange(len(label_ids)), label_ids] + 1e-12)))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def logits_to_score(logits: np.ndarray, T: float = 1.0) -> dict:
    """[CPU] Temperature-scaled softmax over the 3 labels + a scalar score P(pos)-P(neg)."""
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    neg, neu, pos = p[:, 0], p[:, 1], p[:, 2]
    return {"prob": p, "score": (pos - neg)}      # score in [-1, 1]; rankable for portfolios


if __name__ == "__main__":
    # CPU self-test: a model that is CONFIDENT but only ~65% accurate is over-confident,
    # so NLL-optimal temperature should SOFTEN (T>1) and must never worsen train NLL.
    rng = np.random.RandomState(0)
    N = 3000
    true = rng.randint(0, 3, size=N)
    pred = true.copy()
    flip = rng.rand(N) < 0.35                              # wrong on 35% of samples
    pred[flip] = (true[flip] + rng.randint(1, 3, size=flip.sum())) % 3
    base = np.full((N, 3), -3.0)
    base[np.arange(N), pred] = 6.0                         # confident on its (often wrong) prediction
    base += rng.randn(N, 3) * 0.3

    def _nll(logits, T, lab):
        z = logits / T; z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return float(-np.mean(np.log(p[np.arange(len(lab)), lab] + 1e-12)))

    T = fit_temperature(base, true)
    assert _nll(base, T, true) <= _nll(base, 1.0, true) + 1e-9   # calibration never worsens NLL
    assert T > 1.0, T                                            # confidently-wrong -> soften
    sc = logits_to_score(base, T)
    assert sc["prob"].shape == (N, 3) and np.allclose(sc["prob"].sum(1), 1.0, atol=1e-5)
    assert sc["score"].min() >= -1 - 1e-6 and sc["score"].max() <= 1 + 1e-6
    print(f"scoring math OK | fitted T={T:.2f} | train NLL {_nll(base,1.0,true):.3f} -> {_nll(base,T,true):.3f}")
