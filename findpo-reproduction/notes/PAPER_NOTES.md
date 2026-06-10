# FinDPO — faithful method notes (for the reproduction)

Source: Iacovides, Zhou, Mandic (2025), *FinDPO*, ICAIF '25, arXiv:2507.18417.

## 1. Core idea
SFT (instruction tuning) memorizes and generalizes poorly to unseen financial events.
FinDPO instead aligns a general LLM with **Direct Preference Optimization (DPO)** — the
RL-free preference method — which generalizes better. DPO loss (paper Eq. 1) increases the
likelihood of the preferred (correct) sentiment label and decreases the dispreferred one,
relative to a frozen reference policy, scaled by `beta`.

## 2. Base model
`Llama-3-8B-Instruct` is the reference policy π_ref and the policy initialization.

## 3. Training data (all public on HF) → 3-class {negative, neutral, positive}
- **FPB** Financial PhraseBank — ~4,840 expert-annotated sentences.
- **TFNS** Twitter Financial News Sentiment — ~11,930 tweets.
- **NWGI** News-with-GPT-Instructions — ~16,200 articles, **5-class** (strong/mildly ±,
  neutral) **collapsed to 3** (strong+mildly negative→negative; strong+mildly positive→positive).
- Total ≈ **32,970**, split **80/20** train/test.

## 4. Preference-pair construction (§4.1.1) — implemented in `preference.py` + `build_pairs.py`
For each (text x, gold y):
- `chosen`  = y (ground truth).
- Prompt π_ref with x → prediction p.
  - if **p == y** (model right): `rejected` = a **random different** (wrong) label.
  - if **p != y** (model wrong): `rejected` = **p** (its own mistake) → push the model away from it.

## 5. Training hyperparameters
**Specified in the paper:**
- DPO, **5 epochs**, AdamW optimizer.
- **LoRA**: rank r = 16, α = 16, dropout = 0.05 → 41.9M trainable params (0.52% of base).
- Single **A100-40GB**, ~4.5 h.
- "Deliberately small" learning rate; warm-up ratio and weight decay used as regularization.

**NOT specified (we set documented defaults in `configs/`, tune if needed):**
- exact learning rate → `5e-6`
- DPO `beta` → `0.1` (standard default)
- warm-up ratio → `0.1`; weight decay → `0.0`
- LoRA target modules → all attention + MLP proj (q,k,v,o,gate,up,down)
- batch / grad-accum / max_len → set for memory (16GB QLoRA): bs 1, accum 16, max_len 512.

## 6. logit-to-score + calibration (Remark 2, §4.2) — `scoring.py`
A causal LLM emits a discrete label, but portfolios need a continuous, rankable score.
FinDPO extracts the **logits of the first generated token**, softmaxes over the
sentiment-class tokens → a probability distribution = sentiment score. DPO-aligned models
are **overconfident** (assign ~1.0/0.0), so they apply **temperature scaling** (post-hoc;
T fit on the **training** set by minimizing NLL — *not* on the financial corpus, to avoid
leakage). We implement first-token label logits → temperature-scaled softmax →
`score = P(pos) − P(neg)` ∈ [−1, 1].

## 7. Evaluation
- **Classification** (Table 2): weighted-F1 on FPB/TFNS/NWGI test splits. FinDPO avg
  **0.846**, +11% over FinGPT v3.3. → `eval_clf.py`.
- **Portfolio** (Tables 3-4): 417 S&P-500 names, 2015-02..2021-06, daily sentiment →
  rank → **long top 35% / short bottom 35%**, equal weight, daily rebalance, cost sweep
  0–5bps. FinDPO: **67%/yr, Sharpe 2.0 @5bps** — the only sentiment method still profitable
  at 5bps. → `portfolio.py`. **News corpus is private → exact number not reproducible.**

## 8. Reproducibility tiers (what this repo delivers)
1. **Model + classification F1** — fully reproducible (open base + public data). ✅
2. **logit-to-score mechanism** — fully reproducible. ✅
3. **Exact 67% portfolio** — not reproducible (private 204k-news corpus). ❌
4. **Portfolio pipeline on a public substitute** — reproducible; real number depends on the
   substitute corpus (FNSPID / Kaggle / crypto). ✅(proxy)

## 9. Why this matters for the crypto project (bridge)
The trained FinDPO scores arbitrary text. Pointing it at the BTC/ETH Reddit corpus and
running the same 35/35 long-short = a **FinDPO-on-crypto transfer** the original paper never
did — the natural continuation of the related-work reproduction (where, net of cost, only a
*frontier* LLM extracted a faint signal and even that produced no significant alpha). This
repo is the training half; the crypto scoring/backtest half lives in the other project.
