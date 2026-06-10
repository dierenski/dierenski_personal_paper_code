# FinDPO — Reproduction

An **independent reproduction** of:

> Giorgos Iacovides, Wuyang Zhou, Danilo Mandic (2025).
> *FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preference
> Optimization of LLMs.* ICAIF '25. arXiv:[2507.18417](https://arxiv.org/abs/2507.18417)

FinDPO aligns **Llama-3-8B-Instruct** with **Direct Preference Optimization (DPO)** —
rather than supervised fine-tuning — on three public financial-sentiment datasets, then
converts the model's logits into a continuous, rankable sentiment score and trades a
long-short book on it. The authors released **neither weights nor code**; this repo
re-implements the method from the paper so it runs on a single consumer GPU.

---

## What is and isn't reproducible (read this first)

| Component | Reproducible here? | Why |
|---|---|---|
| **DPO/LoRA training** of Llama-3-8B-Instruct | ✅ Yes | open base model + 3 public datasets + hyperparameters in the paper |
| **Classification F1** (paper Table 2, avg 0.846) | ✅ Yes | evaluated on the public datasets' test splits |
| **logit-to-score + temperature scaling** | ✅ Yes | fully described (Remark 2, §4.2) |
| **Exact portfolio result** (67%/yr, Sharpe 2.0 @5bps) | ❌ No | the 204k news articles (Reuters/Motley Fool/MarketWatch, 2015–2021) are **self-scraped and not released** |
| **Portfolio *pipeline*** (35/35 long-short, cost sweep) | ✅ Yes | dataset-agnostic; plug in any public news corpus (see below) |

A few training hyperparameters (DPO `beta`, learning rate, warm-up ratio, weight decay)
are **not stated in the paper**; we use documented sensible defaults in `configs/` — tune
them if your F1 lands short of ~0.846. See `notes/PAPER_NOTES.md` for the exact-vs-assumed
breakdown.

---

## Hardware
Defaults are tuned for a **16GB consumer GPU** (e.g. RTX 3070 16GB) via **4-bit QLoRA** +
`ref_model=None` (TRL reuses the base with adapters disabled as the DPO reference, so you
don't hold two 8B models) + gradient checkpointing. The paper used an A100-40GB with bf16
LoRA — use `configs/lora_a100.yaml` if you have ≥40GB. Apple Silicon: see that config with
`four_bit: false` (drop bitsandbytes).

## Install
```bash
# 1) CUDA-matching PyTorch FIRST (example: CUDA 12.1)
pip install "torch>=2.3,<2.6" --index-url https://download.pytorch.org/whl/cu121
# 2) the rest
pip install -r requirements.txt
# 3) Llama-3 is gated: accept the license on its HF page, then:
huggingface-cli login
#    (or use the ungated mirror: set BASE=NousResearch/Meta-Llama-3-8B-Instruct)
```

## Run (end-to-end)
```bash
bash scripts/run_all.sh          # data check -> build pairs -> train -> eval F1
```
Or step by step:
```bash
export PYTHONPATH=$PWD/src
python -m findpo.data                                   # sanity: dataset counts + label maps
python -m findpo.build_pairs --out data/pref --four-bit # GPU, one-time (try --max-samples 200 first)
python -m findpo.train  --config configs/qlora_16gb.yaml --pairs data/pref --out outputs/findpo
python -m findpo.eval_clf --model outputs/findpo --four-bit   # weighted-F1 vs paper's 0.846
```

## Repo layout
```
findpo-reproduction/
  src/findpo/
    data.py         # load+normalize FPB / TFNS / NWGI to 3 classes (8020 splits)
    preference.py   # build DPO pairs (chosen=gold; rejected=model's mistake or random wrong)
    build_pairs.py  # [GPU] run reference model -> write preference dataset
    train.py        # [GPU] DPO + (Q)LoRA training
    scoring.py      # logit-to-score + temperature scaling (calibrate on train)
    eval_clf.py     # weighted-F1 on test splits (paper Table 2)
    portfolio.py    # dataset-agnostic 35/35 long-short + cost sweep (paper Tables 3-4)
  configs/          # qlora_16gb.yaml (default) | lora_a100.yaml
  scripts/run_all.sh
  notes/PAPER_NOTES.md
  requirements.txt
```

## Expected results
- Reference (untrained) Llama-3-8B-Instruct: modest F1 (it's a generalist).
- After DPO: average weighted-F1 approaching the paper's **0.846**; per-dataset roughly
  FPB ≈ 0.86, TFNS ≈ 0.87, NWGI ≈ 0.83 (your exact numbers depend on the unspecified
  hyperparameters and seed).

## Portfolio backtest (and the honest caveat)
`findpo.portfolio.long_short_backtest(scores, prices)` builds the paper's 35/35 daily
long-short with a transaction-cost sweep. The paper's exact 67%/Sharpe-2.0 number used a
**private** 204k-article corpus, so it can't be reproduced verbatim. Use a **public**
substitute to get a real number:
- **FNSPID** (Dong et al., 2024) — 15.7M financial news records, public on HuggingFace;
- Kaggle financial-news datasets;
- **Bridge to crypto:** `findpo.scoring` scores *any* text, so you can point the trained
  model at a crypto Reddit/news corpus and run the same long-short — a clean
  FinDPO-on-crypto transfer (the direction my other reproduction work pointed to).

## Credits & license
Method and datasets are the original authors'. This is an independent, good-faith
reproduction for research; please **cite the FinDPO paper** and the dataset sources
(Financial PhraseBank; Twitter Financial News Sentiment; News-with-GPT-instructions /
NWGI) if you build on it. Code here is provided as-is for reproduction.
