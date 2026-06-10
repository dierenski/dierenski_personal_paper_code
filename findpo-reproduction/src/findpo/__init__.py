"""FinDPO reproduction — independent re-implementation of

    Iacovides, Zhou, Mandic (2025), "FinDPO: Financial Sentiment Analysis for
    Algorithmic Trading through Preference Optimization of LLMs" (ICAIF '25),
    arXiv:2507.18417.

This package DPO-aligns Llama-3-8B-Instruct on three public financial-sentiment
datasets (FPB + TFNS + NWGI) with LoRA/QLoRA, converts the resulting classifier's
logits into a continuous sentiment score (logit-to-score + temperature scaling),
reproduces the classification F1 benchmark, and provides a dataset-agnostic
long-short portfolio backtest.

The authors released neither weights nor code; this is an independent reproduction
written from the paper. See ../README.md and ../notes/PAPER_NOTES.md for exactly
what is and is not reproducible.
"""
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
