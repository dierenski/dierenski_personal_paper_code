#!/usr/bin/env bash
# End-to-end FinDPO reproduction. Run from the findpo-reproduction/ directory on your GPU box.
#   bash scripts/run_all.sh
# Prereqs: pip install -r requirements.txt ; huggingface-cli login (gated Llama-3) ; ~16GB VRAM.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
BASE="${BASE:-meta-llama/Meta-Llama-3-8B-Instruct}"   # or NousResearch/Meta-Llama-3-8B-Instruct (ungated)
CFG="${CFG:-configs/qlora_16gb.yaml}"

echo "[0/3] sanity: datasets load + label maps"
python -m findpo.data

echo "[1/3] build DPO preference pairs with the reference model (GPU, one-time)"
python -m findpo.build_pairs --base "$BASE" --out data/pref --four-bit
# tip: add --max-samples 200 first for a quick smoke run.

echo "[2/3] DPO + QLoRA training (5 epochs)"
python -m findpo.train --config "$CFG" --pairs data/pref --out outputs/findpo

echo "[3/3] classification benchmark (paper Table 2; target avg weighted-F1 ~0.846)"
python -m findpo.eval_clf --base "$BASE" --model outputs/findpo --four-bit

echo "done. adapter in outputs/findpo. For the portfolio backtest see README.md (needs a public news corpus)."
