"""Load and normalize the three public financial-sentiment datasets used by FinDPO.

FinDPO (§4.1.1) trains on a combination of:
  - Financial PhraseBank (FPB)        ~4.8k  expert-annotated sentences
  - Twitter Financial News (TFNS)     ~11.9k tweets
  - GPT-labeled Financial News (NWGI) ~16.2k articles (5-class -> collapsed to 3)
totalling ~32,970 samples, split 80/20 train/test.

All are public on the HuggingFace Hub. We normalize every label to the 3-class
space {negative, neutral, positive} (LABELS in __init__). Label-string matching is
done robustly (substring on a lowercased string), which also collapses NWGI's 5
classes ("strong/mildly negative|positive") into 3 — exactly as the paper does.

We expose:
  load_dataset_3class(name) -> DatasetDict with 'train'/'test', columns ['text','label_id','label']
  load_all()                -> {"fpb":..,"tfns":..,"nwgi":..}
  combined_train()          -> single Dataset (all train splits concatenated) for DPO pair-building

Split policy (the paper is slightly underspecified; we make reproducible choices,
fixed seed 42, documented here):
  - FPB  : no official split -> deterministic 80/20.
  - TFNS : has train + validation -> use validation as the test split.
  - NWGI : has train + test       -> use as-is.
The per-dataset 'test' splits are what eval_clf.py reports weighted-F1 on (paper Table 2).
"""
from __future__ import annotations

import re
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

from . import LABELS, LABEL2ID

SEED = 42

# Source dataset identifiers on the HF Hub (override here if a mirror is needed).
# FPB via the data-only FLARE mirror: the canonical `financial_phrasebank` is a legacy
# *script* dataset that newer huggingface_hub refuses to load ("must be 'namespace/name'").
# FLARE ships ready train/test/valid splits; columns include 'text' (sentence) + 'answer' (label word).
HF_FPB = ("ChanceFocus/flare-fpb", None)
HF_TFNS = ("zeroshot/twitter-financial-news-sentiment", None)
HF_NWGI = ("oliverwang15/news_with_gpt_instructions", None)


def _to_3class(label_str: str) -> str:
    """Map any sentiment label string to {negative, neutral, positive}."""
    s = str(label_str).strip().lower()
    if "neg" in s or "bear" in s:
        return "negative"
    if "pos" in s or "bull" in s:
        return "positive"
    return "neutral"


def _find_cols(ds: Dataset) -> tuple[str, str]:
    """Heuristically locate the (text, label) columns of an arbitrary sentiment ds."""
    cols = ds.column_names
    text_col = next((c for c in ["text", "sentence", "news", "headline", "content"] if c in cols), None)
    # 'answer' first so FLARE-FPB uses the label WORD, not its numeric 'gold' index
    label_col = next((c for c in ["answer", "label", "labels", "sentiment", "gold", "y"] if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError(f"could not find text/label columns in {cols}")
    return text_col, label_col


def _normalize(ds: Dataset, int_label_map: dict | None = None) -> Dataset:
    """Return a Dataset with columns ['text','label','label_id'] in the 3-class space."""
    text_col, label_col = _find_cols(ds)
    feat = ds.features[label_col]
    names = getattr(feat, "names", None)   # ClassLabel -> int needs name lookup

    def _map(ex):
        raw = ex[label_col]
        if names is not None and isinstance(raw, int):
            lab_str = names[raw]                       # ClassLabel int -> name
        elif int_label_map is not None:
            try:
                lab_str = int_label_map[int(raw)]      # handles int AND digit-strings like '0'
            except (ValueError, TypeError, KeyError):
                lab_str = str(raw)
        else:
            lab_str = str(raw)                         # plain string label (FLARE 'answer', NWGI 'label')
        lab = _to_3class(lab_str)
        return {"text": str(ex[text_col]).strip(), "label": lab, "label_id": LABEL2ID[lab]}

    keep = ds.map(_map, remove_columns=[c for c in ds.column_names if c not in ("text", "label", "label_id")])
    keep = keep.filter(lambda e: len(e["text"]) > 0)
    return keep


def load_dataset_3class(name: str) -> DatasetDict:
    name = name.lower()
    if name == "fpb":
        raw = load_dataset(HF_FPB[0])                 # FLARE mirror: data-only, has train/test/valid
        tr = _normalize(raw["train"])
        te_key = "test" if "test" in raw else ("validation" if "validation" in raw else "valid")
        te = _normalize(raw[te_key]) if te_key in raw else tr.train_test_split(0.2, seed=SEED)["test"]
        return DatasetDict(train=tr, test=te)
    if name == "tfns":
        raw = load_dataset(HF_TFNS[0])
        # TFNS int labels: 0=Bearish(neg), 1=Bullish(pos), 2=Neutral
        m = {0: "negative", 1: "positive", 2: "neutral"}
        tr = _normalize(raw["train"], int_label_map=m)
        te_key = "validation" if "validation" in raw else ("test" if "test" in raw else None)
        te = _normalize(raw[te_key], int_label_map=m) if te_key else tr.train_test_split(0.2, seed=SEED)["test"]
        return DatasetDict(train=tr, test=te)
    if name == "nwgi":
        raw = load_dataset(HF_NWGI[0])
        tr = _normalize(raw["train"])
        te = _normalize(raw["test"]) if "test" in raw else tr.train_test_split(0.2, seed=SEED)["test"]
        return DatasetDict(train=tr, test=te)
    raise ValueError(f"unknown dataset {name!r}; use fpb|tfns|nwgi")


def load_all() -> dict[str, DatasetDict]:
    return {n: load_dataset_3class(n) for n in ("fpb", "tfns", "nwgi")}


def combined_train(all_ds: dict[str, DatasetDict] | None = None) -> Dataset:
    """Concatenate the three train splits (with a 'source' column) for DPO pair-building."""
    all_ds = all_ds or load_all()
    parts = []
    for name, dd in all_ds.items():
        parts.append(dd["train"].add_column("source", [name] * len(dd["train"])))
    return concatenate_datasets(parts).shuffle(seed=SEED)


if __name__ == "__main__":
    # quick self-check: counts + label distribution per dataset (no GPU needed)
    a = load_all()
    for n, dd in a.items():
        from collections import Counter
        c = Counter(dd["train"]["label"])
        print(f"{n:5s} train={len(dd['train']):6d} test={len(dd['test']):5d}  dist(train)={dict(c)}")
    print("combined train:", len(combined_train(a)))
