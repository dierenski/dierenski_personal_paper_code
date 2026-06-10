"""Dataset-agnostic long-short portfolio backtest — FinDPO's real-world evaluation
(§4.2, Tables 3-4).

FinDPO ranks stocks by daily sentiment, longs the top 35% and shorts the bottom 35%
(equal-weight, daily rebalance), and reports net-of-cost performance; it claims 67%/yr
and Sharpe 2.0 at 5bps. We CANNOT reproduce that exact number: the 204k news articles
(Reuters / Motley Fool / MarketWatch, 2015-02..2021-06) are self-scraped and NOT public.

What we provide instead is a faithful, dataset-agnostic pipeline: give it a table of
(date, ticker, sentiment_score) and a price source, and it builds the 35/35 long-short
book with a transaction-cost sweep. Plug in any PUBLIC news corpus to get a real number:
  - FNSPID (Dong et al., 2024) — 15.7M financial news records, public on HF;
  - Kaggle financial-news datasets;
  - or, to connect to our other project, the crypto Reddit corpus (see ../README.md).

Inputs
------
scores : DataFrame[date, ticker, score]   (score in [-1,1] from findpo.scoring)
prices : wide DataFrame indexed by date, columns = tickers, values = adjusted close
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252  # equities


# ---------- metrics (self-contained; no external project deps) ----------
def sharpe(r, ann=ANN):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return float(r.mean() / (r.std() + 1e-12) * np.sqrt(ann)) if len(r) > 1 else np.nan

def sortino(r, ann=ANN):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    d = r[r < 0]; dd = np.sqrt(np.mean(d ** 2)) if len(d) else 1e-12
    return float(r.mean() / (dd + 1e-12) * np.sqrt(ann)) if len(r) > 1 else np.nan

def max_drawdown(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) < 2: return np.nan
    cum = np.cumprod(1 + r); peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())

def metrics(r, ann=ANN):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) < 5: return {k: np.nan for k in ("ann_ret", "ann_vol", "sharpe", "sortino", "mdd", "calmar", "cum_ret")}
    cum = float(np.prod(1 + r)); ann_ret = cum ** (ann / len(r)) - 1; mdd = max_drawdown(r)
    return {"ann_ret": ann_ret, "ann_vol": float(r.std() * np.sqrt(ann)), "sharpe": sharpe(r, ann),
            "sortino": sortino(r, ann), "mdd": mdd,
            "calmar": float(ann_ret / abs(mdd)) if mdd and abs(mdd) > 1e-9 else np.nan,
            "cum_ret": cum - 1}


def long_short_backtest(scores: pd.DataFrame, prices: pd.DataFrame,
                        pct: float = 0.35, costs_bps=(0, 1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Daily-rebalanced equal-weight long-short (top `pct` vs bottom `pct` by score).
    Sentiment at day t -> position over t->t+1 return. Turnover-based cost sweep.
    Returns one row of metrics per cost level."""
    fwd = prices.pct_change().shift(-1)                      # return earned by a position held at t
    piv = scores.pivot_table(index="date", columns="ticker", values="score", aggfunc="mean")
    piv.index = pd.to_datetime(piv.index)
    fwd.index = pd.to_datetime(fwd.index)
    dates = piv.index.intersection(fwd.index)
    piv, fwd = piv.reindex(dates), fwd.reindex(index=dates, columns=piv.columns)

    w = pd.DataFrame(0.0, index=dates, columns=piv.columns)
    for dt in dates:
        s = piv.loc[dt].dropna()
        if len(s) < 6:
            continue
        k = max(1, int(round(len(s) * pct)))
        longs, shorts = s.nlargest(k).index, s.nsmallest(k).index
        w.loc[dt, longs] = 1.0 / k
        w.loc[dt, shorts] = -1.0 / k

    gross = (w * fwd).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    rows = []
    for c in costs_bps:
        net = (gross - (c * 1e-4) * turnover).dropna()
        m = metrics(net); m["cost_bps"] = c; rows.append(m)
    return pd.DataFrame(rows).set_index("cost_bps")


def load_prices(tickers, start, end) -> pd.DataFrame:
    """Adjusted-close wide frame via yfinance (one free public price source)."""
    import yfinance as yf
    df = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
    return px.dropna(how="all")


if __name__ == "__main__":
    # CPU self-test on synthetic data: a signal that genuinely predicts next-day returns
    # should yield a positive gross Sharpe that decays with cost.
    rng = np.random.RandomState(0)
    dates = pd.bdate_range("2020-01-01", periods=400)
    tks = [f"T{i}" for i in range(20)]
    rets = pd.DataFrame(rng.randn(len(dates), len(tks)) * 0.02, index=dates, columns=tks)
    prices = (1 + rets).cumprod() * 100
    # score at t = next-day return + noise (informative by construction)
    sc = rets.shift(-1) + rng.randn(len(dates), len(tks)) * 0.02
    long_df = sc.reset_index().melt(id_vars="index", var_name="ticker", value_name="score").rename(columns={"index": "date"})
    res = long_short_backtest(long_df, prices)
    print(res[["ann_ret", "sharpe", "mdd", "cum_ret"]].round(3).to_string())
    assert res.loc[0, "sharpe"] > res.loc[5, "sharpe"], "cost should reduce Sharpe"
    print("portfolio backtest OK (gross Sharpe > net Sharpe, as expected)")
