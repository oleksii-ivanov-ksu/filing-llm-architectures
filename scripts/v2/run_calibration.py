#!/usr/bin/env python3
"""
C5 -- confidence / signal calibration (RQ-support, $0).

For each view we compute the realized FORWARD excess return over a ~1-quarter
horizon (H trading days), where "excess" = stock forward return minus the
cross-sectional mean forward return of all views triggered in the same quarter
(a same-period market proxy). We then ask two things:

  (1) Does the confidence LABEL mean anything?  hit-rate(sign) and mean forward
      excess return, bucketed high / medium / low.
  (2) Does the view MAGNITUDE mean anything?  same metrics bucketed by |view_bps|
      tertiles, plus by signed view_bps tertiles (does a more bullish view earn more?).

Run for both architectures (two-stage v2 and one-stage). No API cost.

Usage:
    python scripts/v2/run_calibration.py
"""
import os, sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scripts.run_backtest import load_universe, load_all_prices

H = 63   # trading days ~ one quarter forward
VIEWS = {
    "2-stage": "data/v2/views_v2/views_summary.csv",
    "1-stage": "data/v2/views_singlestage/views_summary.csv",
}


def forward_returns(prices, dates, tickers, H):
    """For each (ticker, view_date) return the H-day forward simple return."""
    px = prices.sort_index()
    idx = px.index
    out = {}
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        out[t] = s
    recs = []
    for t, vd in dates:
        if t not in out:
            recs.append(np.nan); continue
        s = out[t]
        pos = s.index.searchsorted(pd.Timestamp(vd))
        if pos >= len(s) or pos + H >= len(s):
            recs.append(np.nan); continue
        p0, p1 = s.iloc[pos], s.iloc[pos + H]
        recs.append(p1 / p0 - 1 if p0 > 0 else np.nan)
    return recs


def bucket_stats(df, by, label):
    rows = []
    for name, g in df.groupby(by, observed=True):
        g = g.dropna(subset=["fwd_excess"])
        if len(g) == 0:
            continue
        hit = (np.sign(g["view_bps"]) == np.sign(g["fwd_excess"]))
        # hit-rate of sign only meaningful where view_bps != 0
        nz = g[g["view_bps"] != 0]
        hitrate = (np.sign(nz["view_bps"]) == np.sign(nz["fwd_excess"])).mean() if len(nz) else np.nan
        rows.append({
            label: name, "n": len(g),
            "mean_fwd_excess_%": round(g["fwd_excess"].mean() * 100, 3),
            "median_%": round(g["fwd_excess"].median() * 100, 3),
            "sign_hit_rate": round(hitrate, 3) if not np.isnan(hitrate) else None,
        })
    return pd.DataFrame(rows)


def main():
    uni = load_universe("config/universe.yaml")
    allt = uni["core"] + uni["holdout"]
    prices = load_all_prices(Path("data/raw"), allt)
    prices = prices[prices.index <= pd.Timestamp(2026, 3, 31)]

    print(f"# C5 -- confidence / signal calibration (H={H} trading days forward)\n")
    for arch, path in VIEWS.items():
        if not Path(path).exists():
            print(f"## {arch}: MISSING ({path})\n"); continue
        v = pd.read_csv(path)
        v["view_date"] = pd.to_datetime(v["view_date"])
        v = v.dropna(subset=["view_date"])
        v["view_bps"] = pd.to_numeric(v["view_bps"], errors="coerce").fillna(0)
        v["fwd"] = forward_returns(prices, list(zip(v["ticker"], v["view_date"])), allt, H)
        # same-period market proxy: mean forward return of all views in the same quarter
        v["q"] = v["view_date"].dt.to_period("Q")
        v["mkt"] = v.groupby("q")["fwd"].transform("mean")
        v["fwd_excess"] = v["fwd"] - v["mkt"]

        print(f"## {arch}  (n views with forward window = {v['fwd'].notna().sum()})\n")

        # (1) by confidence label
        v["confidence"] = v["confidence"].astype(str).str.lower()
        v["confidence"] = pd.Categorical(v["confidence"], ["high", "medium", "low"], ordered=True)
        b1 = bucket_stats(v, "confidence", "confidence")
        print("### by confidence label")
        print(b1.to_string(index=False), "\n")

        # (2a) by |view_bps| tertile
        nz = v[v["view_bps"] != 0].copy()
        try:
            nz["absmag"] = pd.qcut(nz["view_bps"].abs(), 3, labels=["low|bps|", "mid|bps|", "high|bps|"])
            b2 = bucket_stats(nz, "absmag", "abs_magnitude")
            print("### by |view_bps| tertile (does bigger conviction predict bigger |excess|?)")
            print(b2.to_string(index=False), "\n")
        except ValueError:
            print("### by |view_bps| tertile: not enough distinct values\n")

        # (2b) by signed view_bps tertile
        try:
            v["signed"] = pd.qcut(v["view_bps"], 3, labels=["bearish", "neutral", "bullish"], duplicates="drop")
            b3 = bucket_stats(v, "signed", "signed_view")
            print("### by signed view_bps tertile (does a more bullish view earn more excess?)")
            print(b3.to_string(index=False), "\n")
        except ValueError:
            print("### by signed view_bps tertile: not enough distinct values\n")

        # headline: IC (rank corr view_bps vs fwd_excess)
        vv = v.dropna(subset=["fwd_excess"])
        ic = vv["view_bps"].corr(vv["fwd_excess"], method="spearman")
        print(f"### rank-IC(view_bps, fwd_excess) = {ic:.4f}   (n={len(vv)})\n")


if __name__ == "__main__":
    main()
