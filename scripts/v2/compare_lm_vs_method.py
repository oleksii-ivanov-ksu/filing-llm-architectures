#!/usr/bin/env python3
"""
C3 comparison (RQ3): LM scalar -> BL  vs  method (2-stage / 1-stage) -> BL.

Same universe, same rebalance schedule, same BL+tilt strategy. Only the view
source differs. Reported across 10 random 50/50 splits (mean +/- std) plus a
paired t-test of per-split Sharpe (method - LM). $0.
"""
import sys, random, warnings, statistics
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scripts.run_backtest import load_universe, load_all_prices
from src.backtesting.engine import generate_rebalance_dates, load_views_summary, run_backtest
from src.backtesting.metrics import sharpe_ratio as _canon_sharpe   # rf=2%, project standard

STRAT = "bl_llm_tilt_pos"
SOURCES = {
    "LM-scalar": "data/v2/views_lm/views_summary.csv",
    "2-stage":   "data/v2/views_v2/views_summary.csv",
    "1-stage":   "data/v2/views_singlestage/views_summary.csv",
}


def sharpe(daily):
    d = daily.dropna()
    return float(_canon_sharpe(d)) if d.std() > 0 else float("nan")   # canonical, rf=2%


def ann_return(daily):
    d = daily.dropna()
    return float((1 + d).prod() ** (252 / len(d)) - 1) if len(d) else float("nan")


def main():
    uni = load_universe("config/universe.yaml")
    allt = uni["core"] + uni["holdout"]
    ap = load_all_prices(Path("data/raw"), allt)
    ap = ap[ap.index <= pd.Timestamp(2026, 3, 31)]
    allt = [t for t in allt if t in ap.columns]
    dates = generate_rebalance_dates(2006, 2025, "quarterly")

    V = {}
    for name, path in SOURCES.items():
        if Path(path).exists():
            V[name] = load_views_summary(Path(path))
        else:
            print(f"!! missing {name}: {path}")

    random.seed(42)
    splits = []
    for i in range(10):
        s = allt[:]; random.shuffle(s); splits.append(s[:50])

    res = {k: {"sharpe": [], "ann": []} for k in V}
    for A in splits:
        for name, vdf in V.items():
            dr = run_backtest(prices=ap[A], views_df=vdf, tickers=A,
                              rebalance_dates=dates, strategy=STRAT)["daily_returns"]
            res[name]["sharpe"].append(sharpe(dr))
            res[name]["ann"].append(ann_return(dr))

    print(f"\n=== {STRAT} across 10 random 50/50 splits ===\n")
    print(f"{'source':12s} {'Sharpe (mean±std)':>22s} {'Ann.ret (mean±std)':>22s}")
    for name in V:
        s = res[name]["sharpe"]; a = res[name]["ann"]
        print(f"{name:12s} {statistics.mean(s):>10.3f} ± {statistics.pstdev(s):<8.3f} "
              f"{statistics.mean(a)*100:>9.1f}% ± {statistics.pstdev(a)*100:<7.1f}%")

    if "LM-scalar" in res:
        print("\n=== paired t-test of per-split Sharpe (method - LM) ===")
        for m in ("2-stage", "1-stage"):
            if m in res:
                diff = np.array(res[m]["sharpe"]) - np.array(res["LM-scalar"]["sharpe"])
                t, p = stats.ttest_rel(res[m]["sharpe"], res["LM-scalar"]["sharpe"])
                wins = int((diff > 0).sum())
                print(f"  {m} vs LM: ΔSharpe = {diff.mean():+.3f} ± {diff.std():.3f} | "
                      f"t={t:.2f} p={p:.3f} | {m} wins {wins}/10")


if __name__ == "__main__":
    main()
