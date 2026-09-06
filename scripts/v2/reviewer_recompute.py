#!/usr/bin/env python3
"""
Reviewer $0 recomputes: points 6, 7, 9.

  #6 holdout OOS   : top/bottom/spread + FF5+Mom alpha on the untouched holdout-50,
                     reported next to core-50, so the main result has a clean OOS number.
  #7 formal test   : paired t-test on the daily (top - bottom) return difference on the
                     BASE sample (not the overlapping 10 splits), for one-stage / two-stage / LM.
  #9 turnover+cost : average quarterly turnover of one-stage top-10, and the spread at
                     transaction costs of 0 / 10 / 20 bps.

Writes data/v2/analysis/reviewer_recompute.json + prints a summary. $0 (no API).
"""
import sys, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from pathlib import Path
import json
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from scripts.run_backtest import load_universe, load_all_prices
from scripts.factor_attribution import load_factors
from src.backtesting.engine import generate_rebalance_dates, load_views_summary, run_backtest
from src.backtesting.metrics import sharpe_ratio

FACS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
SOURCES = {"one-stage": "data/v2/views_singlestage/views_summary.csv",
           "two-stage": "data/v2/views_v2/views_summary.csv",
           "LM": "data/v2/views_lm/views_summary.csv"}


def bt(prices, vdf, tickers, dates, strat, tc=10.0):
    return run_backtest(prices=prices, views_df=vdf, tickers=tickers, rebalance_dates=dates,
                        strategy=strat, transaction_cost_bps=tc)


def alpha(dr, fac):
    common = dr.index.intersection(fac.index)
    y = dr.loc[common] - fac.loc[common, "RF"]; X = sm.add_constant(fac.loc[common, FACS])
    m = sm.OLS(y, X).fit()
    return m.params["const"] * 252 * 100, m.tvalues["const"]


def avg_turnover(res):
    wh = res["weights_history"]
    tot, n = 0.0, 0
    prev = None
    for _, wd in wh:
        w = pd.Series(wd)
        if prev is not None:
            allk = prev.index.union(w.index)
            tot += np.abs(w.reindex(allk, fill_value=0) - prev.reindex(allk, fill_value=0)).sum()
            n += 1
        prev = w
    return tot / n if n else float("nan")  # round-trip turnover per rebalance


def main():
    uni = load_universe("config/universe.yaml")
    allt = uni["core"] + uni["holdout"]
    ap = load_all_prices(Path("data/raw"), allt); ap = ap[ap.index <= pd.Timestamp(2026, 3, 31)]
    core = [t for t in uni["core"] if t in ap.columns]
    hold = [t for t in uni["holdout"] if t in ap.columns]
    dates = generate_rebalance_dates(2006, 2025, "quarterly")
    fac = load_factors()
    V = {k: load_views_summary(Path(v)) for k, v in SOURCES.items()}
    out = {}

    # ---- #6 holdout OOS vs core: spread + one-stage alpha
    print("=== #6  top/bottom/spread by universe (Sharpe, rf=2%, net 10bps) ===")
    print(f"{'source':10s} {'universe':9s} {'top':>7s} {'bottom':>7s} {'spread':>8s}")
    six = {}
    for name in ("one-stage", "two-stage", "LM"):
        for uni_name, tk in (("core-50", core), ("holdout-50", hold)):
            t = sharpe_ratio(bt(ap[tk], V[name], tk, dates, "top_n_views")["daily_returns"].dropna())
            b = sharpe_ratio(bt(ap[tk], V[name], tk, dates, "bottom_n_views")["daily_returns"].dropna())
            six[f"{name}/{uni_name}"] = {"top": round(t, 3), "bottom": round(b, 3), "spread": round(t - b, 3)}
            print(f"{name:10s} {uni_name:9s} {t:7.3f} {b:7.3f} {t-b:8.3f}")
    out["p6_holdout"] = six
    print("\n--- one-stage alpha (FF5+Mom) by universe ---")
    a_out = {}
    for uni_name, tk in (("core-50", core), ("holdout-50", hold)):
        at, tt = alpha(bt(ap[tk], V["one-stage"], tk, dates, "top_n_views")["daily_returns"].dropna(), fac)
        ab, tb = alpha(bt(ap[tk], V["one-stage"], tk, dates, "bottom_n_views")["daily_returns"].dropna(), fac)
        a_out[uni_name] = {"top_alpha_pct": round(at, 2), "top_t": round(tt, 2),
                           "bottom_alpha_pct": round(ab, 2), "bottom_t": round(tb, 2)}
        print(f"  {uni_name}: top alpha {at:+.2f}% (t={tt:.2f}) | bottom {ab:+.2f}% (t={tb:.2f})")
    out["p6_alpha"] = a_out

    # ---- #7 formal paired t-test on daily (top - bottom) difference, base sample = core-50
    print("\n=== #7  formal test: paired t-test on daily (top - bottom) return, core-50 ===")
    seven = {}
    for name in ("one-stage", "two-stage", "LM"):
        rt = bt(ap[core], V[name], core, dates, "top_n_views")["daily_returns"].dropna()
        rb = bt(ap[core], V[name], core, dates, "bottom_n_views")["daily_returns"].dropna()
        common = rt.index.intersection(rb.index)
        diff = rt.loc[common] - rb.loc[common]
        t_stat, p = stats.ttest_1samp(diff, 0)
        seven[name] = {"mean_daily_diff_bps": round(diff.mean() * 1e4, 3),
                       "t_stat": round(float(t_stat), 2), "p_value": round(float(p), 4),
                       "ann_diff_pct": round(((1 + diff.mean()) ** 252 - 1) * 100, 2)}
        print(f"  {name:10s}: mean diff {diff.mean()*1e4:+.2f} bps/day | t={t_stat:.2f} p={p:.4f}")
    out["p7_ttest"] = seven

    # ---- #9 turnover + cost sensitivity (one-stage top-10, core-50)
    print("\n=== #9  one-stage top-10 turnover & cost sensitivity (core-50) ===")
    res10 = bt(ap[core], V["one-stage"], core, dates, "top_n_views", tc=10.0)
    turn = avg_turnover(res10)
    print(f"  avg round-trip turnover / quarter: {turn:.2f}  (=2.0 means full replacement)")
    nine = {"avg_roundtrip_turnover_per_quarter": round(turn, 2), "cost_sensitivity": {}}
    for tc in (0.0, 10.0, 20.0):
        t = sharpe_ratio(bt(ap[core], V["one-stage"], core, dates, "top_n_views", tc=tc)["daily_returns"].dropna())
        b = sharpe_ratio(bt(ap[core], V["one-stage"], core, dates, "bottom_n_views", tc=tc)["daily_returns"].dropna())
        at, tt = alpha(bt(ap[core], V["one-stage"], core, dates, "top_n_views", tc=tc)["daily_returns"].dropna(), fac)
        nine["cost_sensitivity"][f"{int(tc)}bps"] = {"top": round(t, 3), "spread": round(t - b, 3),
                                                     "top_alpha_pct": round(at, 2), "t": round(tt, 2)}
        print(f"  tc={int(tc):>2}bps: top Sharpe {t:.3f} | spread {t-b:+.3f} | alpha {at:+.2f}% (t={tt:.2f})")
    out["p9_turnover_cost"] = nine

    Path("data/v2/analysis/reviewer_recompute.json").write_text(json.dumps(out, indent=2))
    print("\nsaved data/v2/analysis/reviewer_recompute.json")


if __name__ == "__main__":
    main()
