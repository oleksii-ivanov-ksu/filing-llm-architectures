#!/usr/bin/env python3
"""
Placebo / shuffle ensemble (experiment C1 + C2).

Runs a stochastic strategy (random_views or shuffled_views) N times with
different seeds and reports the Sharpe-ratio distribution. Answers reviewer
concern that the placebo was a single realization, and provides the
content-vs-mechanism decomposition (shuffled_views).

Usage:
    python scripts/run_placebo_ensemble.py --core --strategy random_views --n 100
    python scripts/run_placebo_ensemble.py --core --strategy shuffled_views --n 100

Output:
    data/results/{subset}/ensemble_{strategy}.json
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtesting.engine import generate_rebalance_dates, load_views_summary, run_backtest
from src.backtesting.metrics import compute_all_metrics

# Reuse the exact data-loading helpers from the v1 backtest runner
from scripts.run_backtest import load_universe, load_all_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sharpe_of(daily_returns: pd.Series) -> float:
    m = compute_all_metrics(daily_returns)
    return float(m["sharpe_ratio"])


def main():
    p = argparse.ArgumentParser(description="Placebo/shuffle ensemble over seeds")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--core", action="store_true")
    g.add_argument("--holdout", action="store_true")
    g.add_argument("--all", action="store_true")
    p.add_argument("--strategy", default="random_views",
                   choices=["random_views", "shuffled_views"])
    p.add_argument("--n", type=int, default=100, help="Number of seeds (default 100)")
    p.add_argument("--frequency", default="quarterly",
                   choices=["monthly_active", "biweekly_active", "quarterly"])
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--tc-bps", type=float, default=10.0)
    p.add_argument("--views-csv", type=str, default="data/v2/views_v2/views_summary.csv",
                   help="Views summary CSV (default: v2 gemini views)")
    args = p.parse_args()

    base = Path(".")
    raw_dir = base / "data" / "raw"
    views_csv = base / args.views_csv

    universe = load_universe("config/universe.yaml")
    if args.core:
        tickers, subset = universe["core"], "core"
    elif args.holdout:
        tickers, subset = universe["holdout"], "holdout"
    else:
        tickers, subset = universe["core"] + universe["holdout"], "all"

    start_year, end_year = 2006, 2025
    prices = load_all_prices(raw_dir, tickers)
    prices = prices[prices.index <= pd.Timestamp(end_year + 1, 3, 31)]
    tickers = [t for t in tickers if t in prices.columns]
    views_df = load_views_summary(views_csv)
    rebalance_dates = generate_rebalance_dates(start_year, end_year, args.frequency)

    logger.info(f"Ensemble: {args.strategy} x {args.n} seeds | {subset} ({len(tickers)} tickers)")

    sharpes = []
    for seed in range(args.n):
        result = run_backtest(
            prices=prices, views_df=views_df, tickers=tickers,
            rebalance_dates=rebalance_dates, strategy=args.strategy,
            tau=args.tau, max_weight=args.max_weight,
            transaction_cost_bps=args.tc_bps, placebo_seed=seed,
        )
        s = sharpe_of(result["daily_returns"])
        sharpes.append(s)
        if (seed + 1) % 10 == 0:
            logger.info(f"  {seed + 1}/{args.n} done | running mean Sharpe={np.mean(sharpes):.4f}")

    arr = np.array(sharpes)
    summary = {
        "strategy": args.strategy,
        "subset": subset,
        "n_seeds": args.n,
        "sharpe_mean": float(arr.mean()),
        "sharpe_std": float(arr.std(ddof=1)),
        "sharpe_min": float(arr.min()),
        "sharpe_max": float(arr.max()),
        "sharpe_p05": float(np.percentile(arr, 5)),
        "sharpe_p95": float(np.percentile(arr, 95)),
        "all_sharpes": sharpes,
    }

    out_dir = base / "data" / "v2" / "results" / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ensemble_{args.strategy}.json"
    out_file.write_text(json.dumps(summary, indent=2))

    logger.info(f"Done. Sharpe {summary['sharpe_mean']:.4f} ± {summary['sharpe_std']:.4f} "
                f"[{summary['sharpe_min']:.4f}, {summary['sharpe_max']:.4f}] -> {out_file}")


if __name__ == "__main__":
    main()
