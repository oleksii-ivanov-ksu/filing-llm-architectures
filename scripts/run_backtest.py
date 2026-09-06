#!/usr/bin/env python3
"""
Run portfolio backtesting with Black-Litterman and baselines.

Usage:
    python scripts/run_backtest.py --core              # Core 50 stocks
    python scripts/run_backtest.py --holdout            # Holdout 50 stocks
    python scripts/run_backtest.py --all                # All 100 stocks
    python scripts/run_backtest.py --core --test        # Quick test (2006-2010)

Output:
    data/results/{subset}/daily_returns.csv
    data/results/{subset}/metrics.json
    data/results/{subset}/weights/
"""

import os
import sys
import json
import argparse
import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtesting.engine import (
    generate_rebalance_dates,
    load_views_summary,
    run_backtest,
)
from src.backtesting.metrics import (
    compute_all_metrics,
    compare_strategies,
    compute_turnover,
    bootstrap_sharpe_ci,
    test_sharpe_difference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGIES = ["bl_llm_tilt", "bl_llm_tilt_pos", "equal_weight", "mvo", "random_views"]


def load_universe(universe_path: str) -> dict:
    """Load universe config."""
    with open(universe_path) as f:
        universe = yaml.safe_load(f)
    u = universe["universe"]
    return {
        "core": u.get("core", []),
        "holdout": u.get("holdout", []),
    }


def load_all_prices(raw_dir: Path, tickers: list) -> pd.DataFrame:
    """Load and merge daily close prices for all tickers."""
    frames = {}
    for ticker in tickers:
        price_file = raw_dir / ticker / "prices.csv"
        if not price_file.exists():
            logger.warning(f"No prices for {ticker}")
            continue

        df = pd.read_csv(price_file, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        frames[ticker] = df["close"]

    prices = pd.DataFrame(frames)
    prices = prices.sort_index()

    # Forward fill small gaps (holidays in one exchange but not another)
    prices = prices.ffill(limit=5)

    logger.info(f"Loaded prices: {prices.shape[0]} days × {prices.shape[1]} tickers")
    return prices


def main():
    parser = argparse.ArgumentParser(description="Run portfolio backtesting")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--core", action="store_true", help="Core 50 stocks")
    group.add_argument("--holdout", action="store_true", help="Holdout 50 stocks")
    group.add_argument("--all", action="store_true", help="All 100 stocks")

    parser.add_argument("--test", action="store_true", help="Quick test (2006-2010)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        help=f"Strategies to run (default: {STRATEGIES})")
    parser.add_argument("--frequency", default="quarterly",
                        choices=["monthly_active", "biweekly_active", "quarterly"],
                        help="Rebalancing frequency (default: quarterly)")
    parser.add_argument("--tau", type=float, default=0.05, help="BL tau parameter")
    parser.add_argument("--max-weight", type=float, default=0.10, help="Max weight per stock")
    parser.add_argument("--tc-bps", type=float, default=10.0, help="Transaction cost (bps)")
    parser.add_argument("--views-csv", type=str, default="data/views/views_summary.csv",
                        help="Views summary CSV (use data/v2/views_v2/views_summary.csv for v2)")
    parser.add_argument("--results-dir", type=str, default="data/results",
                        help="Base output dir (use data/v2/results for v2)")
    args = parser.parse_args()

    # Paths
    base_dir = Path(".")
    raw_dir = base_dir / "data" / "raw"
    views_csv = base_dir / args.views_csv

    # Determine subset
    universe = load_universe("config/universe.yaml")
    if args.core:
        tickers = universe["core"]
        subset_name = "core"
    elif args.holdout:
        tickers = universe["holdout"]
        subset_name = "holdout"
    else:
        tickers = universe["core"] + universe["holdout"]
        subset_name = "all"

    # Test mode
    start_year = 2006
    end_year = 2010 if args.test else 2025

    logger.info(f"Subset: {subset_name} ({len(tickers)} tickers)")
    logger.info(f"Period: {start_year}-{end_year}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Frequency: {args.frequency}")

    # Output directory
    output_dir = base_dir / args.results_dir / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weights").mkdir(exist_ok=True)

    # Load data
    logger.info("Loading prices...")
    prices = load_all_prices(raw_dir, tickers)

    # Trim prices to study period (+ 1 quarter buffer for last rebalance holding)
    price_end = pd.Timestamp(end_year + 1, 3, 31)
    prices = prices[prices.index <= price_end]
    logger.info(f"Price range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Filter tickers that have prices
    valid_tickers = [t for t in tickers if t in prices.columns]
    if len(valid_tickers) < len(tickers):
        missing = set(tickers) - set(valid_tickers)
        logger.warning(f"Missing price data for: {missing}")
    tickers = valid_tickers

    logger.info("Loading views...")
    views_df = load_views_summary(views_csv)

    # Generate rebalance dates
    rebalance_dates = generate_rebalance_dates(start_year, end_year, args.frequency)
    logger.info(f"Rebalance dates: {len(rebalance_dates)}")

    # Run all strategies
    all_returns = {}

    for strategy in args.strategies:
        logger.info(f"Running strategy: {strategy}...")
        result = run_backtest(
            prices=prices,
            views_df=views_df,
            tickers=tickers,
            rebalance_dates=rebalance_dates,
            strategy=strategy,
            tau=args.tau,
            max_weight=args.max_weight,
            transaction_cost_bps=args.tc_bps,
        )
        all_returns[strategy] = result["daily_returns"]

        # Save weights
        if result["weights_history"]:
            weights_records = []
            for date, w_dict in result["weights_history"]:
                record = {"date": date.strftime("%Y-%m-%d")}
                record.update(w_dict)
                weights_records.append(record)
            weights_df = pd.DataFrame(weights_records)
            weights_df.to_csv(output_dir / "weights" / f"{strategy}_weights.csv", index=False)

    # Save daily returns
    returns_df = pd.DataFrame(all_returns)
    returns_df.index.name = "date"
    returns_df.to_csv(output_dir / "daily_returns.csv")
    logger.info(f"Saved daily returns: {output_dir / 'daily_returns.csv'}")

    # Compute and save metrics
    logger.info("Computing metrics...")
    metrics_df = compare_strategies(all_returns)

    # Add turnover info for strategies with weights
    for strategy in args.strategies:
        result_file = output_dir / "weights" / f"{strategy}_weights.csv"
        if result_file.exists():
            w_df = pd.read_csv(result_file)
            w_history = []
            for _, row in w_df.iterrows():
                date = pd.Timestamp(row["date"])
                w_dict = {col: row[col] for col in w_df.columns if col != "date"}
                w_history.append((date, w_dict))
            if len(w_history) > 1:
                turnover = compute_turnover(w_history)
                metrics_df.loc[strategy, "avg_turnover"] = turnover.mean()
                metrics_df.loc[strategy, "total_tc_bps"] = turnover.sum() * args.tc_bps

    # Bootstrap CI for BL-LLM Sharpe
    if "bl_llm" in all_returns:
        ci_low, ci_high = bootstrap_sharpe_ci(all_returns["bl_llm"])
        metrics_df.loc["bl_llm", "sharpe_ci_lower"] = ci_low
        metrics_df.loc["bl_llm", "sharpe_ci_upper"] = ci_high

        # Significance tests vs baselines
        for baseline in ["equal_weight", "mvo"]:
            if baseline in all_returns:
                test = test_sharpe_difference(all_returns["bl_llm"], all_returns[baseline])
                metrics_df.loc["bl_llm", f"p_value_vs_{baseline}"] = test["p_value"]

    # Save metrics
    metrics_dict = metrics_df.to_dict(orient="index")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    logger.info(f"Saved metrics: {output_dir / 'metrics.json'}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"BACKTEST RESULTS — {subset_name.upper()} ({len(tickers)} stocks, {start_year}-{end_year})")
    print(f"Frequency: {args.frequency}, τ={args.tau}, max_weight={args.max_weight}, TC={args.tc_bps}bps")
    print("=" * 80)

    # Format for display
    display = metrics_df[["annualized_return", "annualized_volatility", "sharpe_ratio",
                           "sortino_ratio", "max_drawdown", "total_return"]].copy()
    display.columns = ["Ann.Ret", "Ann.Vol", "Sharpe", "Sortino", "MaxDD", "Total Ret"]

    for col in ["Ann.Ret", "Ann.Vol", "MaxDD", "Total Ret"]:
        display[col] = display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    for col in ["Sharpe", "Sortino"]:
        display[col] = display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    print(display.to_string())
    print()

    # Significance
    if "bl_llm" in metrics_dict:
        bl = metrics_dict["bl_llm"]
        if "sharpe_ci_lower" in bl:
            print(f"BL-LLM Sharpe 95% CI: [{bl['sharpe_ci_lower']:.3f}, {bl['sharpe_ci_upper']:.3f}]")
        for baseline in ["equal_weight", "mvo"]:
            key = f"p_value_vs_{baseline}"
            if key in bl:
                sig = "***" if bl[key] < 0.01 else "**" if bl[key] < 0.05 else "*" if bl[key] < 0.10 else ""
                print(f"BL-LLM vs {baseline}: p={bl[key]:.4f} {sig}")

    print()
    logger.info("Done.")


if __name__ == "__main__":
    main()
