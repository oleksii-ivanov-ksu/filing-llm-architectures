"""
Factor Attribution: Fama-French 5 + Momentum regression.

Downloads FF5 + Momentum daily factors from Kenneth French's website,
aligns with our backtest daily returns, and runs OLS regressions
to estimate alpha after controlling for standard risk factors.

Usage:
    python scripts/factor_attribution.py
    python scripts/factor_attribution.py --subset holdout
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm


# URLs for Fama-French factor data (Kenneth French's website)
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")


def load_ff_csv(url: str) -> pd.DataFrame:
    """Download and parse a Fama-French CSV from a zip file."""
    print(f"Downloading {url.split('/')[-1]}...", flush=True)
    resp = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    fname = z.namelist()[0]
    raw = z.read(fname).decode("utf-8")

    # Find where numeric data starts
    lines = raw.split("\n")
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and line.strip()[0].isdigit():
            data_start = i
            break

    # Read data lines until end or non-numeric line
    data_lines = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or not line[0].isdigit():
            break
        data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
    return df


def load_factors() -> pd.DataFrame:
    """Download and merge FF5 + Momentum factors."""
    # FF5
    ff5_raw = load_ff_csv(FF5_URL)
    ff5_raw.columns = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    ff5_raw["date"] = pd.to_datetime(ff5_raw["date"].astype(str), format="%Y%m%d")
    ff5_raw = ff5_raw.set_index("date")
    ff5_raw = ff5_raw / 100  # percent to decimal

    # Momentum
    mom_raw = load_ff_csv(MOM_URL)
    mom_raw.columns = ["date", "Mom"]
    mom_raw["date"] = pd.to_datetime(mom_raw["date"].astype(str), format="%Y%m%d")
    mom_raw = mom_raw.set_index("date")
    mom_raw = mom_raw / 100

    # Merge
    factors = ff5_raw.join(mom_raw, how="inner")
    print(f"Factors loaded: {factors.shape[0]} days, {factors.index[0].date()} to {factors.index[-1].date()}")
    return factors


def load_returns(subset: str = "core") -> pd.DataFrame:
    """Load daily returns from backtest results."""
    path = os.path.join(RESULTS_DIR, subset, "daily_returns.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    ret = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Returns loaded ({subset}): {ret.shape[0]} days, {len(ret.columns)} strategies")
    return ret


def run_regression(excess_returns: pd.Series, factors: pd.DataFrame,
                   factor_cols: list[str], name: str) -> dict:
    """Run OLS regression and return results."""
    X = sm.add_constant(factors[factor_cols])
    model = sm.OLS(excess_returns, X).fit()

    alpha_daily = model.params["const"]
    alpha_annual = alpha_daily * 252 * 100  # annualized percentage

    result = {
        "alpha_daily": alpha_daily,
        "alpha_annual_pct": alpha_annual,
        "t_stat": model.tvalues["const"],
        "p_value": model.pvalues["const"],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
        "factor_loadings": {},
    }

    for f in factor_cols:
        result["factor_loadings"][f] = {
            "beta": model.params[f],
            "t_stat": model.tvalues[f],
            "p_value": model.pvalues[f],
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Factor attribution analysis")
    parser.add_argument("--subset", default="core", choices=["core", "holdout"],
                        help="Which portfolio subset to analyze")
    args = parser.parse_args()

    # Load data
    factors = load_factors()
    returns = load_returns(args.subset)

    # Align dates
    common = returns.index.intersection(factors.index)
    print(f"Common trading days: {len(common)}")
    ret_a = returns.loc[common]
    fac_a = factors.loc[common]
    rf = fac_a["RF"]

    # Strategy name mapping
    strategies = {
        "BL-Filing+": "bl_llm_tilt_pos",
        "BL-Filing": "bl_llm_tilt",
        "Equal Weight": "equal_weight",
        "MVO": "mvo",
    }

    # Optional strategies
    if "random_views" in ret_a.columns:
        strategies["Random Views"] = "random_views"
    if "bl_llm" in ret_a.columns:
        strategies["BL-Filing MVO"] = "bl_llm"

    factor_sets = {
        "CAPM": ["Mkt-RF"],
        "FF3": ["Mkt-RF", "SMB", "HML"],
        "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        "FF5+Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    }

    # Run regressions
    print("\n" + "=" * 80)
    print(f"FACTOR ATTRIBUTION — {args.subset.upper()} PORTFOLIO")
    print("=" * 80)

    all_results = {}

    for name, col in strategies.items():
        if col not in ret_a.columns:
            continue

        excess = ret_a[col] - rf
        all_results[name] = {}

        print(f"\n--- {name} ---")
        print(f"{'Model':12s} {'Alpha(ann%)':>11s} {'t-stat':>7s} {'p-val':>7s} {'R²':>5s}")

        for model_name, factor_cols in factor_sets.items():
            result = run_regression(excess, fac_a, factor_cols, f"{name}_{model_name}")
            all_results[name][model_name] = result

            a = result["alpha_annual_pct"]
            t = result["t_stat"]
            p = result["p_value"]
            r2 = result["r_squared"]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"{model_name:12s} {a:>+10.2f}% {t:>7.2f} {p:>7.4f} {r2:>5.3f} {sig}")

        # Print FF5+Mom factor loadings
        ff5m = all_results[name]["FF5+Mom"]
        print(f"  FF5+Mom factor loadings:")
        for f, vals in ff5m["factor_loadings"].items():
            print(f"    {f:8s}: β={vals['beta']:>+7.3f} (t={vals['t_stat']:>6.2f})")

    # Long-short: BL-Filing+ minus Equal Weight
    if "bl_llm_tilt_pos" in ret_a.columns and "equal_weight" in ret_a.columns:
        print(f"\n{'=' * 80}")
        print("LONG-SHORT: BL-Filing+ minus Equal Weight")
        print("=" * 80)

        ls_ret = ret_a["bl_llm_tilt_pos"] - ret_a["equal_weight"]
        result = run_regression(ls_ret, fac_a, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"], "long_short")
        all_results["Long-Short (BL-Filing+ - EW)"] = {"FF5+Mom": result}

        a = result["alpha_annual_pct"]
        t = result["t_stat"]
        p = result["p_value"]
        print(f"Alpha: {a:>+.2f}%, t={t:.2f}, p={p:.4f}")
        for f, vals in result["factor_loadings"].items():
            print(f"  {f:8s}: β={vals['beta']:>+7.3f} (t={vals['t_stat']:>6.2f})")

    # Save results
    output_dir = os.path.join(RESULTS_DIR, "robustness")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"factor_attribution_{args.subset}.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
