#!/usr/bin/env python3
"""
Build the gold-standard manifest (experiment D): a stratified sample of 90
filings (~1% of the corpus) for manual annotation and model selection.

Stratification: proportional by period (2006-2010 / 2011-2015 / 2016-2020 /
2021-2025), form ratio close to the corpus (~22% 10-K), all 11 GICS sectors
covered, distinct tickers preferred for diversity.

Tiers (non-overlapping):
  - tier1a_tuning: 5  docs (blind, used to tune grounding prompt/threshold)
  - tier1b_eval:  15  docs (blind, final recall + Cohen's kappa)
  - tier2_pool:   70  docs (pooling-based annotation, recall breakdowns)

Deterministic (fixed seed) so the manifest is reproducible and recorded.

Usage:
    python claude_helpers/build_gold_manifest.py
Output:
    data/v2/gold_standard/manifest.json
"""
import os
import re
import json
import glob
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

SEED = 20260829
N_TOTAL = 90
N_TUNING = 5
N_EVAL = 15
# tier2 = remainder (70)

PERIODS = [("2006-2010", 2006, 2010), ("2011-2015", 2011, 2015),
           ("2016-2020", 2016, 2020), ("2021-2025", 2021, 2025)]


def period_of(year: int):
    for name, lo, hi in PERIODS:
        if lo <= year <= hi:
            return name
    return None


def main():
    random.seed(SEED)

    universe = yaml.safe_load(open("config/universe.yaml"))
    tickers = set(universe["universe"]["core"] + universe["universe"]["holdout"])

    cons = pd.read_csv("data/constituents.csv")
    sector_map = dict(zip(cons["Symbol"], cons["GICS Sector"]))

    # Enumerate eligible filings within the study period 2006-2025
    rows = []
    for f in glob.glob("data/raw/*/filings/10-*_*.html"):
        ticker = f.split("/")[2]
        if ticker not in tickers:
            continue
        m = re.match(r"(10-[KQ])_(\d{4})", os.path.basename(f))
        if not m:
            continue
        form, year = m.group(1), int(m.group(2))
        p = period_of(year)
        if p is None:
            continue
        rows.append({
            "ticker": ticker, "form": form, "year": year, "period": p,
            "sector": sector_map.get(ticker, "Unknown"), "path": f,
        })

    # Proportional allocation by period
    by_period = defaultdict(list)
    for r in rows:
        by_period[r["period"]].append(r)
    total = len(rows)
    alloc = {p: round(N_TOTAL * len(by_period[p]) / total) for p, _, _ in PERIODS}
    # fix rounding drift to exactly N_TOTAL
    drift = N_TOTAL - sum(alloc.values())
    alloc[PERIODS[0][0]] += drift

    target_10k_ratio = 0.22
    sample, used_tickers = [], set()

    for period_name, _, _ in PERIODS:
        pool = by_period[period_name][:]
        random.shuffle(pool)
        want = alloc[period_name]
        want_10k = round(want * target_10k_ratio)

        picked = []
        # prefer distinct tickers and all sectors; fill 10-K quota first
        def take(pred, limit):
            nonlocal picked
            for r in pool:
                if len(picked) >= limit:
                    break
                if r in picked:
                    continue
                if pred(r) and r["ticker"] not in used_tickers:
                    picked.append(r)
                    used_tickers.add(r["ticker"])

        take(lambda r: r["form"] == "10-K", want_10k)
        take(lambda r: r["form"] == "10-Q", want)
        # if distinct-ticker constraint left us short, relax it
        if len(picked) < want:
            for r in pool:
                if len(picked) >= want:
                    break
                if r not in picked:
                    picked.append(r)
        sample.extend(picked[:want])

    random.shuffle(sample)
    assert len(sample) == N_TOTAL, f"got {len(sample)} != {N_TOTAL}"

    # Tier assignment, stratified by period (round-robin over shuffled sample)
    sample.sort(key=lambda r: r["period"])
    tiers = {"tier1a_tuning": [], "tier1b_eval": [], "tier2_pool": []}
    for i, r in enumerate(sample):
        if i % 18 == 0 and len(tiers["tier1a_tuning"]) < N_TUNING:
            r["tier"] = "tier1a_tuning"
        elif i % 6 < 1 and len(tiers["tier1b_eval"]) < N_EVAL:
            r["tier"] = "tier1b_eval"
        else:
            r["tier"] = "tier2_pool"
        tiers[r["tier"]].append(r)

    # backfill tiers to exact sizes if round-robin drifted
    def rebalance():
        flat = tiers["tier1a_tuning"] + tiers["tier1b_eval"] + tiers["tier2_pool"]
        for t in tiers:
            tiers[t] = []
        random.shuffle(flat)
        for r in flat[:N_TUNING]:
            r["tier"] = "tier1a_tuning"; tiers["tier1a_tuning"].append(r)
        for r in flat[N_TUNING:N_TUNING + N_EVAL]:
            r["tier"] = "tier1b_eval"; tiers["tier1b_eval"].append(r)
        for r in flat[N_TUNING + N_EVAL:]:
            r["tier"] = "tier2_pool"; tiers["tier2_pool"].append(r)
    rebalance()

    manifest = {
        "seed": SEED,
        "n_total": N_TOTAL,
        "tiers": {k: len(v) for k, v in tiers.items()},
        "stratification": {"by_period": alloc, "target_10k_ratio": target_10k_ratio},
        "documents": [
            {"id": f"{r['ticker']}_{r['form']}_{r['year']}", **{k: r[k] for k in
             ["ticker", "form", "year", "period", "sector", "tier", "path"]}}
            for r in sorted(sample, key=lambda x: (x["tier"], x["ticker"]))
        ],
    }

    out = Path("data/v2/gold_standard")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Report coverage
    from collections import Counter
    print(f"Wrote {out/'manifest.json'} | {N_TOTAL} docs, seed={SEED}")
    print("Tiers:", manifest["tiers"])
    print("By period:", dict(Counter(r["period"] for r in sample)))
    print("By form:", dict(Counter(r["form"] for r in sample)))
    print("Sectors covered:", len(set(r["sector"] for r in sample)), "/ 11")
    print("Distinct tickers:", len(set(r["ticker"] for r in sample)))


if __name__ == "__main__":
    main()
