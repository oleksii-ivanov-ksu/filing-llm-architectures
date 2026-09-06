#!/usr/bin/env python3
"""
Reviewer point 8 — reproducibility of the WINNING (one-stage) architecture.

The LLM's only stochastic output is view_bps; everything downstream (ranking ->
top-10/bottom-10 -> backtest -> spread/alpha) is a deterministic function of it.
So it suffices to show view_bps is stable across independent runs at T=0.1.

We run the one-stage generator R times on the 90 gold-standard documents (~1% of
the corpus) and report, per document and in aggregate:
  - sigma(view_bps)         : std of view_bps across the R runs (magnitude stability)
  - sign agreement          : share of docs where all R runs give the same sign
  - Spearman rank corr       : mean pairwise rank correlation of view_bps across runs
                               (this is what the top/bottom selection actually depends on)
  - confidence agreement     : share of docs where all R runs give the same confidence

Output: data/v2/analysis/singlestage_reproducibility.json  (+ printed summary)

Usage:
    python scripts/v2/run_singlestage_repro.py --runs 3 --workers 6
"""
import os, sys, json, argparse, logging, time, glob, itertools
from pathlib import Path
from statistics import pstdev, mean

import yaml
import numpy as np
from scipy.stats import spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.reasoning.input_builder import InputBuilder
from src.singlestage import SingleStageGenerator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

MANIFEST = Path("data/v2/gold_standard/manifest.json")
EXTRACTED_DIR = Path("data/v2/extracted_v2")
RAW_DIR = Path("data/raw")
OUT = Path("data/v2/analysis/singlestage_reproducibility.json")


def raw_path(ticker, filing_type, filing_date):
    hits = glob.glob(f"data/raw/{ticker}/filings/{filing_type}_{filing_date}*.html")
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml"))
    oc = cfg["openai"]
    model = oc["model"]
    docs = json.load(open(MANIFEST))["documents"]
    if args.limit:
        docs = docs[:args.limit]

    per_doc_cost = 0.016
    print(f"model: {model} | docs: {len(docs)} | runs: {args.runs} | "
          f"calls: {len(docs)*args.runs} | est. ${len(docs)*args.runs*per_doc_cost:.1f}")
    if args.dry_run:
        return

    ib = InputBuilder(extracted_dir=EXTRACTED_DIR, raw_dir=RAW_DIR,
                      constituents_path=Path("data/constituents.csv"))
    gen = SingleStageGenerator(api_key=oc["api_key"], model=model,
                               base_url=oc.get("base_url"), max_tokens=1500)

    # build the fixed work list once (ticker, filing dict, quant)
    work = []
    for d in docs:
        p = Path(d["path"])
        fdate = p.stem.split("_")[-1]
        # find the matching filing dict for quant context
        filing = {"filing_date": fdate, "filing_type": d["form"], "fiscal_period": None}
        ri = ib.build_input_for_filing(d["ticker"], filing)
        q = {"sector": getattr(ri, "sector", None) if ri else None,
             "view_date": str(getattr(ri, "view_date", "")) if ri else None,
             "fiscal_period": getattr(ri, "fiscal_period", None) if ri else None,
             "eps_surprise_pct": None, "revenue_surprise_pct": None, "dispersion": None,
             "price_return_1m": None, "price_return_3m": None, "price_volatility_60d": None}
        if ri and getattr(ri, "earnings_surprise", None):
            es = ri.earnings_surprise
            q["eps_surprise_pct"] = getattr(es, "eps_surprise_pct", None)
        work.append((d["id"], d["ticker"], str(p), d["form"], fdate, q))

    t0 = time.time()
    # runs[r][doc_id] = view_bps ; also collect sign & confidence
    vb = {d[0]: [] for d in work}
    cf = {d[0]: [] for d in work}
    for r in range(args.runs):
        done = 0
        for did, tk, path, form, fdate, q in work:
            try:
                view, _ = gen.generate(tk, path, form, fdate, q)
                vb[did].append(view.view_bps if view else None)
                cf[did].append(view.confidence if view else None)
            except Exception as e:
                vb[did].append(None); cf[did].append(None)
                logging.warning(f"{did}: {e}")
            done += 1
            if done % 30 == 0:
                print(f"  run {r+1}/{args.runs}: {done}/{len(work)} "
                      f"({done/max(1e-9,time.time()-t0)*60:.0f}/min cum)", flush=True)
        print(f"run {r+1} done", flush=True)

    # aggregate
    sigmas, sign_ok, conf_ok = [], [], []
    per_run_vectors = [[] for _ in range(args.runs)]  # for cross-run Spearman
    ids_full = []
    for did in vb:
        vals = vb[did]
        if any(v is None for v in vals):
            continue
        sigmas.append(pstdev(vals))
        sign_ok.append(1.0 if len({np.sign(v) for v in vals}) == 1 else 0.0)
        conf_ok.append(1.0 if len(set(cf[did])) == 1 else 0.0)
        ids_full.append(did)
        for r in range(args.runs):
            per_run_vectors[r].append(vals[r])

    # mean pairwise Spearman across runs
    spearmans = []
    for a, b in itertools.combinations(range(args.runs), 2):
        if len(per_run_vectors[a]) > 2:
            rho = spearmanr(per_run_vectors[a], per_run_vectors[b]).correlation
            if rho == rho:  # not nan
                spearmans.append(rho)

    summary = {
        "model": model, "runs": args.runs, "n_docs_compared": len(sigmas),
        "temperature": 0.3,  # SingleStageGenerator default; see note
        "view_bps_sigma_mean": round(mean(sigmas), 2) if sigmas else None,
        "view_bps_sigma_median": round(float(np.median(sigmas)), 2) if sigmas else None,
        "sign_agreement_rate": round(mean(sign_ok), 4) if sign_ok else None,
        "confidence_agreement_rate": round(mean(conf_ok), 4) if conf_ok else None,
        "mean_pairwise_spearman": round(mean(spearmans), 4) if spearmans else None,
        "cost_est_usd": round(len(work) * args.runs * per_doc_cost, 2),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print("\n=== one-stage reproducibility ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
