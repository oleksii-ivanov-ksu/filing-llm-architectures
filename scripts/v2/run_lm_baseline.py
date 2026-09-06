#!/usr/bin/env python3
"""
C3 -- Loughran-McDonald scalar-sentiment baseline (RQ3).

Same documents, same view-trigger schedule, same downstream BL backtest as the
method. The ONLY difference is the signal: instead of multi-dimensional LLM
extraction we compute a single LM sentiment scalar per filing:

    tone = (#pos - #neg) / (#pos + #neg)          # Loughran-McDonald word lists

then cross-sectionally z-score the tone across all companies on each view date
and map it to view_bps (same clamp as the method). confidence = "medium".

This isolates "multi-dimensional structured extraction" vs "one sentiment
scalar of the same text" -- both fed to the identical BL + tilt backtest.

Cost: $0 (no API). Dictionary: pysentiment2 (bundled LM word lists).

Output:
    data/v2/views_lm/{TICKER}/view_{DATE}_{PERIOD}.json
    data/v2/views_lm/views_summary.csv   (backtest-compatible)

Usage:
    python scripts/v2/run_lm_baseline.py --dry-run
    python scripts/v2/run_lm_baseline.py --workers 8
"""
import os, sys, csv, json, argparse, logging, time, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.reasoning.input_builder import InputBuilder
from src.extraction.parser import SECFilingParser
from src.extraction.chunker import TokenChunker

import pysentiment2 as ps

# Per-process singletons (ProcessPoolExecutor: real parallelism, no GIL contention
# on pysentiment2's timeout-guarded regex tokenizer). Built once per worker process.
_WORKER = {}


def _init_worker():
    _WORKER["ib"] = InputBuilder(extracted_dir=EXTRACTED_DIR, raw_dir=RAW_DIR,
                                 constituents_path=Path("data/constituents.csv"))
    _WORKER["lm"] = ps.LM()
    _WORKER["chunker"] = TokenChunker(model="gpt-4o-mini")


def _score_one(item):
    """Stage 1 (per worker process): raw LM tone for one filing, or None."""
    t, f = item
    ib, lm, chunker = _WORKER["ib"], _WORKER["lm"], _WORKER["chunker"]
    fdate, ftype = f.get("filing_date"), f.get("filing_type")
    path = raw_path(t, ftype, fdate)
    if not path:
        return None
    ri = ib.build_input_for_filing(t, f)
    if ri is None:
        return None
    html = open(path, "r", encoding="utf-8", errors="ignore").read()
    sections = SECFilingParser(html, filename=str(path)).extract_all_sections()
    if not sections:
        return None
    text = "\n\n".join(chunker.chunk_sections(sections, max_total_tokens=80000))
    if not text:
        return None
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    pos, neg = score["Positive"], score["Negative"]
    tone = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    vd = getattr(ri, "view_date", None)
    es = getattr(ri, "earnings_surprise", None)
    return {
        "ticker": t, "view_date": str(vd) if vd is not None else None,
        "fiscal_period": getattr(ri, "fiscal_period", None), "filing_type": ftype,
        "filing_date": fdate, "tone": tone, "pos": pos, "neg": neg,
        "eps_surprise_pct": getattr(es, "eps_surprise_pct", None) if es else None,
    }

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/v2/views_lm")
EXTRACTED_DIR = Path("data/v2/extracted_v2")   # used only to enumerate triggers + quant
RAW_DIR = Path("data/raw")
BPS_SCALE = 100      # view_bps = clip(z * BPS_SCALE, -300, 300); tilt re-standardizes anyway
BPS_CLAMP = 300


def raw_path(ticker, filing_type, filing_date):
    hits = glob.glob(f"data/raw/{ticker}/filings/{filing_type}_{filing_date}*.html")
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="Loughran-McDonald scalar baseline")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--core", action="store_true")
    g.add_argument("--holdout", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    uni = yaml.safe_load(open("config/universe.yaml"))["universe"]
    if args.core:
        tickers = uni["core"]
    elif args.holdout:
        tickers = uni["holdout"]
    else:
        tickers = uni["core"] + uni["holdout"]

    ib = InputBuilder(extracted_dir=EXTRACTED_DIR, raw_dir=RAW_DIR,
                      constituents_path=Path("data/constituents.csv"))

    work = []
    for t in tickers:
        for f in ib.get_all_filings(t, 2006, 2025):
            work.append((t, f))
    if args.limit:
        work = work[:args.limit]

    print(f"tickers: {len(tickers)} | filings (views): {len(work)} | Output: {OUT_DIR}/")
    if args.dry_run:
        print("Cost: $0 (LM dictionary, no API)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    recs = []
    fail = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(_score_one, it): it for it in work}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:
                r = None; fail += 1
                logger.error(f"  {futs[fut][0][0]}: {e}")
            if r:
                recs.append(r)
            if i % 500 == 0:
                logger.info(f"PROGRESS {i}/{len(work)} | scored {len(recs)} fail {fail} | "
                            f"~{i/max(1e-9,time.time()-t0)*60:.0f}/min")

    # Stage 2: cross-sectional z-score of tone per view_date -> view_bps
    df = pd.DataFrame(recs)
    df["view_date"] = pd.to_datetime(df["view_date"])
    def zscore(g):
        s = g["tone"]
        sd = s.std(ddof=0)
        g["z"] = 0.0 if sd == 0 or np.isnan(sd) else (s - s.mean()) / sd
        return g
    df = df.groupby("view_date", group_keys=False).apply(zscore)
    df["view_bps"] = (df["z"] * BPS_SCALE).clip(-BPS_CLAMP, BPS_CLAMP).round().astype(int)
    df["confidence"] = "medium"
    df["sentiment"] = np.where(df["tone"] > 0, "positive",
                               np.where(df["tone"] < 0, "negative", "neutral"))

    # Write per-ticker JSON + backtest-compatible summary CSV
    for _, r in df.iterrows():
        tdir = OUT_DIR / r["ticker"]
        tdir.mkdir(parents=True, exist_ok=True)
        vd = r["view_date"].date().isoformat()
        out = tdir / f"view_{vd}_{r['fiscal_period'] or 'NA'}.json".replace(" ", "_")
        out.write_text(json.dumps({
            "ticker": r["ticker"], "view_date": vd, "fiscal_period": r["fiscal_period"],
            "filing_type": r["filing_type"], "filing_date": r["filing_date"],
            "view_bps": int(r["view_bps"]), "confidence": "medium",
            "lm_tone": round(float(r["tone"]), 4), "lm_pos": int(r["pos"]), "lm_neg": int(r["neg"]),
            "eps_surprise_pct": r["eps_surprise_pct"],
        }, indent=2, ensure_ascii=False))

    csv_path = OUT_DIR / "views_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "view_date", "fiscal_period", "filing_type",
                    "view_bps", "confidence", "sentiment", "eps_surprise"])
        for _, r in df.iterrows():
            w.writerow([r["ticker"], r["view_date"].date().isoformat(), r["fiscal_period"],
                        r["filing_type"], int(r["view_bps"]), "medium", r["sentiment"],
                        r["eps_surprise_pct"]])

    logger.info(f"DONE. views={len(df)} | tone mean={df['tone'].mean():.3f} "
                f"std={df['tone'].std():.3f} | -> {csv_path}")


if __name__ == "__main__":
    main()
