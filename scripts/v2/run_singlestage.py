#!/usr/bin/env python3
"""
One-stage baseline run (experiment B, RQ2): filing text (+ quant summary) -> view
in a single LLM call. Same inputs & study period as the two-stage pipeline, so the
downstream comparison isolates the architecture (one call vs extract-then-reason).

Output:
    data/v2/views_singlestage/{TICKER}/view_{DATE}_{PERIOD}.json
    data/v2/views_singlestage/views_summary.csv   (backtest-compatible)

Usage:
    python scripts/v2/run_singlestage.py --dry-run
    python scripts/v2/run_singlestage.py --limit 20 --workers 6      # pilot
    python scripts/v2/run_singlestage.py --core --workers 6          # core-50
    python scripts/v2/run_singlestage.py --workers 6                 # full corpus
"""
import os, sys, csv, json, argparse, logging, time, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.reasoning.input_builder import InputBuilder
from src.singlestage import SingleStageGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/v2/views_singlestage")
EXTRACTED_DIR = Path("data/v2/extracted_v2")   # used only to enumerate triggers + quant
RAW_DIR = Path("data/raw")


def raw_path(ticker, filing_type, filing_date):
    hits = glob.glob(f"data/raw/{ticker}/filings/{filing_type}_{filing_date}*.html")
    return hits[0] if hits else None


def quant_from_input(ri):
    """Pull the quantitative fields from a ReasonerInput into a flat dict."""
    es = getattr(ri, "earnings_surprise", None)
    vd = getattr(ri, "view_date", None)
    return {
        "sector": getattr(ri, "sector", None),
        "view_date": str(vd) if vd is not None else None,   # date -> str for JSON/CSV
        "fiscal_period": getattr(ri, "fiscal_period", None),
        "eps_surprise_pct": getattr(es, "eps_surprise_pct", None) if es else None,
        "revenue_surprise_pct": getattr(es, "revenue_surprise_pct", None) if es else None,
        "dispersion": getattr(es, "estimate_dispersion", None) if es else None,
        "price_return_1m": getattr(ri, "price_return_1m", None),
        "price_return_3m": getattr(ri, "price_return_3m", None),
        "price_volatility_60d": getattr(ri, "price_volatility_60d", None),
    }


def main():
    ap = argparse.ArgumentParser(description="One-stage view baseline")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--core", action="store_true")
    g.add_argument("--holdout", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml"))
    oc = cfg["openai"]
    model = args.model or oc["model"]
    uni = yaml.safe_load(open("config/universe.yaml"))["universe"]
    if args.core:
        tickers = uni["core"]
    elif args.holdout:
        tickers = uni["holdout"]
    else:
        tickers = uni["core"] + uni["holdout"]

    ib = InputBuilder(extracted_dir=EXTRACTED_DIR, raw_dir=RAW_DIR,
                      constituents_path=Path("data/constituents.csv"))

    # Build the work list: (ticker, filing_dict) for all view-triggering filings
    work = []
    for t in tickers:
        for f in ib.get_all_filings(t, 2006, 2025):
            work.append((t, f))
    if args.limit:
        work = work[:args.limit]

    print(f"Model: {model} | tickers: {len(tickers)} | filings (views): {len(work)}")
    print(f"Output: {OUT_DIR}/")
    if args.dry_run:
        est = len(work) * 0.016  # ~ two-stage extraction cost/doc (reads full filing)
        print(f"Estimated cost: ~${est:.0f}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = SingleStageGenerator(api_key=oc["api_key"], model=model,
                               base_url=oc.get("base_url"), max_tokens=1500)

    done = ok = fail = skip = 0
    t0 = time.time()

    def process(item):
        t, f = item
        fdate, ftype = f.get("filing_date"), f.get("filing_type")
        tdir = OUT_DIR / t
        out = tdir / f"view_{fdate}_{f.get('fiscal_period','NA')}.json".replace(" ", "_")
        if out.exists() and args.resume:
            return "skip", None
        path = raw_path(t, ftype, fdate)
        if not path:
            return "fail", None
        ri = ib.build_input_for_filing(t, f)
        if ri is None:
            return "fail", None
        q = quant_from_input(ri)
        view, ginfo = gen.generate(t, path, ftype, fdate, q)
        if view is None:
            return "fail", None
        rec = {
            "ticker": t, "view_date": q["view_date"], "fiscal_period": q["fiscal_period"],
            "filing_type": ftype, "filing_date": fdate,
            "view_bps": view.view_bps, "confidence": view.confidence,
            "reasoning": view.reasoning, "supporting_quotes": view.supporting_quotes,
            "grounding": ginfo, "eps_surprise_pct": q["eps_surprise_pct"],
        }
        tdir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        return "ok", rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, it): it for it in work}
        for fut in as_completed(futs):
            try:
                status, _ = fut.result()
            except Exception as e:
                status = "fail"
                logger.error(f"  {futs[fut][0]}: {e}")
            done += 1
            ok += status == "ok"; fail += status == "fail"; skip += status == "skip"
            if done % 50 == 0:
                rate = done / max(1e-9, time.time() - t0) * 60
                logger.info(f"PROGRESS {done}/{len(work)} | ok {ok}, fail {fail}, skip {skip} | "
                            f"~{rate:.0f}/min | in {gen.input_tokens:,} out {gen.output_tokens:,} tok")

    # Build backtest-compatible summary CSV from all saved views
    rows = []
    for f in OUT_DIR.glob("*/view_*.json"):
        d = json.loads(f.read_text())
        rows.append([d["ticker"], d["view_date"], d.get("fiscal_period"), d.get("filing_type"),
                     d["view_bps"], d["confidence"], "", d.get("eps_surprise_pct")])
    csv_path = OUT_DIR / "views_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "view_date", "fiscal_period", "filing_type",
                    "view_bps", "confidence", "sentiment", "eps_surprise"])
        w.writerows(rows)

    logger.info(f"DONE. ok={ok} fail={fail} skip={skip} | views={len(rows)} | "
                f"in {gen.input_tokens:,} / out {gen.output_tokens:,} tok -> {csv_path}")


if __name__ == "__main__":
    main()
