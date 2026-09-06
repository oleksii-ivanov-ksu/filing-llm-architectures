#!/usr/bin/env python3
"""
C4 -- extraction reproducibility (engineering stability, ~$5 API).

Runs schema-v2 extraction R times (default 3) at production temperature (T=0.1)
on the SAME fixed document set (the 90-doc gold-standard manifest) and measures
run-to-run agreement:

  - risk-category Jaccard      (mean pairwise over the set of RiskFactor.category)
  - sentiment agreement        (fraction of docs where all R runs give same sentiment)
  - guidance-direction agree.  (same, over Guidance.direction)
  - #risk_factors / #key_events dispersion (mean per-doc std across runs)
  - grounding-rate stability   (mean per-doc std of grounded fraction)

Output: data/v2/analysis/reproducibility.json  (+ printed summary)

Usage:
    python scripts/v2/run_reproducibility.py --dry-run
    python scripts/v2/run_reproducibility.py --runs 3 --workers 6
"""
import os, sys, json, argparse, logging, time, itertools
from pathlib import Path
from statistics import pstdev, mean

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.extraction.extractor import FilingExtractor

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

MANIFEST = Path("data/v2/gold_standard/manifest.json")
OUT = Path("data/v2/analysis/reproducibility.json")


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def facts_signature(ef):
    """Reduce an ExtractedFacts to the comparable fields."""
    if ef is None:
        return None
    g = ef.grounding or {}
    tot = g.get("total_quotes") or 0
    gr = g.get("grounded") or 0
    return {
        "risk_categories": [str(getattr(r, "category", "")) for r in (ef.risk_factors or [])],
        "sentiment": str(ef.sentiment),
        "guidance_direction": str(getattr(ef.guidance, "direction", "none")) if ef.guidance else "none",
        "n_risks": len(ef.risk_factors or []),
        "n_events": len(ef.key_events or []),
        "grounded_frac": (gr / tot) if tot else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="limit #docs (debug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml"))
    oc = cfg["openai"]
    model = oc["model"]
    docs = json.load(open(MANIFEST))["documents"]
    if args.limit:
        docs = docs[:args.limit]

    per_doc_cost = 0.0163  # gemini-2.5-flash, from model_eval
    print(f"model: {model} | docs: {len(docs)} | runs: {args.runs} | "
          f"extractions: {len(docs)*args.runs} | est. cost ~${len(docs)*args.runs*per_doc_cost:.1f}")
    if args.dry_run:
        return

    inc_dir = Path("data/v2/analysis/repro_incidents")
    inc_dir.mkdir(parents=True, exist_ok=True)
    extractor = FilingExtractor(api_key=oc["api_key"], model=model, base_url=oc.get("base_url"),
                                incidents_dir=inc_dir, max_workers=args.workers,
                                enable_grounding=True, grounding_threshold=90.0, max_tokens=8000)

    t0 = time.time()
    # runs[r][doc_id] = signature
    runs = [dict() for _ in range(args.runs)]
    for r in range(args.runs):
        done = 0
        for d in docs:
            p = Path(d["path"])
            if not p.exists():
                continue
            fdate = p.stem.split("_")[-1]  # 10-Q_2023-08-02 -> 2023-08-02
            ef = extractor.extract_from_file(p, d["ticker"], d["form"], fdate)
            runs[r][d["id"]] = facts_signature(ef)
            done += 1
            if done % 20 == 0:
                print(f"  run {r+1}/{args.runs}: {done}/{len(docs)} "
                      f"({done/max(1e-9,time.time()-t0)*60:.0f}/min cumulative)", flush=True)
        print(f"run {r+1} done ({done} docs)", flush=True)

    # Aggregate agreement across runs, per doc
    jac, sent_agree, guid_agree, nrisk_sd, nev_sd, gr_sd = [], [], [], [], [], []
    doc_ids = [d["id"] for d in docs]
    for did in doc_ids:
        sigs = [runs[r].get(did) for r in range(args.runs)]
        if any(s is None for s in sigs):
            continue
        # pairwise category Jaccard
        pj = [jaccard(a["risk_categories"], b["risk_categories"])
              for a, b in itertools.combinations(sigs, 2)]
        jac.append(mean(pj))
        sent_agree.append(1.0 if len(set(s["sentiment"] for s in sigs)) == 1 else 0.0)
        guid_agree.append(1.0 if len(set(s["guidance_direction"] for s in sigs)) == 1 else 0.0)
        nrisk_sd.append(pstdev([s["n_risks"] for s in sigs]))
        nev_sd.append(pstdev([s["n_events"] for s in sigs]))
        gr_sd.append(pstdev([s["grounded_frac"] for s in sigs]))

    summary = {
        "model": model, "runs": args.runs, "n_docs_compared": len(jac),
        "temperature": 0.1,
        "risk_category_jaccard_mean": round(mean(jac), 4) if jac else None,
        "sentiment_agreement_rate": round(mean(sent_agree), 4) if sent_agree else None,
        "guidance_direction_agreement_rate": round(mean(guid_agree), 4) if guid_agree else None,
        "n_risks_std_mean": round(mean(nrisk_sd), 4) if nrisk_sd else None,
        "n_events_std_mean": round(mean(nev_sd), 4) if nev_sd else None,
        "grounded_frac_std_mean": round(mean(gr_sd), 4) if gr_sd else None,
        "cost_est_usd": round(len(docs) * args.runs * per_doc_cost, 2),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print("\n=== C4 reproducibility ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
