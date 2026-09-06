#!/usr/bin/env python3
"""
Model selection on the gold standard (experiment D0).

Runs each candidate EXTRACTION model (schema v2, with grounding) over the 90
gold-standard filings, so we can later measure precision/recall against the
human annotations and pick the best model for the full corpus run.

Storage (per model, per document):
    data/v2/model_selection/
    ├── run_meta.json                       # models, threshold, cost/latency/success
    └── {model_slug}/{NN}_{DOC_ID}.json     # ExtractedFacts v2 (+ grounding) for that doc

model_slug = model id with "/" replaced by "_" (e.g. x-ai_grok-4.1-fast).

Resumable: existing per-doc outputs are skipped. One model at a time.

Usage:
    python scripts/v2/run_model_selection.py --dry-run
    python scripts/v2/run_model_selection.py --models x-ai/grok-4.1-fast
    python scripts/v2/run_model_selection.py            # all DEFAULT_MODELS

NOTE: confirm the exact OpenRouter slugs (openrouter.ai/models) before a real
run — Qwen/Llama/Haiku ids change over time. Override with --models a,b,c.
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.extraction.extractor import FilingExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Candidate extraction models (OpenRouter slugs + $/corpus, verified 2026-08-29).
# Extraction is high-volume (9,101 docs), so ONLY corpus-affordable models qualify.
# DROPPED: x-ai/grok-4.5/4.6 ($0.118/doc = ~$1074/corpus) — flagship, too expensive;
#          the cheap "grok-4.1-fast" tier used in v1 was deprecated/removed by xAI.
DEFAULT_MODELS = [
    # Shortlist after doc-#01 smoke tests + the short-quote/smart-retry fix (2026-08-30).
    # All below produced valid grounded output on the test doc; ranked by corpus cost.
    "google/gemini-2.5-flash-lite",       # ⭐ thorough (10RF+9EV), fastest (11s), cheap
    "openai/gpt-4o-mini",                 # cheapest & reliable, but low recall (2 EV) on test
    "meta-llama/llama-4-maverick",        # open-weight, fast
    "google/gemini-2.5-flash",            # thorough, mid cost
    "z-ai/glm-5.3-flash",                 # open-weight, works after fix (verbose)
    "qwen/qwen3.8-27b",                   # open-weight, thorough
    "deepseek/deepseek-chat-v3.1",        # open-weight, thorough but slow (~120-160s/doc)
    # QUALITY-CEILING REFERENCE (expensive, corpus-infeasible — run on gold standard ONLY
    # to show flagships don't materially beat cheap models; NOT a corpus candidate):
    "anthropic/claude-haiku-4.5",         # ~$0.06/doc base ($555 corpus) — Anthropic reference
    # DROPPED: x-ai/grok-4.5/4.6 (slow/timeout, $1074+ corpus),
    #          deepseek/deepseek-v4-flash-0731 (fails schema validation on our enums).
]

GROUNDING_THRESHOLD = 90.0
OUT_ROOT = Path("data/v2/model_selection")
MANIFEST = Path("data/v2/gold_standard/manifest.json")


def slug(model: str) -> str:
    return model.replace("/", "_")


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Run candidate models over the gold standard")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated model slugs (default: DEFAULT_MODELS)")
    ap.add_argument("--limit", type=int, default=0, help="Limit docs (0 = all 90)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="List what would run, no API calls")
    args = ap.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    manifest = json.loads(MANIFEST.read_text())
    docs = manifest["documents"]
    if args.limit:
        docs = docs[:args.limit]

    cfg = load_config()
    oc = cfg.get("openai", {})
    api_key = oc.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    base_url = oc.get("base_url", "https://openrouter.ai/api/v1")

    print(f"Models ({len(models)}): {models}")
    print(f"Documents: {len(docs)} | grounding threshold: {GROUNDING_THRESHOLD}")
    print(f"Output root: {OUT_ROOT}/")
    if args.dry_run:
        for m in models:
            d = OUT_ROOT / slug(m)
            done = len(list(d.glob("*.json"))) if d.exists() else 0
            print(f"  {m:40s} -> {slug(m)}/  ({done}/{len(docs)} already done)")
        est = len(models) * len(docs) * 0.011  # ~1 cent/doc
        print(f"\nEstimated cost: ~${est:.2f} ({len(models)} models x {len(docs)} docs)")
        return

    if not api_key:
        logger.error("No OpenRouter API key in config/config.yaml (openai.api_key)")
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_meta = {"threshold": GROUNDING_THRESHOLD, "n_docs": len(docs), "models": {}}

    for model in models:
        mdir = OUT_ROOT / slug(model)
        mdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== MODEL: {model} -> {mdir}/ ===")

        extractor = FilingExtractor(
            api_key=api_key, model=model, base_url=base_url,
            incidents_dir=mdir, max_workers=args.workers, max_tokens=6000,
            enable_grounding=True, grounding_threshold=GROUNDING_THRESHOLD,
        )

        todo = [d for d in docs if not (mdir / f"{d['index']:02d}_{d['id']}.json").exists()]
        skipped = len(docs) - len(todo)
        ok = fail = 0
        done = 0
        t0 = time.time()

        def process(d):
            # Extractor is thread-safe (shared rate limiter + cost tracker use locks).
            t = time.time()
            facts = extractor.extract_from_file(
                filing_path=Path(d["path"]), ticker=d["ticker"],
                filing_type=d["form"], filing_date=f"{d['year']}-01-01",
            )
            rec = {
                "model": model, "doc_id": d["id"], "index": d["index"],
                "latency_sec": round(time.time() - t, 2),
                "ok": facts is not None,
                "facts": json.loads(facts.model_dump_json()) if facts else None,
            }
            (mdir / f"{d['index']:02d}_{d['id']}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False))
            return facts is not None

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, d): d for d in todo}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    ok += 1 if fut.result() else 0
                    fail += 0 if fut.result() else 1
                except Exception as e:
                    logger.error(f"  {d['id']}: {e}")
                    fail += 1
                done += 1
                if done % 10 == 0:
                    logger.info(f"  {model}: {done}/{len(todo)} done ({ok} ok, {fail} fail)")

        run_meta["models"][model] = {
            "ok": ok, "fail": fail, "skipped": skipped,
            "elapsed_sec": round(time.time() - t0, 1),
            "cost": extractor.cost_tracker.summary(),
        }
        logger.info(f"  DONE {model}: {ok} ok, {fail} fail, {skipped} skipped, "
                    f"{run_meta['models'][model]['cost']}")

    (OUT_ROOT / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    logger.info(f"All models done. Meta -> {OUT_ROOT/'run_meta.json'}")


if __name__ == "__main__":
    main()
