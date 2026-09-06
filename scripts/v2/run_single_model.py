#!/usr/bin/env python3
"""
Single-model test run (inspection tool for experiment D0).

Runs ONE extraction model over the first N gold-standard documents and prints a
readable summary of what was extracted (events, quotes, grounding), so you can
eyeball quality before committing to the full 7-model run. Output is saved to
the SAME location as the full run, so a test run also counts toward it:

    data/v2/model_selection/{model_slug}/{NN}_{DOC_ID}.json

Usage:
    python scripts/v2/run_single_model.py --model x-ai/grok-4.1-fast --n 1
    python scripts/v2/run_single_model.py --model anthropic/claude-haiku-4.5 --n 3
    python scripts/v2/run_single_model.py --model x-ai/grok-4.1-fast --n 1 --force
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.extraction.extractor import FilingExtractor

GROUNDING_THRESHOLD = 90.0
OUT_ROOT = Path("data/v2/model_selection")
MANIFEST = Path("data/v2/gold_standard/manifest.json")


def slug(model: str) -> str:
    return model.replace("/", "_")


def print_facts(rec: dict):
    """Readable dump of one extraction result."""
    f = rec.get("facts")
    if not f:
        print("  ⚠️  extraction returned None (failed)")
        return
    g = f.get("grounding") or {}
    rf = f.get("risk_factors", [])
    ev = f.get("key_events", [])
    print(f"  latency: {rec['latency_sec']}s | risk_factors: {len(rf)} | key_events: {len(ev)}")
    if g:
        print(f"  grounding: {g.get('grounded')}/{g.get('total_facts')} grounded, "
              f"{g.get('dropped')} dropped (threshold {g.get('threshold')})")
    print(f"  sentiment: {f.get('sentiment')}")
    if ev:
        print("  --- key events (with source quotes):")
        for e in ev[:6]:
            print(f"    • [{e.get('event_type')}] {e.get('description')}")
            print(f"        quote: \"{(e.get('source_quote') or '')[:120]}\"")
    if rf:
        print("  --- risk factors (first 3):")
        for r in rf[:3]:
            print(f"    • [{r.get('category')}/{r.get('severity')}] {r.get('description')}")
            print(f"        quote: \"{(r.get('source_quote') or '')[:120]}\"")


def main():
    ap = argparse.ArgumentParser(description="Run ONE model over the first N gold docs (test).")
    ap.add_argument("--model", required=True, help="OpenRouter model slug, e.g. x-ai/grok-4.1-fast")
    ap.add_argument("--n", type=int, default=1, help="Number of gold documents to run (default 1)")
    ap.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    ap.add_argument("--threshold", type=float, default=GROUNDING_THRESHOLD)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    docs = manifest["documents"][:args.n]

    cfg = yaml.safe_load(open("config/config.yaml"))
    oc = cfg.get("openai", {})
    api_key = oc.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    base_url = oc.get("base_url", "https://openrouter.ai/api/v1")
    if not api_key:
        print("ERROR: no OpenRouter API key in config/config.yaml (openai.api_key)")
        sys.exit(1)

    mdir = OUT_ROOT / slug(args.model)
    mdir.mkdir(parents=True, exist_ok=True)

    print(f"MODEL: {args.model}  |  docs: {len(docs)}  |  threshold: {args.threshold}")
    print(f"OUTPUT: {mdir}/\n")

    extractor = FilingExtractor(
        api_key=api_key, model=args.model, base_url=base_url,
        incidents_dir=mdir, max_workers=1, max_tokens=6000,
        enable_grounding=True, grounding_threshold=args.threshold,
    )

    for d in docs:
        out = mdir / f"{d['index']:02d}_{d['id']}.json"
        print(f"[{d['index']:02d}] {d['id']}  ({d['form']} {d['year']}, {d['sector']})")
        if out.exists() and not args.force:
            print(f"  already done -> {out} (use --force to re-run)")
            print_facts(json.loads(out.read_text()))
            print()
            continue
        t = time.time()
        try:
            facts = extractor.extract_from_file(
                filing_path=Path(d["path"]), ticker=d["ticker"],
                filing_type=d["form"], filing_date=f"{d['year']}-01-01",
            )
            rec = {
                "model": args.model, "doc_id": d["id"], "index": d["index"],
                "latency_sec": round(time.time() - t, 2),
                "ok": facts is not None,
                "facts": json.loads(facts.model_dump_json()) if facts else None,
            }
            out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            print_facts(rec)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print(f"cost: {extractor.cost_tracker.summary()}")


if __name__ == "__main__":
    main()
