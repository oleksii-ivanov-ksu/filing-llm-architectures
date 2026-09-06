#!/usr/bin/env python3
"""
Evaluate extraction models against the gold standard (experiment D0).

For each candidate model, matches its extracted key_events to the human-annotated
gold events and computes:
  - recall (gold)      = matched gold events / total gold events   -> completeness
  - grounding rate     = grounded facts / total facts (from model) -> non-fabrication (precision proxy)
  - success rate       = docs with valid output / docs attempted   -> reliability
  - avg events/doc, avg latency

Matching: a model event matches a gold event if their text is similar enough
(rapidfuzz on description + source_quote overlap), greedy best-first, each gold
event matched at most once. Event-type labels are NOT required to match (gold and
model taxonomies differ); matching is on event identity via text.

Primary metric = recall on the BLIND tiers (tier1a_tuning + tier1b_eval, 20 docs),
where the human annotation is exhaustive. tier2 (pooling) is reported separately
as an upper bound.

Usage:
    python scripts/v2/eval_extraction.py                    # all models found
    python scripts/v2/eval_extraction.py --match-threshold 60
    python scripts/v2/eval_extraction.py --show-borderline  # print pairs near threshold

Output:
    data/v2/analysis/model_eval.json   (full per-model, per-doc detail)
    data/v2/analysis/model_eval.md     (summary table)
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

from rapidfuzz import fuzz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.extraction.grounding import normalize

GOLD_DIR = Path("data/v2/gold_standard")
RUNS_DIR = Path("data/v2/model_selection")
OUT_DIR = Path("data/v2/analysis")
BLIND_TIERS = {"tier1a_tuning", "tier1b_eval"}

# OpenRouter pricing ($/M tokens: input, output), keyed by dir slug (model with / -> _).
# Verified 2026-08-29. Used to estimate $/doc and corpus cost (9,101 docs).
PRICING = {
    "z-ai_glm-5.3-flash":            (0.07, 0.25),
    "google_gemini-2.5-flash-lite":  (0.10, 0.40),
    "openai_gpt-4o-mini":            (0.15, 0.60),
    "meta-llama_llama-4-maverick":   (0.20, 0.80),
    "google_gemini-2.5-flash":       (0.30, 2.50),
    "qwen_qwen3.8-27b":              (0.42, 2.55),
    "deepseek_deepseek-chat-v3.1":   (0.55, 1.65),
    "anthropic_claude-haiku-4.5":    (1.00, 5.00),   # expensive reference (quality ceiling)
    "x-ai_grok-4.6":                 (2.00, 6.00),   # expensive reference
    "x-ai_grok-4.5":                 (2.00, 6.00),
}
AVG_INPUT_TOKENS = 39920   # measured over the 90 gold docs
CORPUS_DOCS = 9101

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _toklen(s): return len(_enc.encode(s))
except Exception:
    def _toklen(s): return len(s) // 4


def cost_per_doc(slug: str, avg_output_tokens: float):
    """Clean cost per document = 1 request/doc, NO retry overhead.
    Uses the model's ACTUAL average output size (measured from its results)."""
    p = PRICING.get(slug)
    if not p:
        return None
    return p[0] * AVG_INPUT_TOKENS / 1e6 + p[1] * avg_output_tokens / 1e6

MATCH_THRESHOLD = 60.0        # rapidfuzz score to count a match
BORDERLINE = (50.0, 72.0)     # range to flag for manual review


def event_sim(a: dict, b: dict) -> float:
    """Similarity between two events via description + source_quote."""
    da, db = normalize(a.get("description", "")), normalize(b.get("description", ""))
    qa, qb = normalize(a.get("source_quote", "")), normalize(b.get("source_quote", ""))
    desc_sim = fuzz.token_set_ratio(da, db) if da and db else 0
    # quote overlap: the shorter quote appearing inside the longer
    quote_sim = fuzz.partial_ratio(qa, qb) if qa and qb else 0
    return max(desc_sim, quote_sim)


def match_events(model_events, gold_events, threshold):
    """Greedy best-first matching. Returns (matched_pairs, borderline_pairs)."""
    pairs = []
    for i, m in enumerate(model_events):
        for j, g in enumerate(gold_events):
            pairs.append((event_sim(m, g), i, j))
    pairs.sort(reverse=True)
    used_m, used_g, matched, borderline = set(), set(), [], []
    for score, i, j in pairs:
        if i in used_m or j in used_g:
            continue
        if score >= threshold:
            used_m.add(i); used_g.add(j)
            matched.append((i, j, score))
            if BORDERLINE[0] <= score <= BORDERLINE[1]:
                borderline.append((score, model_events[i], gold_events[j]))
        elif BORDERLINE[0] <= score <= BORDERLINE[1]:
            borderline.append((score, model_events[i], gold_events[j]))
    return matched, borderline


def load_gold():
    gold = {}
    for f in GOLD_DIR.glob("[0-9]*.json"):
        d = json.loads(f.read_text())
        gold[d["id"]] = d
    return gold


def main():
    ap = argparse.ArgumentParser(description="Evaluate extraction models vs gold standard")
    ap.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--show-borderline", action="store_true")
    ap.add_argument("--models", type=str, default=None, help="comma-separated slugs to eval")
    args = ap.parse_args()

    gold = load_gold()
    print(f"Gold: {len(gold)} docs, {sum(len(g['events']) for g in gold.values())} events")

    model_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()] if RUNS_DIR.exists() else []
    if args.models:
        want = set(args.models.split(","))
        model_dirs = [d for d in model_dirs if d.name.replace("_", "/", 1) in want or d.name in want]

    results = {}
    all_borderline = []

    for mdir in sorted(model_dirs):
        model = mdir.name
        # aggregate counters, split by blind vs pooling
        agg = {scope: {"tp": 0, "gold": 0, "model_ev": 0, "docs": 0} for scope in ("blind", "pool", "all")}
        grounded = total_facts = ok = attempted = 0
        lat = []
        out_toks = []
        per_doc = []

        for f in mdir.glob("[0-9]*.json"):
            rec = json.loads(f.read_text())
            gid = rec["doc_id"]
            if gid not in gold:
                continue
            attempted += 1
            lat.append(rec.get("latency_sec", 0))
            fa = rec.get("facts")
            if not fa:
                # failed extraction: counts against recall (found nothing) and reliability
                g_events = gold[gid]["events"]
                scope = "blind" if gold[gid]["tier"] in BLIND_TIERS else "pool"
                for s in (scope, "all"):
                    agg[s]["gold"] += len(g_events); agg[s]["docs"] += 1
                continue
            ok += 1
            out_toks.append(_toklen(json.dumps(fa, ensure_ascii=False)))  # actual output size
            gnd = fa.get("grounding") or {}
            grounded += gnd.get("grounded", 0); total_facts += gnd.get("total_facts", 0)

            m_events = fa.get("key_events", [])
            g_events = gold[gid]["events"]
            matched, borderline = match_events(m_events, g_events, args.match_threshold)
            scope = "blind" if gold[gid]["tier"] in BLIND_TIERS else "pool"
            for s in (scope, "all"):
                agg[s]["tp"] += len(matched)
                agg[s]["gold"] += len(g_events)
                agg[s]["model_ev"] += len(m_events)
                agg[s]["docs"] += 1
            per_doc.append({"doc": gid, "tier": gold[gid]["tier"], "matched": len(matched),
                            "gold": len(g_events), "model": len(m_events)})
            for b in borderline:
                all_borderline.append((model, gid, *b))

        def rate(n, d): return round(n / d, 3) if d else None
        results[model] = {
            "attempted": attempted, "ok": ok,
            "success_rate": rate(ok, attempted),
            "grounding_rate": rate(grounded, total_facts),
            "avg_latency": round(sum(lat) / len(lat), 1) if lat else None,
            "recall_blind": rate(agg["blind"]["tp"], agg["blind"]["gold"]),
            "recall_all": rate(agg["all"]["tp"], agg["all"]["gold"]),
            # precision vs gold on BLIND tier (exhaustive annotation): of the model's
            # events, how many are annotated gold events. 1 - precision = "extra" events
            # (grounded but not in gold = over-extraction / less-material / annotator miss).
            "precision_blind": rate(agg["blind"]["tp"], agg["blind"]["model_ev"]),
            "extra_ev_per_doc_blind": rate(agg["blind"]["model_ev"] - agg["blind"]["tp"], agg["blind"]["docs"]),
            "avg_events_per_doc": rate(agg["all"]["model_ev"], agg["all"]["docs"]),
            "avg_output_tokens": round(sum(out_toks) / len(out_toks)) if out_toks else None,
            "detail": agg, "per_doc": per_doc,
        }

    # F1 from recall_blind + grounding (precision proxy) + cost (needs avg_output_tokens)
    for m, r in results.items():
        rc, pr = r["recall_blind"], r["grounding_rate"]
        r["f1_blind_grounded"] = round(2 * pr * rc / (pr + rc), 3) if pr and rc else None
        avg_out = r.get("avg_output_tokens")
        cpd = cost_per_doc(m, avg_out) if avg_out else None
        r["cost_per_doc"] = round(cpd, 4) if cpd else None
        r["corpus_cost"] = round(cpd * CORPUS_DOCS) if cpd else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model_eval.json").write_text(json.dumps(results, indent=2))

    # summary table
    lines = ["# Model evaluation vs gold standard\n",
             f"Match threshold: {args.match_threshold} | blind tiers: {sorted(BLIND_TIERS)}\n",
             f"Cost = clean (1 request/doc, NO retries): OpenRouter price x {AVG_INPUT_TOKENS:,} input "
             f"+ each model's ACTUAL avg output tokens.\n",
             "| Model | done | success | recall(blind) | precision(blind) | extra/doc | grounding | F1 | ev/doc | $/doc | corpus |",
             "|-------|------|---------|---------------|------------------|-----------|-----------|----|--------|-------|--------|"]
    for m, r in sorted(results.items(), key=lambda x: -(x[1]["f1_blind_grounded"] or 0)):
        avg_out = r.get("avg_output_tokens")
        cpd = r.get("cost_per_doc")
        cstr = f"${cpd:.4f}" if cpd else "?"
        corp = f"${cpd*CORPUS_DOCS:,.0f}" if cpd else "?"
        lines.append(f"| {m} | {r['ok']}/{r['attempted']} | {r['success_rate']} | "
                     f"{r['recall_blind']} | {r['precision_blind']} | {r['extra_ev_per_doc_blind']} | "
                     f"{r['grounding_rate']} | {r['f1_blind_grounded']} | {r['avg_events_per_doc']} | "
                     f"{cstr} | {corp} |")
    table = "\n".join(lines)
    (OUT_DIR / "model_eval.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nWrote {OUT_DIR/'model_eval.md'} and model_eval.json")

    if args.show_borderline and all_borderline:
        print(f"\n=== BORDERLINE matches ({len(all_borderline)}) — review these ===")
        for model, gid, score, me, ge in sorted(all_borderline, key=lambda x: x[2])[:30]:
            print(f"[{score:.0f}] {model} {gid}")
            print(f"    model: {me.get('description','')[:70]}")
            print(f"    gold : {ge.get('description','')[:70]}")


if __name__ == "__main__":
    main()
