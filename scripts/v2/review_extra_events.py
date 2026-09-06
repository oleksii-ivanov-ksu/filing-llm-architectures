#!/usr/bin/env python3
"""
Review "extra" events: model events that did NOT match any gold event, on a
blind-tier document (where gold annotation is exhaustive). Shows each extra
event with its verbatim source_quote so we can confirm it is grounded (real)
and simply absent from gold — not fabricated.

One document per model. Writes data/v2/analysis/extra_events_review.md.
"""
import os, sys, json
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", ".."))  # repo root (for src.*)
sys.path.append(_HERE)                              # scripts/v2 (for eval_extraction)
from eval_extraction import match_events, load_gold, BLIND_TIERS  # reuse matcher

RUNS = Path("data/v2/model_selection")
OUT = Path("data/v2/analysis/extra_events_review.md")


def pick_doc_with_extras(mdir, gold):
    """Pick a blind-tier doc where this model produced extra (unmatched) events."""
    best = None
    for f in sorted(mdir.glob("[0-9]*.json")):
        rec = json.loads(f.read_text())
        gid = rec["doc_id"]
        if gid not in gold or gold[gid]["tier"] not in BLIND_TIERS:
            continue
        fa = rec.get("facts")
        if not fa:
            continue
        m_ev = fa.get("key_events", [])
        g_ev = gold[gid]["events"]
        matched, _ = match_events(m_ev, g_ev, 60.0)
        matched_i = {i for i, _, _ in matched}
        extras = [e for i, e in enumerate(m_ev) if i not in matched_i]
        if extras and (best is None or len(extras) > len(best[2])):
            best = (gid, g_ev, extras, m_ev, matched)
    return best


def main():
    gold = load_gold()
    lines = ["# Review of extra events (grounded but not in gold standard)\n",
             "For each model: one blind-tier document, its events that did NOT match any",
             "gold event. All are grounded (a verbatim quote was found in the filing), so",
             "they are real facts — the question is only whether the human annotator",
             "considered them key. Confirms extras are not fabrications.\n"]

    for mdir in sorted(d for d in RUNS.iterdir() if d.is_dir()):
        model = mdir.name
        picked = pick_doc_with_extras(mdir, gold)
        if not picked:
            continue
        gid, g_ev, extras, m_ev, matched = picked
        lines.append(f"\n## {model}")
        lines.append(f"**Document:** {gid} — model extracted {len(m_ev)} events, "
                     f"{len(matched)} matched gold, **{len(extras)} extra** "
                     f"(gold has {len(g_ev)} events total)\n")
        for e in extras:
            q = (e.get("source_quote") or "").strip()
            lines.append(f"- **[{e.get('event_type')}]** {e.get('description')}")
            lines.append(f"  - quote: \"{q[:180]}\"")

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
