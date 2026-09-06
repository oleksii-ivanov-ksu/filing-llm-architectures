#!/usr/bin/env python3
"""
Figures for the extraction model-selection experiment (D0).
Reads data/v2/analysis/model_eval.json, writes PNG+SVG to paper/v2_architecture/figures/.

Figures (numbered for easy reference in the paper; final numbers set on integration):
  fig1_ms_metrics      — Fig. 1: recall / grounding / F1 per model (grouped bars)
  fig2_ms_cost_recall  — Fig. 2: value frontier — corpus cost (log x) vs recall of important events
  fig3_ms_events       — Fig. 3: events/doc split into matched-gold (important) vs extra (real, lower-priority)
"""
import os, sys, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = Path("data/v2/analysis/model_eval.json")
OUT = Path("paper/v2_architecture/figures")

# validated categorical palette (dataviz skill) + ink tokens
BLUE, ORANGE, GREEN = "#2a78d6", "#e8833a", "#3aa17e"
INK, MUTED, GRID, SURF = "#1a1a19", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

SHORT = {  # readable labels
 "google_gemini-2.5-flash": "Gemini 2.5 Flash", "google_gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
 "openai_gpt-4o-mini": "GPT-4o-mini", "meta-llama_llama-4-maverick": "Llama-4 Maverick",
 "anthropic_claude-haiku-4.5": "Claude Haiku 4.5", "deepseek_deepseek-chat-v3.1": "DeepSeek V3.1",
 "qwen_qwen3.8-27b": "Qwen3.8-27B", "z-ai_glm-5.3-flash": "GLM-5.3-Flash",
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT/name}.png")


def main():
    R = json.loads(EVAL.read_text())
    models = list(R.keys())

    # ---- Fig 1: cost vs recall (value frontier) ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in models:
        r = R[m]
        x, y = r.get("corpus_cost"), r.get("recall_blind")
        if x is None or y is None:
            continue
        # reliable models filled, unreliable (success<0.9) hollow
        reliable = (r.get("success_rate") or 0) >= 0.9
        color = BLUE if reliable else MUTED
        ax.scatter(x, y, s=90, color=color if reliable else "none",
                   edgecolor=color, linewidth=1.6, zorder=3)
        ax.annotate(SHORT.get(m, m), (x, y), textcoords="offset points",
                    xytext=(7, 4), fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Corpus cost (9,101 docs), $  ·  log scale")
    ax.set_ylabel("Recall of important events (blind tier)")
    ax.set_title("Value frontier: coverage of important events vs cost",
                 fontsize=11, weight="bold", loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.annotate("filled = reliable (success ≥ 90%)   hollow = unreliable",
                (0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig2_ms_cost_recall")

    # ---- Fig 2: metrics bars (recall, grounding, F1) sorted by F1 ----
    order = sorted(models, key=lambda m: -(R[m].get("f1_blind_grounded") or 0))
    labels = [SHORT.get(m, m) for m in order]
    recall = [R[m].get("recall_blind") or 0 for m in order]
    ground = [R[m].get("grounding_rate") or 0 for m in order]
    f1 = [R[m].get("f1_blind_grounded") or 0 for m in order]
    import numpy as np
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (vals, c, lab) in enumerate([(recall, BLUE, "Recall (important)"),
                                        (ground, GREEN, "Grounding"),
                                        (f1, ORANGE, "F1 (grounding-weighted)")]):
        bars = ax.bar(x + (i-1)*w, vals, w, color=c, label=lab, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}", ha="center",
                    fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.18)
    ax.set_title("Fig. 1. Extraction quality by model (sorted by F1)", fontsize=11, weight="bold", loc="left", pad=24)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.10))
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    save(fig, "fig1_ms_metrics")

    # ---- Fig 3: matched vs extra events per doc (blind) ----
    order3 = sorted(models, key=lambda m: -((R[m].get("detail", {}).get("blind", {}).get("tp") or 0)))
    labels3, matched, extra = [], [], []
    for m in order3:
        b = R[m].get("detail", {}).get("blind", {})
        docs = b.get("docs") or 1
        labels3.append(SHORT.get(m, m))
        matched.append((b.get("tp") or 0) / docs)
        extra.append(R[m].get("extra_ev_per_doc_blind") or 0)
    x = np.arange(len(order3))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x, matched, 0.6, color=BLUE, label="Matched gold (important)", zorder=3)
    ax.bar(x, extra, 0.6, bottom=matched, color=ORANGE, label="Extra (real, lower-priority)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels3, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Events per document (blind tier)")
    ax.set_title("Fig. 3. Selectivity: important events found vs extra real events",
                 fontsize=11, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    save(fig, "fig3_ms_events")


if __name__ == "__main__":
    main()
