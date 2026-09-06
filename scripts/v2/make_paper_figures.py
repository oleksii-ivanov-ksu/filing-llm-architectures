#!/usr/bin/env python3
"""
Paper figures for the architecture-comparison article. ALL LABELS IN ENGLISH.

Fig 1 — the two pipeline architectures (schematic).
Fig 2 — quote-based fact verification mechanism (schematic).
Fig 3 — the evaluation instrument: quarterly top-10 / bottom-10 / random-10 (schematic).
Fig 5 — MAIN result: top-10 / bottom-10 Sharpe by signal source, 10 splits, random null band.
Fig 6 — cumulative return over the full period, top-10 vs bottom-10 for each source.
Fig 7 — factor attribution (FF5+Mom) of the one-stage top-10 vs bottom-10.

PNG + SVG to paper/v2_architecture/figures/. $0 (no API).
"""
import sys, warnings, statistics, random
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, RegularPolygon

FIG = Path("paper/v2_architecture/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})

SRC = {"One-stage": "#1b7837", "Two-stage": "#c51b7d", "Dictionary sentiment": "#7f7f7f"}

# Restrained, academic schematic palette: neutral surfaces + one steel-blue accent.
INK, EDGE = "#1a1a1a", "#3a3a3a"
NEUTRAL = "#eceef1"      # ordinary process step
ACCENT = "#c9d6e8"       # emphasized node (input / output / decision result)
GOOD = "#2f6b3d"         # muted green (text only)
BAD = "#9c3a3a"          # muted red (text only)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}", flush=True)


def _box(ax, x, y, w, h, text, fc=NEUTRAL, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=EDGE, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight=weight)


def _arrow(ax, x1, y1, x2, y2, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color=EDGE,
                                 shrinkA=2, shrinkB=2))


def fig1_architectures():
    fig, ax = plt.subplots(figsize=(10, 5.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    _box(ax, 0.15, 2.45, 1.7, 1.15, "Filing 10-K / 10-Q\n+ quantitative\ncontext", ACCENT, fs=9.5)
    ax.text(5.15, 5.72, "Two-stage architecture", ha="center", fontsize=11.5, weight="bold", color=INK)
    _box(ax, 2.5, 4.45, 1.75, 1.0, "Extract structured\nfacts + quotes", fs=9.5)
    _box(ax, 4.75, 4.45, 1.55, 1.0, "Verify quotes", fs=9.5)
    _box(ax, 6.8, 4.45, 1.6, 1.0, "Reason over facts", fs=9.5)
    _box(ax, 8.9, 4.45, 1.0, 1.0, "view_bps", ACCENT, fs=9.5)
    _arrow(ax, 1.85, 3.5, 2.5, 4.95); _arrow(ax, 4.25, 4.95, 4.75, 4.95)
    _arrow(ax, 6.3, 4.95, 6.8, 4.95); _arrow(ax, 8.4, 4.95, 8.9, 4.95)
    ax.text(5.15, 2.15, "One-stage architecture", ha="center", fontsize=11.5, weight="bold", color=INK)
    _box(ax, 2.5, 0.75, 2.3, 1.0, "View directly\n(view_bps + quotes),\nsingle call", fs=9.5)
    _box(ax, 5.3, 0.75, 1.55, 1.0, "Verify quotes", fs=9.5)
    _box(ax, 7.4, 0.75, 1.0, 1.0, "view_bps", ACCENT, fs=9.5)
    _arrow(ax, 1.85, 2.55, 2.5, 1.25); _arrow(ax, 4.8, 1.25, 5.3, 1.25); _arrow(ax, 6.85, 1.25, 7.4, 1.25)
    ax.plot([1.0, 1.0], [2.45, 3.6], color=EDGE, lw=0)  # spacing anchor
    ax.text(5.15, 3.02, "shared input   ·   shared model   ·   shared verification",
            ha="center", fontsize=9, style="italic", color="#555")
    save(fig, "fig1_architectures")


def fig2_verification():
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.2)
    yc = 2.35
    _box(ax, 0.15, yc - 0.5, 1.95, 1.0, "Extracted fact\n+ source quote", ACCENT, fs=9.5)
    _box(ax, 2.55, yc - 0.5, 1.95, 1.0, "Normalize text\n(quotes, dashes,\nspaces, Unicode)", fs=9)
    _box(ax, 4.95, yc - 0.5, 2.1, 1.0, "Locate in filing:\nexact, then fuzzy\n(similarity ≥ 90)", fs=9)
    # decision diamond
    dx, dy = 8.05, yc
    ax.add_patch(RegularPolygon((dx, dy), numVertices=4, radius=0.72, orientation=0,
                                fc=ACCENT, ec=EDGE, lw=1.1))
    ax.text(dx, dy, "quote\nmatched?", ha="center", va="center", fontsize=8.8, color=INK)
    _box(ax, 9.25, 3.05, 1.25, 0.8, "Keep fact", fs=9.5)
    _box(ax, 9.25, 0.35, 1.25, 0.8, "Drop fact", fs=9.5)
    _arrow(ax, 2.1, yc, 2.55, yc); _arrow(ax, 4.5, yc, 4.95, yc); _arrow(ax, 7.05, yc, 7.33, yc)
    _arrow(ax, dx + 0.35, dy + 0.35, 9.25, 3.35)
    _arrow(ax, dx + 0.35, dy - 0.35, 9.25, 0.9)
    ax.text(8.75, 3.15, "yes", fontsize=9, color=GOOD, weight="bold", ha="center")
    ax.text(8.72, 1.35, "no", fontsize=9, color=BAD, weight="bold", ha="center")
    ax.text(5.3, 0.15, "Rule: no matched source quote  →  the fact does not exist",
            ha="center", fontsize=9.5, style="italic", color="#444")
    save(fig, "fig2_verification")


def fig3_instrument():
    fig, ax = plt.subplots(figsize=(10, 3.8)); ax.axis("off"); ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.4)
    _box(ax, 0.15, 1.2, 2.4, 1.0, "Each quarter:\nrank 50 stocks\nby signal", ACCENT, fs=9.5)
    _box(ax, 3.35, 2.25, 3.2, 0.8, "Top-10  —  hold, equal weight", fs=9.5)
    _box(ax, 3.35, 1.2, 3.2, 0.8, "Bottom-10  —  control", fs=9.5)
    _box(ax, 3.35, 0.15, 3.2, 0.8, "Random-10 ×100  —  null", fs=9.5)
    for yy in (2.65, 1.6, 0.55):
        _arrow(ax, 2.55, 1.7, 3.35, yy)
    _box(ax, 7.25, 1.2, 3.15, 0.8, "Sharpe ratio;\nspread = top − bottom", ACCENT, fs=9.5)
    _arrow(ax, 6.55, 2.65, 7.25, 1.75); _arrow(ax, 6.55, 1.6, 7.25, 1.6); _arrow(ax, 6.55, 0.55, 7.25, 1.45)
    ax.text(5.3, 0.0, "rebalanced quarterly; the whole test repeated over 10 random 50-stock universes",
            ha="center", fontsize=9, style="italic", color="#555")
    save(fig, "fig3_instrument")


def load_env():
    from scripts.run_backtest import load_universe, load_all_prices
    from src.backtesting.engine import generate_rebalance_dates, load_views_summary, run_backtest
    from src.backtesting.metrics import sharpe_ratio
    uni = load_universe("config/universe.yaml"); allt = uni["core"] + uni["holdout"]
    ap = load_all_prices(Path("data/raw"), allt); ap = ap[ap.index <= pd.Timestamp(2026, 3, 31)]
    allt = [t for t in allt if t in ap.columns]
    dates = generate_rebalance_dates(2006, 2025, "quarterly")
    V = {"Dictionary sentiment": "data/v2/views_lm/views_summary.csv",
         "One-stage": "data/v2/views_singlestage/views_summary.csv",
         "Two-stage": "data/v2/views_v2/views_summary.csv"}
    V = {k: load_views_summary(Path(v)) for k, v in V.items()}
    return uni, ap, allt, dates, V, run_backtest, sharpe_ratio


def fig5_main(env):
    uni, ap, allt, dates, V, run_backtest, sharpe = env
    random.seed(42); splits = []
    for _ in range(10):
        s = allt[:]; random.shuffle(s); splits.append(s[:50])

    def sh(vdf, A, strat, seed=0):
        dr = run_backtest(prices=ap[A], views_df=vdf, tickers=A, rebalance_dates=dates,
                          strategy=strat, placebo_seed=seed)["daily_returns"].dropna()
        return sharpe(dr)

    order = ["Dictionary sentiment", "Two-stage", "One-stage"]
    # bar height = full-universe (100 companies); whiskers = std over 10 random 50-subsamples
    top_full, bot_full, top_s, bot_s = {}, {}, {}, {}
    for name in order:
        top_full[name] = sh(V[name], allt, "top_n_views")
        bot_full[name] = sh(V[name], allt, "bottom_n_views")
        top_s[name] = statistics.pstdev([sh(V[name], A, "top_n_views") for A in splits])
        bot_s[name] = statistics.pstdev([sh(V[name], A, "bottom_n_views") for A in splits])
    rnd = [sh(V["Two-stage"], A, "random_n", seed=s) for A in splits for s in range(20)]
    rmean, rstd = statistics.mean(rnd), statistics.pstdev(rnd)

    fig, ax = plt.subplots(figsize=(8.4, 5))
    x = np.arange(len(order)); w = 0.36
    ax.bar(x - w/2, [top_full[n] for n in order], w, yerr=[top_s[n] for n in order],
           capsize=4, label="Top-10 (best-rated)", color="#2c7fb8")
    ax.bar(x + w/2, [bot_full[n] for n in order], w, yerr=[bot_s[n] for n in order],
           capsize=4, label="Bottom-10 (worst-rated)", color="#d95f0e")
    ax.axhspan(rmean - rstd, rmean + rstd, color="#cccccc", alpha=0.4, zorder=0)
    ax.axhline(rmean, color="#888", ls="--", lw=1.2, label=f"Random-10 null ≈ {rmean:.2f}")
    for i, n in enumerate(order):
        sp = top_full[n] - bot_full[n]
        ax.text(i, max(top_full[n], bot_full[n]) + 0.055, f"spread {sp:+.2f}", ha="center",
                fontsize=9, weight="bold", color="#1b7837" if sp > 0 else "#c51b7d")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Sharpe ratio (full universe; rf = 2%, net 10 bps)")
    ax.set_title("Stock-selection skill: only the one-stage pipeline ranks stocks")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(top_full.values()) + 0.28)
    ax.text(0.5, -0.14, "bars: full 100-company universe · whiskers: ±std over 10 random 50-company subsamples",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#666")
    save(fig, "fig5_selection_spread")


def fig6_cumulative(env):
    uni, ap, allt, dates, V, run_backtest, sharpe = env
    universe = allt  # full 100-company universe (core/holdout discarded)

    def curve(vdf, strat, seed=0):
        dr = run_backtest(prices=ap[universe], views_df=vdf, tickers=universe, rebalance_dates=dates,
                          strategy=strat, placebo_seed=seed)["daily_returns"].dropna()
        return (1 + dr).cumprod()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for name, col in SRC.items():
        ct = curve(V[name], "top_n_views"); cb = curve(V[name], "bottom_n_views")
        ax.plot(ct.index, ct.values, color=col, lw=2.0, label=f"{name} — top-10")
        ax.plot(cb.index, cb.values, color=col, lw=1.3, ls="--", alpha=0.9,
                label=f"{name} — bottom-10")
    # random null: mean cumulative curve over seeds
    rnd = [curve(V["Two-stage"], "random_n", seed=s) for s in range(20)]
    idx = rnd[0].index
    rnd_mean = pd.concat([r.reindex(idx).ffill() for r in rnd], axis=1).mean(axis=1)
    ax.plot(idx, rnd_mean.values, color="black", lw=1.1, ls=":", label="Random-10 (null)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Cumulative return of top-10 vs bottom-10 by signal source, 2006–2025")
    ax.legend(fontsize=8.3, ncol=2, loc="upper left")
    save(fig, "fig6_cumulative_returns")


def fig7_factors(env):
    import statsmodels.api as sm
    from scripts.factor_attribution import load_factors
    uni, ap, allt, dates, V, run_backtest, sharpe = env
    universe = allt  # full 100-company universe (core/holdout split discarded)
    fac = load_factors(); v1 = V["One-stage"]
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

    def reg(strat):
        r = run_backtest(prices=ap[universe], views_df=v1, tickers=universe, rebalance_dates=dates,
                         strategy=strat)["daily_returns"].dropna()
        common = r.index.intersection(fac.index)
        y = r.loc[common] - fac.loc[common, "RF"]; X = sm.add_constant(fac.loc[common, facs])
        return sm.OLS(y, X).fit()

    mt, mb = reg("top_n_views"), reg("bottom_n_views")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
    labels = ["Top-10", "Bottom-10"]
    alphas = [mt.params["const"] * 252 * 100, mb.params["const"] * 252 * 100]
    errs = [mt.bse["const"] * 252 * 100, mb.bse["const"] * 252 * 100]
    ts = [mt.tvalues["const"], mb.tvalues["const"]]
    bars = ax1.bar(labels, alphas, yerr=errs, capsize=5, color=["#2c7fb8", "#d95f0e"])
    for b, a, t in zip(bars, alphas, ts):
        ax1.text(b.get_x() + b.get_width()/2, a + 0.4, f"{a:+.1f}%\nt={t:.1f}",
                 ha="center", fontsize=9, weight="bold")
    ax1.axhline(0, color="#333", lw=0.8)
    ax1.set_ylabel("Annualized alpha, % (after FF5 + Mom)")
    ax1.set_title("A. One-stage alpha: real in top-10")
    keyf = ["HML", "Mom"]; xt = np.arange(len(keyf)); w = 0.36
    ax2.bar(xt - w/2, [mt.params[f] for f in keyf], w, yerr=[mt.bse[f] for f in keyf],
            capsize=4, label="Top-10", color="#2c7fb8")
    ax2.bar(xt + w/2, [mb.params[f] for f in keyf], w, yerr=[mb.bse[f] for f in keyf],
            capsize=4, label="Bottom-10", color="#d95f0e")
    ax2.axhline(0, color="#333", lw=0.8); ax2.set_xticks(xt)
    ax2.set_xticklabels(["HML\n(value)", "Mom\n(momentum)"])
    ax2.set_ylabel("Factor loading (beta)")
    ax2.set_title("B. Bottom-10 = cheap losers (HML+, Mom−)")
    ax2.legend(fontsize=9)
    save(fig, "fig7_factor_attribution")


if __name__ == "__main__":
    print("Fig 1..."); fig1_architectures()
    print("Fig 2..."); fig2_verification()
    print("Fig 3..."); fig3_instrument()
    print("loading backtest env..."); env = load_env()
    print("Fig 5..."); fig5_main(env)
    print("Fig 6..."); fig6_cumulative(env)
    print("Fig 7..."); fig7_factors(env)
    print("DONE")
