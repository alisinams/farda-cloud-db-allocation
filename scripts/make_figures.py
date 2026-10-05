# -*- coding: utf-8 -*-
"""Figures for the manuscript, generated from results/runs and results/summary."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from farda.analysis import load_runs, pareto_front, CELL_LABEL  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"FARDA (LSTM + MILP)": "#1b4f72", "MILP, reactive": "#5d8aa8", "MILP, oracle forecast": "#7fb3d5",
          "PSO, reactive": "#c0392b", "GA, reactive": "#d68910", "ACO, reactive": "#7d6608",
          "Proportional spreading, no optimiser": "#566573"}
# legend names matching the manuscript's display names (fill_manuscript.DISPLAY)
SURGE_LABEL = {"FARDA (LSTM + MILP)": "FARDA: exact allocator, LSTM + margin", "MILP, reactive": "Exact allocator, last observed value",
               "PSO, reactive": "PSO, last observed value", "GA, reactive": "GA, last observed value",
               "ACO, reactive": "ACO, last observed value", "Proportional spreading, no optimiser": "Proportional spreading"}


def fig_surge():
    """Per-interval mean latency under the threefold step, averaged over seeds, FARDA vs comparators."""
    series = {}
    for p in glob.glob(os.path.join(RES, "runs", "surge_h10_*_regimes.json")):
        with open(p) as fh:
            d = json.load(fh)
        cell = CELL_LABEL[(d["config"]["forecaster"], d["config"]["allocator"])]
        if cell not in ("FARDA (LSTM + MILP)", "PSO, reactive", "GA, reactive", "ACO, reactive", "MILP, reactive", "Proportional spreading, no optimiser"):
            continue
        series.setdefault(cell, []).append(d["series"]["interval_mean_latency_ms"])
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    t = np.arange(288) * 5 / 60.0
    for cell, arrs in sorted(series.items(), key=lambda kv: list(SURGE_LABEL).index(kv[0])):
        a = np.array(arrs); m = np.nanmedian(a, axis=0)
        lo, hi = np.nanpercentile(a, [25, 75], axis=0)
        ax.plot(t, m, color=COLORS[cell], lw=1.3, label=f"{SURGE_LABEL[cell]} (n={a.shape[0]})")
        ax.fill_between(t, lo, hi, color=COLORS[cell], alpha=0.15, lw=0)
    ax.axvline(12, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Hour of the simulated day"); ax.set_ylabel("Mean latency per interval (ms)")
    ax.set_yscale("log"); ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Threefold arrival step at hour 12 (median over seeds, IQR shaded)", fontsize=8)
    fig.tight_layout(); p = os.path.join(FIG, "fig_surge.png"); fig.savefig(p, dpi=300); plt.close(fig)
    return p


def fig_overhead():
    """Commit-delay sweep from results/summary/overhead_sweep.csv (includes the reused default point)."""
    p_csv = os.path.join(RES, "summary", "overhead_sweep.csv")
    if not os.path.exists(p_csv):
        return None
    df = pd.read_csv(p_csv)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))
    for ax, regime, met, lab in [(axes[0], "steady", "mean_latency_ms", "Mean latency (ms), steady regime"),
                                 (axes[1], "adversarial", "mean_latency_tampered_ms", "Latency of tampered requests (ms)")]:
        g = df[(df.regime == regime) & (df.metric == met)].sort_values("commit_delay")
        if g.empty:
            continue
        x = g.commit_delay.to_numpy(float); x_plot = np.where(x == 0, 0.003, x)
        yerr = np.vstack([g["mean"] - g["ci_low"], g["ci_high"] - g["mean"]])
        ax.errorbar(x_plot, g["mean"], yerr=yerr, fmt="o-", color="#1b4f72", ms=3, lw=1, capsize=2)
        ax.set_xscale("log"); ax.set_xlabel("Commit delay OH (s); 0 plotted at 0.003"); ax.set_ylabel(lab, fontsize=8)
        if regime == "adversarial":
            ax.set_yscale("log")
    fig.tight_layout(); p = os.path.join(FIG, "fig_overhead.png"); fig.savefig(p, dpi=300); plt.close(fig)
    return p


def fig_pareto():
    p_csv = os.path.join(RES, "summary", "weight_sweep.csv")
    if not os.path.exists(p_csv):
        return None
    agg = pd.read_csv(p_csv)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    names = {"milp": "Exact allocator (FARDA)", "pso": "PSO", "ga": "GA", "aco": "ACO"}
    cols = {"milp": "#1b4f72", "pso": "#c0392b", "ga": "#d68910", "aco": "#7d6608"}
    try:
        with open(os.path.join(RES, "summary", "report.json")) as fh:
            objs = json.load(fh)["weight_sweep"].get("objectives", ["energy", "exposure", "latency"])
    except Exception:
        objs = ["energy", "exposure", "latency"]
    for al, g in agg.groupby("cfg_allocator"):
        m = g["non_dominated"].astype(bool).to_numpy() if "non_dominated" in g else pareto_front(g[objs].values)
        axes[0].scatter(g.energy, g.latency, s=8, color=cols[al], alpha=0.35)
        axes[0].scatter(g.energy[m], g.latency[m], s=14, color=cols[al], label=names[al])
        axes[1].scatter(g.exposure, g.latency, s=8, color=cols[al], alpha=0.35)
        axes[1].scatter(g.exposure[m], g.latency[m], s=14, color=cols[al])
    axes[0].set_xlabel("Energy (kWh)"); axes[0].set_ylabel("Mean latency (ms)"); axes[0].set_yscale("log")
    axes[1].set_xlabel("Exposure of sensitive requests"); axes[1].set_ylabel("Mean latency (ms)"); axes[1].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    # triples, seeds and the dominance objectives are stated in the manuscript caption (fill_manuscript.py)
    fig.tight_layout(); p = os.path.join(FIG, "fig_pareto.png"); fig.savefig(p, dpi=300); plt.close(fig)
    return p


def fig_forecast():
    p_csv = os.path.join(RES, "summary", "forecast_metrics.csv")
    if not os.path.exists(p_csv):
        return None
    m = pd.read_csv(p_csv)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    style = {"naive": ("#7f8c8d", "--"), "seasonal_naive": ("#7f8c8d", ":"), "lightgbm": ("#c0392b", "-"),
             "lstm_16": ("#aed6f1", "-"), "lstm_32": ("#7fb3d5", "-"), "lstm_64": ("#1b4f72", "-"), "lstm_128": ("#2e86c1", "-"), "lstm_256": ("#154360", "-")}
    for model, g in m.groupby("model"):
        g = g.sort_values("h"); c, ls = style.get(model, ("k", "-"))
        ax.errorbar(g.h, g.mae, yerr=g.mae_sd, color=c, ls=ls, marker="o", ms=3, lw=1, capsize=2, label=model.replace("_", " "))
    ax.set_xlabel("Horizon (steps of 300 s)"); ax.set_ylabel("MAE (utilisation percentage points)")
    ax.set_xticks([1, 5, 15, 30]); ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(); p = os.path.join(FIG, "fig_forecast.png"); fig.savefig(p, dpi=300); plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_forecast, fig_surge, fig_overhead, fig_pareto):
        print(f.__name__, "->", f())
