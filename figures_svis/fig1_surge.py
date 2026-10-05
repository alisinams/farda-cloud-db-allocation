# -*- coding: utf-8 -*-
"""Fig. 1: per-interval mean latency under the threefold arrival step at hour 12 (10 hosts).

Data: results/runs/surge_h10_*_regimes.json (series.interval_mean_latency_ms, 288 intervals of 300 s).
Lines: median over seeds; band: interquartile range. Six cells, colour fixed per cell (fig_config.GROUP_INDEX).
"""
import glob
import json
import sys

import numpy as np

import fig_config as fc
import sciviz as sv

sys.path.insert(0, str(fc.ARTEFACT))
from farda.analysis import CELL_LABEL  # noqa: E402

CELLS = ["FARDA (LSTM + MILP)", "MILP, reactive", "Proportional spreading, no optimiser",
         "PSO, reactive", "GA, reactive", "ACO, reactive"]


def load():
    series = {}
    for p in glob.glob(str(fc.RESULTS / "runs" / "surge_h10_*_regimes.json")):
        with open(p) as fh:
            d = json.load(fh)
        cell = CELL_LABEL[(d["config"]["forecaster"], d["config"]["allocator"])]
        if cell in CELLS:
            series.setdefault(cell, []).append(d["series"]["interval_mean_latency_ms"])
    return {c: np.asarray(series[c], float) for c in CELLS if c in series}


SHORT = {"FARDA (LSTM + MILP)": "FARDA", "MILP, reactive": "Exact, reactive",
         "Proportional spreading, no optimiser": "Spreading", "PSO, reactive": "PSO",
         "GA, reactive": "GA", "ACO, reactive": "ACO"}


def after_step_per_seed():
    """Mean latency after the step, one value per seed (results/summary/all_runs.csv)."""
    import pandas as pd
    runs = pd.read_csv(fc.SUMMARY / "all_runs.csv")
    runs = runs[(runs.cfg_regime == "surge") & (runs.cfg_n_hosts == 10)]
    runs = runs.assign(cell=[CELL_LABEL[(f, a)] for f, a in zip(runs.cfg_forecaster, runs.cfg_allocator)])
    return {c: runs.loc[runs.cell == c, "mean_latency_after_ms"].to_numpy(float) for c in CELLS}


def main():
    pal = fc.setup()
    data = load()
    after = after_step_per_seed()
    fig, axs = sv.figure(fc.FINAL_WIDTH_MM, aspect=0.54, ncols=2, width_ratios=[2.0, 1.1])
    ax, bx = axs[0]
    t = np.arange(288) * 5 / 60.0
    for cell, a in data.items():
        st = fc.style(pal, cell)
        med = np.nanmedian(a, axis=0)
        q1, q3 = np.nanpercentile(a, [25, 75], axis=0)
        ax.fill_between(t, q1, q3, color=sv.with_alpha(st["color"][:3], "band_sd"), linewidth=0)
        ax.plot(t, med, color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markevery=24,
                markersize=6, linewidth=1.8, label=f"{fc.DISPLAY[cell]} (n = {a.shape[0]})")
    ax.axvline(12, color=sv.INK, linestyle=":", linewidth=1.2, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Time of simulated day (h)")
    ax.set_ylabel("Mean latency (ms)")
    sv.enforce_closed_frame(ax, minor=False)

    # (b) every seed's mean latency after the step, median (vertical bar) and interquartile range
    rng = np.random.default_rng(20260928)
    ypos = {c: len(CELLS) - 1 - k for k, c in enumerate(CELLS)}
    for cell in CELLS:
        v, st, y = after[cell], fc.style(pal, cell), ypos[cell]
        jit = y + rng.uniform(-0.18, 0.18, v.size)
        bx.scatter(v, jit, s=18, marker=st["marker"], color=sv.with_alpha(st["color"][:3], "scatter"),
                   linewidths=0, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        bx.plot([q1, q3], [y, y], color=sv.INK, linewidth=1.6, zorder=3)
        bx.plot([med, med], [y - 0.32, y + 0.32], color=sv.INK, linewidth=2.2, zorder=3)
    bx.set_xscale("log")
    bx.set_yticks([ypos[c] for c in CELLS], [SHORT[c] for c in CELLS])
    bx.set_ylim(-0.6, len(CELLS) - 0.4)
    bx.set_xlabel("Latency after step (ms)")
    sv.enforce_closed_frame(bx, minor=False)

    handles, labels = ax.get_legend_handles_labels()
    sv.figure_legend(fig, handles, labels, where="bottom", ncol=2)
    return fc.finish(fig, "Fig1", axes=[ax, bx],
                     encodes="(a) median over seeds with interquartile range; (b) one point per seed, "
                             "median (vertical bar) and interquartile range (horizontal bar); n = 30 per cell; log axes",
                     exceptions=["Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel letters, "
                                 "file stem Fig1"])


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
