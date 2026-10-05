# -*- coding: utf-8 -*-
"""Fig. 4: objective-weight sweep at 10 hosts, every allocator under the calibrated LSTM forecast.

Data: results/summary/weight_sweep.csv (one row per allocator and weight triple, means over seeds;
column non_dominated computed on the objectives listed in report.json -> weight_sweep.objectives).
Filled markers: non-dominated triples; open markers: dominated triples. Shared logarithmic latency axis.
"""
import json

import pandas as pd
from matplotlib.lines import Line2D

import fig_config as fc
import sciviz as sv

ALLOCATORS = ["milp", "pso", "ga", "aco"]


def main():
    pal = fc.setup()
    df = pd.read_csv(fc.SUMMARY / "weight_sweep.csv")
    with open(fc.SUMMARY / "report.json") as fh:
        ws = json.load(fh)["weight_sweep"]
    objectives = ws.get("objectives", ["energy", "exposure", "latency"])
    n_seeds = int(df["n"].max())
    fig, axs = sv.figure(fc.FINAL_WIDTH_MM, aspect=0.50, ncols=2, sharey=True)
    ax, bx = axs[0]
    handles = []
    for al in ALLOCATORS:
        g = df[df.cfg_allocator == al]
        st = fc.style(pal, al)
        nd = g["non_dominated"].astype(bool)
        for axis, xcol in ((ax, "energy"), (bx, "exposure")):
            axis.scatter(g.loc[~nd, xcol], g.loc[~nd, "latency"], s=34, marker=st["marker"], facecolors="white",
                         edgecolors=[st["color"]], linewidths=1.3, zorder=2)
            axis.scatter(g.loc[nd, xcol], g.loc[nd, "latency"], s=46, marker=st["marker"], color=st["color"],
                         linewidths=0, zorder=3)
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"], linestyle="none", markersize=7,
                              label=f"{fc.DISPLAY[al]} ({len(g)} triples)"))
    handles += [Line2D([], [], color=sv.INK, marker="o", linestyle="none", markersize=7, label="Non-dominated"),
                Line2D([], [], color=sv.INK, marker="o", markerfacecolor="white", linestyle="none", markersize=7,
                       label="Dominated")]
    ax.set_yscale("log")
    ax.set_xlabel("Total energy (kWh)")
    ax.set_ylabel("Mean latency (ms)")
    bx.set_xlabel("Exposure of sensitive requests")
    for axis in (ax, bx):
        sv.enforce_closed_frame(axis, minor=False)
    sv.figure_legend(fig, handles, [h.get_label() for h in handles], where="bottom", ncol=3)
    return fc.finish(fig, "Fig4", axes=[ax, bx],
                     encodes=f"means over {n_seeds} seeds per triple; dominance on ({', '.join(objectives)}); "
                             "shared logarithmic latency axis",
                     exceptions=["Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel "
                                 "letters, file stem Fig4"])


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
