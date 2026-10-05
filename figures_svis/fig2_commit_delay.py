# -*- coding: utf-8 -*-
"""Fig. 2: commit-delay (OH) sweep at 10 hosts with the FARDA allocator.

Data: results/summary/overhead_sweep.csv (mean and 95% confidence interval per OH value;
steady regime n = 30 seeds, adversarial regime n = 10 seeds). The OH axis is symmetric-log
(linear below 0.01 s) so that OH = 0 is drawn at its true position.
"""
import json

import numpy as np
import pandas as pd

import fig_config as fc
import sciviz as sv


def series(df, regime, metric):
    g = df[(df.regime == regime) & (df.metric == metric)].sort_values("commit_delay")
    return (g.commit_delay.to_numpy(float), g["mean"].to_numpy(float), g.ci_low.to_numpy(float),
            g.ci_high.to_numpy(float), int(g.n.min()), int(g.n.max()))


def main():
    pal = fc.setup()
    df = pd.read_csv(fc.SUMMARY / "overhead_sweep.csv")
    fig, axs = sv.figure(fc.FINAL_WIDTH_MM, aspect=0.44, ncols=2)
    ax, bx = axs[0]
    st = fc.style(pal, "FARDA (LSTM + MILP)")
    ticks = [0, 0.01, 0.1, 1, 10, 60]
    labels = ["0", "0.01", "0.1", "1", "10", "60"]
    for axis, (regime, metric, ylab) in zip(
            (ax, bx), [("steady", "mean_latency_ms", "Mean latency (ms)"),
                       ("adversarial", "mean_latency_tampered_ms", "Tampered latency (ms)")]):
        x, m, lo, hi, n_lo, n_hi = series(df, regime, metric)
        axis.errorbar(x, m, yerr=np.vstack([m - lo, hi - m]), color=st["color"], marker=st["marker"],
                      linestyle=st["linestyle"], markersize=6, linewidth=1.8, capsize=4, elinewidth=1.4)
        axis.set_xscale("symlog", linthresh=0.01, linscale=0.5)
        axis.set_xlim(-0.002, 90)
        axis.set_xticks(ticks, labels)
        axis.minorticks_off()
        axis.set_xlabel("Commit delay OH (s)")
        axis.set_ylabel(ylab)
        sv.enforce_closed_frame(axis, minor=False)
    bx.set_yscale("log")
    return fc.finish(fig, "Fig2", axes=[ax, bx],
                     encodes="mean and 95% confidence interval; (a) steady regime, n = 30 seeds per OH value; "
                             "(b) adversarial regime, n = 10 seeds per OH value; symmetric-log OH axis "
                             "(linear below 0.01 s)",
                     exceptions=["Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel "
                                 "letters, file stem Fig2", "no legend: single series per panel (SVIS 8, rule 1)"])


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
