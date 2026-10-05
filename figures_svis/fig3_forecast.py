# -*- coding: utf-8 -*-
"""Fig. 3: forecast error under the rolling-origin protocol (four held-out days).

Data: results/summary/forecast_metrics.csv (MAE mean and SD over the four folds).
(a) MAE per horizon for the seasonal-naive and last-value baselines, LightGBM and the 64-unit LSTM;
(b) MAE of the LSTM against hidden-unit count at horizons 1 and 30, with LightGBM as reference.
Both panels share the MAE axis so that the size of the hidden-unit effect is not exaggerated.
"""
import json

import numpy as np
import pandas as pd

import fig_config as fc
import sciviz as sv

HORIZONS = [1, 5, 15, 30]
UNITS = [16, 32, 64, 128, 256]


def main():
    pal = fc.setup()
    m = pd.read_csv(fc.SUMMARY / "forecast_metrics.csv").set_index(["model", "h"])
    folds = int(m["folds"].min())
    fig, axs = sv.figure(fc.FINAL_WIDTH_MM, aspect=0.50, ncols=2, sharey=True)
    ax, bx = axs[0]

    x = np.arange(len(HORIZONS))
    models = [("seasonal_naive", "seasonal_naive"), ("naive", "naive"), ("lightgbm", "lightgbm"), ("lstm_64", "lstm")]
    for k, (key, grp) in enumerate(models):
        st = fc.style(pal, grp)
        mae = [m.loc[(key, h), "mae"] for h in HORIZONS]
        sd = [m.loc[(key, h), "mae_sd"] for h in HORIZONS]
        ax.errorbar(x + (k - 1.5) * 0.09, mae, yerr=sd, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], markersize=6, linewidth=1.8, capsize=4, elinewidth=1.4,
                    label=fc.DISPLAY[grp])
    ax.set_xticks(x, [str(h) for h in HORIZONS])
    ax.set_xlim(-0.4, len(HORIZONS) - 0.6)
    ax.set_xlabel("Horizon (steps of 300 s)")
    ax.set_ylabel("MAE (utilisation points)")
    sv.enforce_closed_frame(ax, minor=False)
    sv.outside_legend(ax, where="top", ncol=2)

    st_l, st_g = fc.style(pal, "lstm"), fc.style(pal, "lightgbm")
    u = np.log2(UNITS)
    for j, (h, face, ls) in enumerate([(1, st_l["color"], "-"), (30, "white", "--")]):
        mae = [m.loc[(f"lstm_{n}", h), "mae"] for n in UNITS]
        sd = [m.loc[(f"lstm_{n}", h), "mae_sd"] for n in UNITS]
        bx.errorbar(u + (j - 0.5) * 0.08, mae, yerr=sd, color=st_l["color"], marker="o", markerfacecolor=face,
                    linestyle=ls, markersize=6, linewidth=1.8, capsize=4, elinewidth=1.4, label=f"LSTM, h = {h}")
        bx.axhline(m.loc[("lightgbm", h), "mae"], color=st_g["color"], linestyle=":" if h == 1 else (0, (5, 2)),
                   linewidth=1.4, zorder=0, label=f"LightGBM, h = {h}")
    bx.set_xticks(u, [str(n) for n in UNITS])
    bx.set_xlim(u[0] - 0.4, u[-1] + 0.4)
    bx.set_xlabel("LSTM hidden units per layer")
    sv.enforce_closed_frame(bx, minor=False)
    sv.outside_legend(bx, where="top", ncol=2)
    return fc.finish(fig, "Fig3", axes=[ax, bx],
                     encodes=f"MAE mean and SD over the {folds} held-out days (rolling origin); shared MAE axis; "
                             "LightGBM drawn as horizontal reference lines in (b)",
                     exceptions=["Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel "
                                 "letters, file stem Fig3", "per-panel legends above the axes (encodings differ "
                                 "between panels, so no shared legend)"])


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
