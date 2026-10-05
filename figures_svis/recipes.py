# -*- coding: utf-8 -*-
"""recipes.py - reference implementations of common publication figures.
Every recipe: synthetic demo data (seeded), full style compliance, strict QC.
Run: python recipes.py [path/to/RULES]  -> gallery/<recipe>_<white|transparent>.<pdf|tiff|png>"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import sciviz as sv

RNG = np.random.default_rng(20261003)
OUT = "gallery"


def r01_timecourse():
    """Mean +/- SD trajectories, two groups, dodged, outside legend."""
    pal = sv.active_palette()
    t = np.arange(4)
    groups = {"Garlic (n = 65)": (np.array([29.7, 28.7, 28.2, 28.2]), np.array([2.1, 1.4, 1.2, 1.2])),
              "Placebo (n = 64)": (np.array([30.1, 30.3, 30.8, 30.7]), np.array([1.9, 2.0, 1.6, 1.6]))}
    fig, ax = sv.figure(85, aspect=0.85)
    for i, (lab, (m, s)) in enumerate(groups.items()):
        st = pal.series(i)
        ax.errorbar(t + (i - 0.5) * 0.16, m, yerr=s, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], markerfacecolor=st["color"] if i == 0 else "white",
                    capsize=4, label=lab)
    ax.set(xticks=t, xlim=(-0.5, 3.5), xlabel="Menstrual cycle", ylabel="Cycle length (days)")
    sv.enforce_closed_frame(ax)
    sv.outside_legend(ax, where="top", ncol=1)
    return fig


def r02_scatter_fit():
    """Scatter + OLS fit + 95% confidence band + equation call-out."""
    x = RNG.uniform(0, 10, 80)
    y = 1.8 + 0.62 * x + RNG.normal(0, 1.1, x.size)
    res = stats.linregress(x, y)
    xs = np.linspace(0, 10, 200)
    n = x.size
    se = res.stderr * np.sqrt(np.sum((x - x.mean()) ** 2)) * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    tq = stats.t.ppf(0.975, n - 2)
    fit = res.intercept + res.slope * xs
    fig, ax = sv.figure(85, aspect=0.9)
    ax.fill_between(xs, fit - tq * se, fit + tq * se, color=sv.with_alpha(sv.get_sequential_color(1), "band_ci"),
                    linewidth=0, label="95% CI")
    ax.scatter(x, y, s=28, color=sv.with_alpha(sv.get_sequential_color(0), "scatter"), edgecolors="none", label="Observations")
    ax.plot(xs, fit, color=sv.get_sequential_color(1), label="OLS fit")
    ax.text(0.04, 0.96, f"$y = {res.intercept:.2f} + {res.slope:.2f}x$\n$r = {res.rvalue:.2f}$, $n = {n}$",
            transform=ax.transAxes, va="top", fontsize=sv.FS["L1"])
    ax.set(xlabel="Dose (mg kg$^{-1}$)", ylabel="Response (a.u.)")
    sv.enforce_closed_frame(ax)
    sv.outside_legend(ax, where="top", ncol=3)
    return fig


def r03_bars_points():
    """Bar = mean, error = 95% CI, every observation shown, stacked significance brackets."""
    data = [RNG.normal(m, 1.0, 12) for m in (4.0, 5.6, 7.1)]
    labels = ["Control", "Low dose", "High dose"]
    fig, ax = sv.figure(85, aspect=0.95)
    for i, d in enumerate(data):
        m, ci = d.mean(), stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
        ax.bar(i, m, width=0.62, color=sv.with_alpha(sv.get_sequential_color(i + 1), 0.35),
               edgecolor=sv.get_sequential_color(i + 1), linewidth=2.0)
        ax.errorbar(i, m, yerr=ci, color=sv.INK, capsize=6, lw=2, zorder=4)
        jit = RNG.uniform(-0.17, 0.17, d.size)
        ax.scatter(i + jit, d, s=22, color=sv.get_sequential_color(i + 1), zorder=5, edgecolors="white", linewidths=0.6)
    ax.set_ylim(0, 11.5)
    y = 9.2
    for a, b in ((0, 1), (1, 2), (0, 2)):
        p = stats.ttest_ind(data[a], data[b], equal_var=False).pvalue
        y = sv.sig_bracket(ax, a, b, y, sv.p_to_stars(p))
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Concentration (µmol L$^{-1}$)")
    sv.enforce_closed_frame(ax)
    return fig


def r04_raincloud():
    """Raincloud: half-violin (density) + box + jittered raw data."""
    groups = [RNG.normal(m, s, 60) for m, s in ((10, 2), (12, 2.5), (11, 1.4), (14, 3))]
    fig, ax = sv.figure(120, aspect=0.6)
    for i, d in enumerate(groups):
        col = sv.get_sequential_color(i)
        kde = stats.gaussian_kde(d)
        yy = np.linspace(d.min() - 1, d.max() + 1, 200)
        dens = kde(yy) / kde(yy).max() * 0.38
        ax.fill_betweenx(yy, i + 0.06, i + 0.06 + dens, color=sv.with_alpha(col, "hist"), linewidth=0)
        ax.boxplot(d, positions=[i - 0.02], widths=0.08, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor=sv.INK, linewidth=1.5),
                   medianprops=dict(color=sv.INK, linewidth=2), whiskerprops=dict(color=sv.INK, linewidth=1.5),
                   capprops=dict(color=sv.INK, linewidth=1.5))
        ax.scatter(i - 0.2 + RNG.uniform(-0.06, 0.06, d.size), d, s=10, color=sv.with_alpha(col, "scatter"),
                   edgecolors="none")
    ax.set_xticks(range(4), ["WT", "KO-1", "KO-2", "Rescue"])
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylabel("Latency (s)")
    sv.enforce_closed_frame(ax)
    return fig


def r05_corr_heatmap():
    """Lower-triangle correlation matrix, diverging palette map centred at 0."""
    X = RNG.normal(size=(120, 6))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 4] -= 0.7 * X[:, 2]
    C = np.corrcoef(X.T)
    names = ["IL-6", "TNF-α", "CRP", "IL-10", "PGF$_{2α}$", "BMI"]
    M = np.ma.masked_where(np.triu(np.ones_like(C, bool), k=1), C)
    fig, ax = sv.figure(120, aspect=0.82)
    im = ax.imshow(M, cmap=sv.CMAPS["div"], norm=TwoSlopeNorm(0, -1, 1))
    for i in range(6):
        for j in range(i + 1):
            L = sv.srgb_to_lab(sv.CMAPS["div"]((C[i, j] + 1) / 2))[0]
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=sv.FS["L1"],
                    color="#ffffff" if L < 50 else sv.INK)
    ax.set_xticks(range(6), names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(6), names)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Pearson r", fontsize=sv.FS["L3"], fontweight="bold")
    cb.outline.set_linewidth(sv.FRAME_LW)
    cb.ax.tick_params(width=sv.TICK_W, length=sv.TICK_LEN, labelsize=sv.FS["L2"])
    sv.enforce_closed_frame(ax)
    return fig


def r06_forest():
    """Forest plot: estimates with 95% CI, reference line, numeric column outside the axes."""
    names = ["Spotting", "Bleeding", "Cycle length", "PSST total", "Mood", "Physical"]
    g = np.array([-0.99, -0.52, -1.84, -1.81, -1.50, -0.98])
    se = np.array([0.19, 0.18, 0.21, 0.21, 0.20, 0.19])
    y = np.arange(len(names))[::-1]
    fig, ax = sv.figure(120, aspect=0.55)
    for k in range(len(names)):
        c = sv.get_sequential_color(0 if k < 3 else 1)
        ax.errorbar(g[k], y[k], xerr=1.96 * se[k], fmt=sv.active_palette().marker(0 if k < 3 else 1),
                    color=c, capsize=4, markersize=8)
        ax.annotate(f"{g[k]:.2f} ({g[k]-1.96*se[k]:.2f}, {g[k]+1.96*se[k]:.2f})", (1.02, y[k]),
                    xycoords=("axes fraction", "data"), va="center", fontsize=sv.FS["L1"])
    ax.axvline(0, color=sv.INK, lw=1.2, ls="--")
    ax.set_yticks(y, names)
    ax.set_xlabel("Hedges' g (95% CI)")
    ax.set_ylim(-0.7, len(names) - 0.3)
    sv.enforce_closed_frame(ax)
    return fig


def r07_roc():
    """ROC curves with AUC in the outside legend, chance diagonal, square axes."""
    fig, ax = sv.figure(85, aspect=0.8)
    for i, sep in enumerate((0.6, 1.2, 2.0)):
        neg, pos = RNG.normal(0, 1, 300), RNG.normal(sep, 1, 300)
        thr = np.sort(np.r_[neg, pos])[::-1]
        tpr = np.r_[0, [(pos >= t).mean() for t in thr], 1]
        fpr = np.r_[0, [(neg >= t).mean() for t in thr], 1]
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=sv.get_sequential_color(i), ls=sv.active_palette().linestyle(i),
                label=f"Model {i+1} (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=sv.INK, lw=1, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − specificity", ylabel="Sensitivity", aspect="equal")
    sv.enforce_closed_frame(ax)
    sv.outside_legend(ax, where="right")
    return fig


def r08_volcano():
    """Volcano plot: dense points rasterised, thresholds, repelled call-outs."""
    n = 6000
    lfc = RNG.normal(0, 1.2, n)
    p = 10 ** (-np.abs(lfc) * RNG.uniform(0.2, 2.4, n))
    nl = -np.log10(p)
    up, down = (lfc > 1) & (nl > 2), (lfc < -1) & (nl > 2)
    fig, ax = sv.figure(120, aspect=0.75)
    rest = ~(up | down)
    ax.scatter(lfc[rest], nl[rest], s=5, color=sv.with_alpha(sv.get_sequential_color(3), "dense"),
               edgecolors="none", rasterized=True, label="Not significant")
    ax.scatter(lfc[down], nl[down], s=8, color=sv.get_sequential_color(1), edgecolors="none", rasterized=True, label="Down")
    ax.scatter(lfc[up], nl[up], s=8, color=sv.get_sequential_color(8), edgecolors="none", rasterized=True, label="Up")
    for v in (-1, 1):
        ax.axvline(v, color=sv.INK, lw=1, ls="--")
    ax.axhline(2, color=sv.INK, lw=1, ls="--")
    top = np.argsort(-nl * (up | down))[:6]
    anns = [ax.annotate(f"Gene{k}", (lfc[k], nl[k]), xytext=(8, 0), textcoords="offset points",
                        fontsize=sv.FS["L1"], arrowprops=dict(arrowstyle="-", lw=0.8, color=sv.INK))
            for k in top]
    sv.repel_texts(fig, anns)
    ax.set(xlabel="log$_2$ fold change", ylabel="−log$_{10}$ p")
    sv.enforce_closed_frame(ax)
    sv.outside_legend(ax, where="right", markerscale=2.5)
    return fig


def r09_bland_altman():
    a = RNG.normal(50, 10, 90)
    b = a + RNG.normal(1.2, 3.0, 90)
    mean, diff = (a + b) / 2, b - a
    md, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = sv.figure(85, aspect=0.85)
    ax.scatter(mean, diff, s=24, color=sv.with_alpha(sv.get_sequential_color(1), "scatter"), edgecolors="none")
    for v, lab in ((md, "Bias"), (md + 1.96 * sd, "+1.96 SD"), (md - 1.96 * sd, "−1.96 SD")):
        ax.axhline(v, color=sv.INK, lw=1.4, ls="-" if lab == "Bias" else "--")
        ax.annotate(f"{lab}\n{v:.2f}", (1.01, v), xycoords=("axes fraction", "data"), va="center",
                    fontsize=sv.FS["L1"])
    ax.set(xlabel="Mean of methods (mmHg)", ylabel="Difference (mmHg)")
    sv.enforce_closed_frame(ax)
    return fig


def r10_contour():
    """Filled contour of a 2D field with labelled iso-lines and colour bar."""
    x, y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    z = np.exp(-((x - 1) ** 2 + y ** 2)) + 0.8 * np.exp(-((x + 1.2) ** 2 + (y - 1) ** 2) / 0.6)
    fel = -np.log(z + 1e-3)
    fel -= fel.min()
    fig, ax = sv.figure(120, aspect=0.8)
    cf = ax.contourf(x, y, fel, levels=20, cmap=sv.CMAPS["seq_low"].reversed())
    cs = ax.contour(x, y, fel, levels=[1, 2, 3, 4], colors=sv.INK, linewidths=1.0)
    ax.clabel(cs, fmt="%d", fontsize=sv.FS["L1"])
    cb = fig.colorbar(cf, ax=ax, pad=0.03)
    cb.set_label("ΔG (kcal mol$^{-1}$)", fontsize=sv.FS["L3"], fontweight="bold")
    cb.outline.set_linewidth(sv.FRAME_LW)
    cb.ax.tick_params(width=sv.TICK_W, length=sv.TICK_LEN)
    ax.set(xlabel="PC1 (nm)", ylabel="PC2 (nm)", aspect="equal")
    sv.enforce_closed_frame(ax)
    return fig


def r11_surface3d():
    """3D free-energy surface with floor projection; surface rasterised in vector output."""
    x, y = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    z = np.exp(-((x - 1) ** 2 + y ** 2)) + 0.8 * np.exp(-((x + 1.2) ** 2 + (y - 1) ** 2) / 0.6)
    fel = -np.log(z + 1e-3)
    fel -= fel.min()
    fig, ax = sv.figure(120, aspect=0.85, projection="3d")
    surf = ax.plot_surface(x, y, fel, cmap=sv.CUSTOM_CMAP, linewidth=0, antialiased=True,
                           rcount=120, ccount=120, alpha=sv.ALPHA["surface"], rasterized=True)
    ax.contourf(x, y, fel, zdir="z", offset=-1.5, levels=15, cmap=sv.CUSTOM_CMAP, alpha=0.9)
    ax.set_zlim(-1.5, fel.max())
    ax.view_init(elev=28, azim=-58)
    ax.set_xlabel("PC1 (nm)", labelpad=10)
    ax.set_ylabel("PC2 (nm)", labelpad=10)
    ax.set_zlabel("ΔG (kcal mol$^{-1}$)", labelpad=10)
    cb = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("ΔG (kcal mol$^{-1}$)", fontsize=sv.FS["L3"], fontweight="bold")
    cb.outline.set_linewidth(sv.FRAME_LW)
    return fig


def r12_trajectory3d():
    """3D trajectories coloured by series index, start/end markers, outside legend."""
    fig, ax = sv.figure(120, aspect=0.8, projection="3d")
    for i in range(3):
        t = np.linspace(0, 6 * np.pi, 600)
        xyz = np.c_[np.cos(t + i), np.sin(t + i), t / 6] + RNG.normal(0, 0.03, (600, 3)).cumsum(0) * 0.1
        c = sv.get_sequential_color(i * 3)
        ax.plot(*xyz.T, color=c, lw=1.6, label=f"Replica {i+1}")
        ax.scatter(*xyz[0], color=c, s=40, marker="o", depthshade=False)
        ax.scatter(*xyz[-1], color=c, s=50, marker="X", depthshade=False)
    ax.set_xlabel("x (nm)", labelpad=8)
    ax.set_ylabel("y (nm)", labelpad=8)
    ax.set_zlabel("z (nm)", labelpad=8)
    ax.view_init(22, -50)
    sv.outside_legend(ax, where="right")
    return fig


def r13_multipanel():
    """Small multiples with one shared figure legend outside all panels + panel labels."""
    fig, axes = sv.figure(180, aspect=0.36, ncols=3)
    t = np.linspace(0, 10, 200)
    handles = []
    for k, ax in enumerate(axes.ravel()):
        for i in range(4):
            y = np.sin(t * (0.5 + 0.2 * k) + i) * np.exp(-t / (8 + 2 * i))
            h, = ax.plot(t, y, color=sv.get_sequential_color(i), ls=sv.active_palette().linestyle(i),
                         label=f"Condition {i+1}")
            if k == 0:
                handles.append(h)
        ax.set(xlabel="Time (h)", ylabel="Signal (a.u.)" if k == 0 else None, title=["Liver", "Kidney", "Brain"][k])
        sv.enforce_closed_frame(ax)
    sv.figure_legend(fig, handles, [h.get_label() for h in handles], where="right")
    sv.panel_labels(fig, axes)
    return fig


def r14_hist_ecdf():
    """Overlapping histograms (alpha only) + ECDF in a second panel."""
    a, b = RNG.gamma(2.0, 2.0, 800), RNG.gamma(3.2, 1.6, 800)
    fig, axes = sv.figure(180, aspect=0.38, ncols=2)
    bins = np.linspace(0, 20, 41)
    for i, (d, lab) in enumerate(((a, "Cohort A"), (b, "Cohort B"))):
        c = sv.get_sequential_color(i * 5)
        axes[0, 0].hist(d, bins=bins, color=sv.with_alpha(c, "hist"), edgecolor=c, linewidth=1.0, label=lab)
        xs = np.sort(d)
        axes[0, 1].step(xs, np.arange(1, xs.size + 1) / xs.size, color=c, where="post",
                        ls=sv.active_palette().linestyle(i), label=lab)
    axes[0, 0].set(xlabel="Biomarker (ng mL$^{-1}$)", ylabel="Count")
    axes[0, 1].set(xlabel="Biomarker (ng mL$^{-1}$)", ylabel="Cumulative proportion")
    for ax in axes.ravel():
        sv.enforce_closed_frame(ax)
    h, l = axes[0, 1].get_legend_handles_labels()
    sv.figure_legend(fig, h, l, where="right")
    sv.panel_labels(fig, axes)
    return fig


RECIPES = [r01_timecourse, r02_scatter_fit, r03_bars_points, r04_raincloud, r05_corr_heatmap,
           r06_forest, r07_roc, r08_volcano, r09_bland_altman, r10_contour, r11_surface3d,
           r12_trajectory3d, r13_multipanel, r14_hist_ecdf]

if __name__ == "__main__":
    import sys
    sv.setup(rules_dir=sys.argv[1] if len(sys.argv) > 1 else None)
    failures = []
    for rec in RECIPES:
        fig = rec()
        probs = sv.qc_figure(fig, strict=False, check_palette=True)
        if probs:
            failures.append((rec.__name__, probs))
        sv.export_figure(fig, rec.__name__, OUT)
        plt.close(fig)
    print("FAILURES:", failures)
