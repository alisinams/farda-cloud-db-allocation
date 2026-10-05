# -*- coding: utf-8 -*-
"""Statistics and tables for the manuscript.

Reads results/runs/*.json and the forecasting, tuning and scaling outputs and
writes results/summary/*.csv plus results/summary/report.json with every
number quoted in the paper.
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .simulator import RESULTS_DIR

RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
CELL_LABEL = {("lstm_m", "milp"): "FARDA (LSTM + MILP)", ("lstm", "milp"): "MILP, LSTM without margin",
              ("last", "milp"): "MILP, reactive", ("last_m", "milp"): "MILP, reactive with margin",
              ("oracle", "milp"): "MILP, oracle forecast", ("lstm_m", "pso"): "PSO + LSTM", ("last", "pso"): "PSO, reactive",
              ("lstm_m", "ga"): "GA + LSTM", ("last", "ga"): "GA, reactive", ("lstm_m", "aco"): "ACO + LSTM",
              ("last", "aco"): "ACO, reactive", ("last", "spread"): "Proportional spreading, no optimiser",
              ("lstm", "pso"): "PSO + LSTM without margin", ("lstm", "ga"): "GA + LSTM without margin", ("lstm", "aco"): "ACO + LSTM without margin"}
# Cells whose runs are known to be invalid. The 30 runs of this cell at 50 hosts were written on
# 2026-09-22 between 07:56 and 08:54, before the final LSTM-64 forecast files they read were
# regenerated (10:10-12:15 the same day); every other LSTM-dependent cell postdates its files.
EXCLUDED_CELLS = {(50, "MILP, LSTM without margin"): "runs used superseded LSTM forecast files (written before the final forecasts)"}
METRICS = ["mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "sla_violation_pct", "energy_kwh",
           "exposure_sensitive", "standard_path_pct", "commit_cost_index", "energy_idle_kwh"]


# ------------------------------------------------------------------ loading
def load_runs(tag: str | None = None) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(RUNS_DIR, "*.json")):
        with open(p) as fh:
            d = json.load(fh)
        cfg = d.pop("config"); d.pop("series", None)
        d.update({f"cfg_{k}": (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k != "meta_params"})
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["cfg_tag", "cfg_n_hosts", "cfg_seed", "cfg_forecaster", "cfg_allocator", "cfg_regime",
                                   "cfg_weights", "cfg_commit_delay"] + METRICS)
    if tag is not None:
        df = df[df["cfg_tag"] == tag]
    return df


# ------------------------------------------------------------------ descriptive statistics
def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    n = x.size
    m = x.mean(); sd = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return dict(n=int(n), mean=float(m), sd=float(sd), ci_low=float(m - h), ci_high=float(m + h),
                cv_pct=float(100 * sd / m) if m else float("nan"), min=float(x.min()), max=float(x.max()),
                median=float(np.median(x)))


# ------------------------------------------------------------------ inferential statistics
def welch_anova(groups: list[np.ndarray]) -> dict:
    k = len(groups)
    n = np.array([len(g) for g in groups], float); m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups]); w = n / v
    mw = (w * m).sum() / w.sum()
    num = ((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    den = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = num / den
    df1 = k - 1; df2 = (k ** 2 - 1) / (3 * lam)
    return dict(F=float(F), df1=float(df1), df2=float(df2), p=float(stats.f.sf(F, df1, df2)))


def games_howell(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    na, nb = len(a), len(b); va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    q = abs(t) * np.sqrt(2)
    p = float(stats.studentized_range.sf(q, k, df))
    return dict(diff=float(a.mean() - b.mean()), t=float(t), df=float(df), p=p)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)[:, None]; b = np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def paired_stats(a: np.ndarray, b: np.ndarray, days=None, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired summaries of d = a - b over seeds (a = FARDA): mean and median difference, percentile bootstrap
    95% interval of the mean difference, share of seeds in which a < b, matched-pairs rank-biserial
    correlation (negative when a tends to be lower) and, if `days` is given, the mean difference per trace day."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
    out = dict(mean_diff=float(d.mean()), median_diff=float(np.median(d)), ci_low=float(np.percentile(boot, 2.5)),
               ci_high=float(np.percentile(boot, 97.5)), share_a_lower=float(np.mean(d < 0)), rank_biserial=rb)
    if days is not None:
        out["per_day_mean_diff"] = {str(k): float(d[np.asarray(days) == k].mean()) for k in sorted(set(days))}
    return out


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float); m = p.size
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i]); running = max(running, val); adj[i] = running
    return adj.tolist()


def correlation_block(x: np.ndarray, y: np.ndarray, bounds=(0.30, 0.40)) -> dict:
    n = len(x); r, p = stats.pearsonr(x, y)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    ci = (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))
    tost = {}
    for b in bounds:
        zb = np.arctanh(b)
        p_low = stats.norm.sf((z + zb) / se)           # H0: r <= -b
        p_high = stats.norm.cdf((z - zb) / se)         # H0: r >= +b
        tost[f"tost_{b:.2f}"] = float(max(p_low, p_high))
    return dict(n=int(n), r=float(r), t=float(t), p_two_sided=float(p), ci95=ci, **tost)


# ------------------------------------------------------------------ Pareto and hypervolume
def pareto_front(points: np.ndarray) -> np.ndarray:
    """Non-dominated points for minimisation; returns boolean mask."""
    P = np.asarray(points, float); n = len(P)
    mask = np.ones(n, bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def hypervolume(points: np.ndarray, ref: np.ndarray, samples: int = 200000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume (minimisation) of the dominated region inside the box [0, ref]."""
    P = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    lo = P.min(axis=0)
    box = rng.uniform(lo, ref, size=(samples, P.shape[1]))
    dominated = np.zeros(samples, bool)
    for p in P:
        dominated |= np.all(box >= p, axis=1)
    return float(dominated.mean() * np.prod(ref - lo))


# ------------------------------------------------------------------ report
def summarise():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    report = {}
    df_all = load_runs()
    df_all.to_csv(os.path.join(SUMMARY_DIR, "all_runs.csv"), index=False)

    # ---- main matrix
    main = df_all[df_all["cfg_tag"] == "main"].copy()
    if len(main):
        main["cell"] = [CELL_LABEL[(f, a)] for f, a in zip(main["cfg_forecaster"], main["cfg_allocator"])]
        # cells whose runs are known to be invalid are excluded from every statistic and listed in the report
        for (nh_x, cell_x), why in EXCLUDED_CELLS.items():
            main = main[~((main.cfg_n_hosts == nh_x) & (main.cell == cell_x))]
        report["excluded_cells"] = [dict(n_hosts=k[0], cell=k[1], reason=v) for k, v in EXCLUDED_CELLS.items()]
        rows = []
        for (nh, cell), g in main.groupby(["cfg_n_hosts", "cell"]):
            for met in METRICS + ["solver_runtime_median", "solver_runtime_p95", "solver_runtime_p99", "solver_time_limit_hits", "solver_gap_mean", "wall_seconds"]:
                if met in g and g[met].notna().any():
                    d = describe(g[met].dropna().values); d.update(n_hosts=nh, cell=cell, metric=met); rows.append(d)
        desc = pd.DataFrame(rows); desc.to_csv(os.path.join(SUMMARY_DIR, "main_descriptives.csv"), index=False)
        report["main_descriptives"] = rows
        # service-level violations per request class (c = 2*size + security), mean over seeds, and per size class
        # weighted by the class probabilities (standard 0.7, sensitive 0.3)
        cls_rows = []
        for (nh, cell), g in main.groupby(["cfg_n_hosts", "cell"]):
            arr = np.array([json.loads(x) if isinstance(x, str) else x for x in g["sla_violation_by_class_pct"]], float)
            per_class = np.nanmean(arr, axis=0)
            per_size = [float(0.7 * per_class[2 * s] + 0.3 * per_class[2 * s + 1]) for s in range(3)]
            cls_rows.append(dict(n_hosts=int(nh), cell=cell, sla_pct_by_class=[float(x) for x in per_class],
                                 sla_pct_by_size=per_size))
        report["main_sla_by_class"] = cls_rows
        # per-run values for the paired comparisons
        wide = main.pivot_table(index=["cfg_n_hosts", "cfg_seed"], columns="cell", values="mean_latency_ms")
        wide.to_csv(os.path.join(SUMMARY_DIR, "main_latency_per_seed.csv"))
        tests = []
        for nh, g in main.groupby("cfg_n_hosts"):
            for met in ["mean_latency_ms", "energy_kwh", "sla_violation_pct", "p95_latency_ms", "p99_latency_ms", "exposure_sensitive"]:
                cells = ["FARDA (LSTM + MILP)", "PSO, reactive", "GA, reactive", "ACO, reactive"]
                groups = [g[g.cell == c].sort_values("cfg_seed")[met].values for c in cells]
                if any(len(x) < 3 for x in groups):
                    continue
                wa = welch_anova(groups)
                fr = stats.friedmanchisquare(*groups)
                days = g[g.cell == cells[0]].sort_values("cfg_seed")["cfg_day"].values
                pair_rows = []
                for c in cells[1:]:
                    a = groups[0]; b = groups[cells.index(c)]
                    gh = games_howell(a, b, len(cells))
                    wil = stats.wilcoxon(a, b)
                    pair_rows.append(dict(baseline=c, games_howell_p=gh["p"], diff=gh["diff"], cliffs_delta=cliffs_delta(a, b),
                                          wilcoxon_p=float(wil.pvalue), reduction_pct=float(100 * (b.mean() - a.mean()) / b.mean()),
                                          paired=paired_stats(a, b, days)))
                adj = holm([r["wilcoxon_p"] for r in pair_rows])
                for r, pa in zip(pair_rows, adj):
                    r["wilcoxon_p_holm"] = pa
                tests.append(dict(n_hosts=nh, metric=met, welch=wa, friedman=dict(chi2=float(fr.statistic), p=float(fr.pvalue)), pairs=pair_rows))
                # factorial contrasts of FARDA against the other cells (one Holm family per host count and metric)
                cells2 = ["FARDA (LSTM + MILP)", "MILP, reactive", "MILP, oracle forecast", "MILP, LSTM without margin",
                          "MILP, reactive with margin", "Proportional spreading, no optimiser", "PSO + LSTM", "GA + LSTM", "ACO + LSTM"]
                cells2 = [c for c in cells2 if len(g[g.cell == c]) >= 3]
                g2 = [g[g.cell == c].sort_values("cfg_seed")[met].values for c in cells2]
                if len(cells2) > 1 and all(len(x) == len(g2[0]) for x in g2):
                    fps = [dict(other=c, diff=float(g2[0].mean() - g2[i].mean()),
                                wilcoxon_p=float(stats.wilcoxon(g2[0], g2[i]).pvalue),
                                cliffs_delta=cliffs_delta(g2[0], g2[i]), paired=paired_stats(g2[0], g2[i], days))
                           for i, c in enumerate(cells2) if i > 0]
                    for r, pa in zip(fps, holm([r["wilcoxon_p"] for r in fps])):
                        r["wilcoxon_p_holm"] = pa
                    tests[-1]["forecaster_pairs"] = fps
        report["main_tests"] = tests
        # energy versus exposure correlation on FARDA runs, 10 hosts
        f10 = main[(main.cell == "FARDA (LSTM + MILP)") & (main.cfg_n_hosts == 10)].sort_values("cfg_seed")
        if len(f10) >= 5:
            report["energy_exposure_correlation"] = correlation_block(f10["energy_kwh"].values, f10["exposure_sensitive"].values)
            report["energy_exposure_correlation"]["n_for_tost_030"] = int(np.ceil(((stats.norm.ppf(0.95) + stats.norm.ppf(0.80)) / (np.arctanh(0.30) - abs(np.arctanh(report["energy_exposure_correlation"]["r"])))) ** 2 + 3))
        # solve-time distribution per host configuration (interval level)
        st_rows = []
        for nh in sorted(main.cfg_n_hosts.unique()):
            rts, gaps, hits = [], [], []
            for p in glob.glob(os.path.join(RUNS_DIR, f"steady_h{nh}_lstm_m_milp_*_main.json")):
                with open(p) as fh:
                    s = json.load(fh)["series"]
                rts += s["solver_runtime"]; gaps += [g for g in s["solver_gap"] if g is not None]; hits += s["solver_time_limit_hit"]
            if rts:
                rts = np.array(rts); gaps = np.array(gaps); hits = np.array(hits)
                st_rows.append(dict(n_hosts=int(nh), n_solves=int(rts.size), median_s=float(np.median(rts)), p95_s=float(np.percentile(rts, 95)),
                                    p99_s=float(np.percentile(rts, 99)), max_s=float(rts.max()), mean_gap=float(gaps.mean()),
                                    median_gap=float(np.median(gaps)), fail_pct=float(100 * np.mean(gaps >= 0.5)),
                                    max_gap=float(gaps.max()), gap_p95=float(np.percentile(gaps, 95)),
                                    time_limit_hit_pct=float(100 * hits.mean()),
                                    mean_gap_at_limit=float(gaps[hits.astype(bool)].mean()) if hits.any() else 0.0))
        report["solve_time"] = st_rows
        pd.DataFrame(st_rows).to_csv(os.path.join(SUMMARY_DIR, "solve_time.csv"), index=False)

    # ---- regimes
    reg = df_all[df_all["cfg_tag"] == "regimes"].copy()
    if len(reg):
        reg["cell"] = [CELL_LABEL[(f, a)] for f, a in zip(reg["cfg_forecaster"], reg["cfg_allocator"])]
        rows = []
        for (regime, cell), g in reg.groupby(["cfg_regime", "cell"]):
            mets = METRICS + [c for c in ["mean_latency_before_ms", "mean_latency_after_ms", "sla_violation_before_pct",
                                          "sla_violation_after_pct", "energy_before_kwh", "energy_after_kwh",
                                          "mean_latency_tampered_ms", "mean_latency_clean_ms"] if c in g and g[c].notna().any()]
            for met in mets:
                d = describe(g[met].dropna().values); d.update(regime=regime, cell=cell, metric=met); rows.append(d)
        pd.DataFrame(rows).to_csv(os.path.join(SUMMARY_DIR, "regimes_descriptives.csv"), index=False)
        report["regimes_descriptives"] = rows
        tests = []
        for regime, g in reg.groupby("cfg_regime"):
            for met in ["mean_latency_ms", "sla_violation_pct", "energy_kwh"] + (["mean_latency_after_ms", "sla_violation_after_pct"] if regime == "surge" else []):
                cells = ["FARDA (LSTM + MILP)", "PSO, reactive", "GA, reactive", "ACO, reactive"]
                groups = [g[g.cell == c].sort_values("cfg_seed")[met].values for c in cells]
                if any(len(x) < 3 for x in groups):
                    continue
                pair_rows = []
                for c in cells[1:]:
                    a = groups[0]; b = groups[cells.index(c)]
                    pair_rows.append(dict(baseline=c, diff=float(a.mean() - b.mean()), games_howell_p=games_howell(a, b, 4)["p"],
                                          wilcoxon_p=float(stats.wilcoxon(a, b).pvalue), cliffs_delta=cliffs_delta(a, b),
                                          paired=paired_stats(a, b)))
                adj = holm([r["wilcoxon_p"] for r in pair_rows])
                for r, pa in zip(pair_rows, adj):
                    r["wilcoxon_p_holm"] = pa
                tests.append(dict(regime=regime, metric=met, welch=welch_anova(groups), pairs=pair_rows))
        report["regime_tests"] = tests

    # ---- weight sweep
    ws = df_all[df_all["cfg_tag"] == "weights"].copy()
    if len(ws):
        ws["w"] = ws["cfg_weights"].apply(lambda t: tuple(round(x, 2) for x in t))
        # keep only weight triples that are complete (every seed) for every allocator
        cnt = ws.groupby(["w", "cfg_allocator"]).size().unstack(fill_value=0)
        n_seeds = int(ws["cfg_seed"].nunique())
        complete = cnt[(cnt.shape[1] == 4) & (cnt >= n_seeds).all(axis=1)].index
        ws = ws[ws["w"].isin(set(complete))]
    if len(ws):
        agg = ws.groupby(["cfg_allocator", "w"]).agg(latency=("mean_latency_ms", "mean"), energy=("energy_kwh", "mean"),
                                                     exposure=("exposure_sensitive", "mean"), sla=("sla_violation_pct", "mean"),
                                                     n=("cfg_seed", "count")).reset_index()
        # objectives to minimise: energy (cost), exposure (risk), latency.  Energy is dropped from the frontier when its
        # spread over all swept points is below 1% of its mean: min-max normalisation would otherwise stretch noise to the
        # full range and give it the weight of a real trade-off.  For the hypervolume every retained objective is min-max
        # normalised over all swept points of all allocators (latency on a log scale, since it spans orders of magnitude)
        # and the reference point is 1.1 in every normalised objective.
        energy_spread_pct = float(100 * (agg["energy"].max() - agg["energy"].min()) / agg["energy"].mean())
        # threshold: the seed-to-seed coefficient of variation of FARDA's energy at 10 hosts in the main experiment
        e10 = df_all[(df_all["cfg_tag"] == "main") & (df_all["cfg_n_hosts"] == 10) & (df_all["cfg_forecaster"] == "lstm_m")
                     & (df_all["cfg_allocator"] == "milp")]["energy_kwh"]
        energy_seed_cv_pct = float(100 * e10.std(ddof=1) / e10.mean()) if len(e10) > 2 else 1.0
        objectives = ["exposure", "latency"] if energy_spread_pct < energy_seed_cv_pct else ["energy", "exposure", "latency"]
        obj = agg[objectives].to_numpy(dtype=float).copy()
        obj[:, -1] = np.log10(obj[:, -1])
        lo_, hi_ = obj.min(axis=0), obj.max(axis=0)
        norm = (obj - lo_) / np.where(hi_ > lo_, hi_ - lo_, 1.0)
        ncols = [f"n_{o}" for o in objectives]
        for j, cname in enumerate(ncols):
            agg[cname] = norm[:, j]
        ref = np.full(len(objectives), 1.1)
        hv, fronts, front_pts = {}, {}, {}
        for al, g in agg.groupby("cfg_allocator"):
            P = g[ncols].to_numpy(dtype=float)
            m = pareto_front(g[objectives].to_numpy(dtype=float))
            fronts[al] = g[m].drop(columns=ncols).to_dict("records")
            front_pts[al] = g[m][objectives].to_numpy(dtype=float)
            hv[al] = hypervolume(P[m], ref, samples=400000)
        agg[["cfg_allocator", "w", "latency", "energy", "exposure", "sla", "n"]].assign(
            non_dominated=[bool(any(tuple(r.w) == tuple(q["w"]) for q in fronts[r.cfg_allocator])) for r in agg.itertuples()]
        ).to_csv(os.path.join(SUMMARY_DIR, "weight_sweep.csv"), index=False)
        # share of each allocator's frontier dominated by another allocator's frontier
        cross = {}
        for a in front_pts:
            for b in front_pts:
                if a == b:
                    continue
                A, Bp = front_pts[a], front_pts[b]
                dom = [bool(np.any(np.all(Bp <= x, axis=1) & np.any(Bp < x, axis=1))) for x in A]
                cross[f"{a}_by_{b}"] = float(np.mean(dom)) if dom else float("nan")
        # effect of the risk weight on the exact allocator
        mg = agg[agg.cfg_allocator == "milp"].copy()
        mg["w2"] = mg["w"].apply(lambda t: t[1])
        w2_effect = dict(low=mg[mg.w2 == 0.0][["latency", "exposure", "energy"]].mean().to_dict(),
                         high=mg[mg.w2 >= 0.6][["latency", "exposure", "energy"]].mean().to_dict(),
                         corr_w2_exposure=float(np.corrcoef(mg.w2, mg.exposure)[0, 1]) if len(mg) > 2 else float("nan"),
                         corr_w2_latency=float(np.corrcoef(mg.w2, mg.latency)[0, 1]) if len(mg) > 2 else float("nan"))
        # the neutral triple used throughout, on the same three seeds (main plan, 10 hosts)
        neutral = None
        mm = df_all[(df_all["cfg_tag"] == "main") & (df_all["cfg_n_hosts"] == 10) & (df_all["cfg_forecaster"] == "lstm_m")
                    & (df_all["cfg_allocator"] == "milp") & (df_all["cfg_seed"].isin(sorted(ws["cfg_seed"].unique())))]
        if len(mm):
            full = {"energy": mm["energy_kwh"].mean(), "exposure": mm["exposure_sensitive"].mean(), "latency": mm["mean_latency_ms"].mean()}
            npnt = np.array([full[o] for o in objectives])
            Fm = front_pts.get("milp", np.empty((0, len(objectives))))
            dominated_by = int(np.sum(np.all(Fm <= npnt, axis=1) & np.any(Fm < npnt, axis=1))) if len(Fm) else 0
            neutral = dict(energy=float(full["energy"]), exposure=float(full["exposure"]), latency=float(full["latency"]), n=int(len(mm)),
                           dominated_by_milp_front=dominated_by)
        report["weight_sweep"] = dict(hypervolume=hv, hypervolume_note="normalised objectives, log latency, reference 1.1",
                                      objectives=objectives, energy_spread_pct=energy_spread_pct, energy_seed_cv_pct=energy_seed_cv_pct,
                                      front_sizes={a: len(f) for a, f in fronts.items()}, fronts=fronts, cross_dominance=cross,
                                      n_points=int(agg.groupby("cfg_allocator").size().min()),
                                      seeds=sorted(int(x) for x in ws["cfg_seed"].unique()), w2_effect=w2_effect, neutral=neutral)

    # ---- overhead sweep
    oh = df_all[df_all["cfg_tag"] == "overhead"].copy()
    if len(oh):
        # the default OH = 0.05 s comes from the identical main-plan (steady) and regimes-plan (adversarial) FARDA runs
        dflt_s = df_all[(df_all["cfg_tag"] == "main") & (df_all["cfg_n_hosts"] == 10) & (df_all["cfg_forecaster"] == "lstm_m")
                        & (df_all["cfg_allocator"] == "milp") & (df_all["cfg_regime"] == "steady")]
        dflt_a = df_all[(df_all["cfg_tag"] == "regimes") & (df_all["cfg_forecaster"] == "lstm_m") & (df_all["cfg_allocator"] == "milp")
                        & (df_all["cfg_regime"] == "adversarial") & (df_all["cfg_seed"] < 10)]
        oh = pd.concat([oh, dflt_s, dflt_a], ignore_index=True)
        rows = []
        for (regime, val), g in oh.groupby(["cfg_regime", "cfg_commit_delay"]):
            for met in ["mean_latency_ms", "p95_latency_ms", "sla_violation_pct", "standard_path_pct", "commit_cost_index", "mean_latency_tampered_ms"]:
                if met in g and g[met].notna().any():
                    d = describe(g[met].dropna().values); d.update(regime=regime, commit_delay=val, metric=met); rows.append(d)
        pd.DataFrame(rows).to_csv(os.path.join(SUMMARY_DIR, "overhead_sweep.csv"), index=False)
        report["overhead_sweep"] = rows

    # ---- forecasting study
    frows = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "forecast_baselines_fold*.json")):
        with open(p) as fh:
            frows += json.load(fh)
    costs = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "forecast_lstm_fold*_h*.json")):
        with open(p) as fh:
            d = json.load(fh)
        frows += d["metrics"]; costs.append(d["cost"])
    if frows:
        fdf = pd.DataFrame(frows)
        fdf.to_csv(os.path.join(SUMMARY_DIR, "forecast_metrics_by_fold.csv"), index=False)
        agg = fdf.groupby(["model", "h"]).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mape=("mape", "mean"),
                                              mae_sd=("mae", "std"), rmse_sd=("rmse", "std"), mape_sd=("mape", "std"),
                                              folds=("fold", "nunique"), excluded=("mape_excluded_share", "mean")).reset_index()
        agg.to_csv(os.path.join(SUMMARY_DIR, "forecast_metrics.csv"), index=False)
        report["forecast_metrics"] = agg.to_dict("records")
        if costs:
            cdf = pd.DataFrame(costs)
            gpu = cdf[cdf.device == "cuda"] if "device" in cdf else cdf
            cagg = cdf.groupby("hidden").agg(params=("params", "first"), epochs=("epochs", "mean"),
                                             infer_ms_cpu=("infer_ms_per_1000_vms_cpu", "mean"),
                                             val_mse=("val_mse", "mean"), n_fit=("n_fit", "mean")).reset_index()
            gagg = gpu.groupby("hidden").agg(train_seconds=("train_seconds", "mean"), infer_ms_gpu=("infer_ms_per_1000_vms_gpu", "mean"),
                                             gpu_fits=("device", "count")).reset_index()
            cagg = cagg.merge(gagg, on="hidden", how="left")
            cagg.to_csv(os.path.join(SUMMARY_DIR, "forecast_cost.csv"), index=False)
            report["forecast_cost"] = cagg.to_dict("records")
        # paired comparison LSTM-64 vs seasonal naive and LightGBM across folds at each horizon
        comp = []
        for h in sorted(fdf.h.unique()):
            sub = fdf[fdf.h == h].pivot_table(index="fold", columns="model", values="mae")
            for base in ["seasonal_naive", "naive", "lightgbm", "lstm_32"]:
                if base in sub and "lstm_64" in sub:
                    comp.append(dict(h=int(h), baseline=base, lstm64_minus_base_mae=float((sub["lstm_64"] - sub[base]).mean()),
                                     folds_lstm64_better=int((sub["lstm_64"] < sub[base]).sum()), n_folds=int(len(sub))))
        report["forecast_comparisons"] = comp

    # ---- tuning and scaling
    tp = os.path.join(RESULTS_DIR, "tuning.json")
    if os.path.exists(tp):
        with open(tp) as fh:
            t = json.load(fh)
        report["tuning"] = dict(best=t["best"], n_instances=t["n_instances"], n_configs=t["n_configs"], restarts=t["restarts"],
                                evaluations=t["evaluations_per_search"], searches=t["searches_per_algorithm"], tuning_day=t["tuning_day"],
                                wall_minutes=t.get("wall_minutes"))
    sp = os.path.join(RESULTS_DIR, "solve_time_scaling.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            s = pd.DataFrame(json.load(fh))
        rows = []
        for nh, g in s.groupby("n_hosts"):
            rows.append(dict(n_hosts=int(nh), n_vms=int(g.n_vms.iloc[0]), int_vars=float(g.int_vars.mean()), slots=float(g.slots.mean()),
                             n=int(len(g)), median_s=float(g.runtime.median()), p95_s=float(g.runtime.quantile(0.95)),
                             p99_s=float(g.runtime.quantile(0.99)), hit_pct=float(100 * g.time_limit_hit.mean()),
                             mean_gap=float(g.mip_gap.mean()), max_gap=float(g.mip_gap.max()), median_gap=float(g.mip_gap.median()),
                             fail_pct=float(100 * (g.mip_gap >= 0.5).mean()),
                             overflow_share_mean=float((g.overflow / g.slots).mean()), overflow_share_max=float((g.overflow / g.slots).max())))
        report["solve_time_scaling"] = rows
        pd.DataFrame(rows).to_csv(os.path.join(SUMMARY_DIR, "solve_time_scaling.csv"), index=False)

    ip = os.path.join(RESULTS_DIR, "solve_time_isolated.json")
    if os.path.exists(ip):
        with open(ip) as fh:
            s = pd.DataFrame(json.load(fh))
        report["solve_time_isolated"] = [dict(n_hosts=int(nh), n=int(len(g)), median_s=float(g.runtime.median()), p95_s=float(g.runtime.quantile(0.95)),
                                              p99_s=float(g.runtime.quantile(0.99)), max_s=float(g.runtime.max()), hit_pct=float(100 * g.time_limit_hit.mean()),
                                              mean_gap=float(g.mip_gap.mean())) for nh, g in s.groupby("n_hosts")]
    with open(os.path.join(SUMMARY_DIR, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    return report


if __name__ == "__main__":
    r = summarise()
    print(json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} rows") for k, v in r.items()}, indent=1, default=str)[:3000])
