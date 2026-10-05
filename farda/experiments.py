# -*- coding: utf-8 -*-
"""Experiment matrix.  Every run is one simulated day; results are saved as JSON.

Plans
-----
main      steady regime, 10/30/50 hosts, 9 forecaster x allocator cells, 30 seeds
regimes   surge and adversarial regimes, 10 hosts, 9 cells, 30 seeds
weights   objective-weight simplex (step 0.1, 66 points), 10 hosts, LSTM forecaster,
          all four allocators, 3 seeds
overhead  commit-delay sweep, 10 hosts, FARDA, steady (30 seeds) and adversarial (10 seeds)
scale     solve-time scaling of the exact programme at 100 to 1000 hosts (planning instances only)

Seeds 0..29 are spread round-robin over the four held-out days, so the same
seed always means the same day, host draw, trace sample and arrival stream
for every cell.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .simulator import RunConfig, simulate, Environment, build_instance, STEP_SECONDS, CLASS_PROB, RESULTS_DIR
from .allocators import solve_milp

TEST_DAYS = ("20110409", "20110411", "20110412", "20110420")
HOSTS = ((10, 5), (30, 5), (50, 4))
CELLS = (("lstm_m", "milp"), ("lstm", "milp"), ("last", "milp"), ("last_m", "milp"), ("oracle", "milp"),
         ("lstm_m", "pso"), ("last", "pso"), ("lstm_m", "ga"), ("last", "ga"), ("lstm_m", "aco"), ("last", "aco"),
         ("last", "spread"))
REGIME_CELLS = (("lstm_m", "milp"), ("last", "milp"), ("oracle", "milp"), ("last", "pso"), ("last", "ga"), ("last", "aco"), ("last", "spread"))
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
OH_VALUES = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0)


def load_margin_alpha() -> float:
    p = os.path.join(RESULTS_DIR, "margin_alpha.json")
    if not os.path.exists(p):
        raise FileNotFoundError("run scripts/tune_margin.py first")
    with open(p) as fh:
        return float(json.load(fh)["alpha"])


def load_meta_params() -> dict:
    p = os.path.join(RESULTS_DIR, "tuning.json")
    if not os.path.exists(p):
        raise FileNotFoundError("run farda.tuning first")
    with open(p) as fh:
        best = json.load(fh)["best"]
    return {a: best[a]["params"] for a in best}


def simplex_grid(step=0.1):
    n = int(round(1 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((round(i / n, 2), round(j / n, 2), round(k / n, 2)))
    return pts


def day_for(seed: int) -> str:
    return TEST_DAYS[seed % len(TEST_DAYS)]


def build_plan(name: str, meta: dict, seeds=range(30)) -> list[RunConfig]:
    cfgs = []
    alpha = load_margin_alpha()

    def mk(fc, al, **kw):
        return RunConfig(forecaster=fc, allocator=al, margin_alpha=alpha if fc.endswith("_m") else 0.0,
                         meta_params=meta.get(al, {}), **kw)
    if name == "main":
        for (nh, vph), (fc, al), s in itertools.product(HOSTS, CELLS, seeds):
            cfgs.append(mk(fc, al, seed=s, day=day_for(s), n_hosts=nh, vms_per_host=vph, tag="main"))
    elif name == "regimes":
        # surge: every regime cell; adversarial: FARDA only (the paper reports the tampered/clean split of FARDA)
        for (fc, al), s in itertools.product(REGIME_CELLS, seeds):
            cfgs.append(mk(fc, al, seed=s, day=day_for(s), n_hosts=10, vms_per_host=5, regime="surge", tag="regimes"))
        for s in seeds:
            cfgs.append(mk("lstm_m", "milp", seed=s, day=day_for(s), n_hosts=10, vms_per_host=5, regime="adversarial", tag="regimes"))
    elif name in ("weights", "weights_fine"):
        coarse = set(simplex_grid(0.2))
        pts = sorted(coarse) if name == "weights" else [w for w in simplex_grid(0.1) if w not in coarse]
        for w, al, s in itertools.product(pts, ("milp", "pso", "ga", "aco"), range(3)):
            cfgs.append(mk("lstm_m", al, seed=s, day=day_for(s), n_hosts=10, vms_per_host=5, weights=w, keep_series=False, tag="weights"))
    elif name == "overhead":
        # the default OH = 0.05 s is not re-run: those runs are the main-plan FARDA runs at 10 hosts (steady)
        # and the regimes-plan FARDA runs (adversarial, seeds 0-9); farda.analysis merges them in
        for oh, s in itertools.product([v for v in OH_VALUES if v != 0.05], seeds):
            cfgs.append(mk("lstm_m", "milp", seed=s, day=day_for(s), n_hosts=10, vms_per_host=5, commit_delay=oh, keep_series=False, tag="overhead"))
        for oh, s in itertools.product([v for v in OH_VALUES if v != 0.05], range(10)):
            cfgs.append(mk("lstm_m", "milp", seed=s, day=day_for(s), n_hosts=10, vms_per_host=5, regime="adversarial", commit_delay=oh,
                           keep_series=False, tag="overhead"))
    else:
        raise ValueError(name)
    return cfgs


def run_one(cfg: RunConfig) -> dict:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, cfg.name() + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            return {k: v for k, v in json.load(fh).items() if k != "series"}
    try:
        out = simulate(cfg)
    except Exception:
        return dict(name=cfg.name(), error=traceback.format_exc())
    with open(path, "w") as fh:
        json.dump(out, fh)
    return {k: v for k, v in out.items() if k != "series"}


def plan_configs(name: str, seeds=range(30)) -> list[RunConfig]:
    meta = load_meta_params() if name != "overhead" else {}
    cfgs = build_plan(name, meta, seeds)
    cfgs.sort(key=lambda c: (-c.n_hosts, c.allocator != "milp", c.allocator == "spread", c.seed))     # heaviest first
    return cfgs


def worker(name: str, index: int, stride: int, seeds=range(30)):
    cfgs = plan_configs(name, seeds)
    mine = cfgs[index::stride]
    t0 = time.time()
    for j, cfg in enumerate(mine):
        out = run_one(cfg)
        msg = (f"[{j+1}/{len(mine)} {(time.time()-t0)/60:.1f} min] {out.get('name')} "
               + (f"ERROR {out['error'][-400:]}" if "error" in out else
                  f"lat {out['mean_latency_ms']:.1f} ms sla {out['sla_violation_pct']:.2f}% "
                  f"E {out['energy_kwh']:.2f} kWh wall {out['wall_seconds']:.0f}s"))
        print(msg, flush=True)


def run_plan(name: str, processes: int, seeds=range(30)):
    cfgs = plan_configs(name, seeds)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = os.path.join(root, "logs"); os.makedirs(logs, exist_ok=True)
    print(f"plan {name}: {len(cfgs)} runs on {processes} worker processes", flush=True)
    t0 = time.time()

    def run(i):
        with open(os.path.join(logs, f"plan_{name}_worker{i}.log"), "w", encoding="utf-8") as fh:
            return subprocess.call([sys.executable, "-m", "farda.experiments", "worker", "--plan", name, "--index", str(i),
                                    "--stride", str(processes), "--seeds", str(len(seeds))],
                                   cwd=root, stdout=fh, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                        "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"})
    with ThreadPoolExecutor(processes) as ex:
        rcs = list(ex.map(run, range(processes)))
    n_done = len([c for c in cfgs if os.path.exists(os.path.join(RUNS_DIR, c.name() + ".json"))])
    print(f"plan {name} finished in {(time.time()-t0)/60:.1f} min; {n_done}/{len(cfgs)} result files present; rcs {set(rcs)}", flush=True)


# ------------------------------------------------------------------ solve-time scaling
def _scale_job(job):
    """All six planning instances of one (host count, seed) pair, planned exactly as FARDA plans them."""
    n_hosts, seed = job
    vph = 4
    alpha = load_margin_alpha()
    cfg = RunConfig(seed=seed, day=day_for(seed), n_hosts=n_hosts, vms_per_host=vph, forecaster="lstm_m", allocator="milp",
                    margin_alpha=alpha)
    env = Environment(cfg, with_arrivals=False)
    demand = np.ceil(cfg.arrival_margin * CLASS_PROB * env.lambda0 * STEP_SECONDS).astype(np.int64)
    rows = []
    for k in SCALE_INTERVALS:
        inst = build_instance(env, env.F[:, k], np.zeros(env.vm_mips.size), demand, cfg.weights)
        res = solve_milp(inst, time_limit=5.0, mip_rel_gap=cfg.mip_rel_gap)
        rows.append(dict(n_hosts=n_hosts, n_vms=cfg.n_vms, seed=seed, interval=k, slots=int(demand.sum()), int_vars=res.extra["n_int"],
                         rows=res.extra["n_rows"], runtime=res.runtime, status=res.status, mip_gap=res.mip_gap,
                         time_limit_hit=bool(res.extra["time_limit_hit"]), overflow=int(res.overflow.sum())))
    return rows


SCALE_INTERVALS = (12, 60, 108, 156, 204, 252)


def scale_jobs():
    return [(nh, s) for nh in (100, 200, 500, 1000) for s in range(10)]


def scale_worker(index: int, stride: int):
    rows = []
    for i, j in enumerate(scale_jobs()):
        if i % stride == index:
            rows += _scale_job(j)
            print(f"scale job {j} done", flush=True)
    with open(os.path.join(RESULTS_DIR, f"solve_time_scaling_part{index}.json"), "w") as fh:
        json.dump(rows, fh)


def run_scale(processes: int):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = os.path.join(root, "logs"); os.makedirs(logs, exist_ok=True)
    t0 = time.time()

    def run(i):
        with open(os.path.join(logs, f"scale_worker{i}.log"), "w", encoding="utf-8") as fh:
            return subprocess.call([sys.executable, "-m", "farda.experiments", "scale-worker", "--index", str(i), "--stride", str(processes)],
                                   cwd=root, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    with ThreadPoolExecutor(processes) as ex:
        list(ex.map(run, range(processes)))
    rows = []
    for i in range(processes):
        p = os.path.join(RESULTS_DIR, f"solve_time_scaling_part{i}.json")
        if os.path.exists(p):
            with open(p) as fh:
                rows += json.load(fh)
            os.remove(p)
    with open(os.path.join(RESULTS_DIR, "solve_time_scaling.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"scale study: {len(rows)} solves in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["main", "regimes", "weights", "weights_fine", "overhead", "scale", "worker", "scale-worker"])
    ap.add_argument("--plan", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--processes", type=int, default=28)
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    if a.command == "worker":
        worker(a.plan, a.index, a.stride, range(a.seeds))
    elif a.command == "scale-worker":
        scale_worker(a.index, a.stride)
    elif a.command == "scale":
        run_scale(a.processes)
    else:
        run_plan(a.command, a.processes, range(a.seeds))


if __name__ == "__main__":
    main()
