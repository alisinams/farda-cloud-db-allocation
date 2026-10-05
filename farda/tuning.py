# -*- coding: utf-8 -*-
"""Equal-budget random-search tuning of the PSO, GA and ACO comparators.

Tuning set: 24 planning instances drawn from the tuning day 20110403, which is
not one of the four evaluation days.  Two seeds x two host counts (10, 30)
x six control intervals spread over the day, with the last-observed-value
forecaster.  Every algorithm gets the same budget: 30 sampled configurations,
each scored on all 24 instances with two independent restarts of the
20,000-evaluation search, i.e. 1,440 searches per algorithm.  The score is the
mean relative gap between the objective reached and the exact optimum of the
same instance returned by HiGHS.  The best configuration by mean gap is used
for every experiment reported in the manuscript.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .allocators import Instance, objective, solve_milp, solve_pso, solve_ga, solve_aco, CAPACITY_PENALTY
from .simulator import RunConfig, Environment, build_instance, STEP_SECONDS, CLASS_PROB, RESULTS_DIR

TUNING_DAY = "20110403"
N_CONFIGS = 30
RESTARTS = 2
EVALUATIONS = 20000
INTERVALS = (12, 60, 108, 156, 204, 252)


def sample_config(algo: str, rng: np.random.Generator) -> dict:
    if algo == "pso":
        return dict(swarm=int(rng.choice([20, 30, 50, 80, 100])), w=float(rng.uniform(0.4, 0.9)),
                    c1=float(rng.uniform(0.5, 2.5)), c2=float(rng.uniform(0.5, 2.5)), vmax=float(rng.uniform(0.05, 0.5)))
    if algo == "ga":
        return dict(pop=int(rng.choice([20, 30, 50, 80, 100])), tournament=int(rng.choice([2, 3, 5])),
                    pc=float(rng.uniform(0.6, 1.0)), pm=float(10 ** rng.uniform(-3, -1.3)),
                    sigma=float(rng.uniform(0.02, 0.3)), blx_alpha=float(rng.uniform(0.0, 0.5)), elites=int(rng.choice([1, 2, 5])))
    if algo == "aco":
        return dict(ants=int(rng.choice([10, 20, 30, 50])), alpha=float(rng.uniform(0.5, 2.0)), beta=float(rng.uniform(0.5, 5.0)),
                    rho=float(rng.uniform(0.02, 0.5)), q=float(10 ** rng.uniform(-1, 1)))
    raise ValueError(algo)


def tuning_instances() -> list[tuple[str, Instance, float]]:
    """(label, instance, exact objective) triples."""
    out = []
    for seed in (100, 101):
        for n_hosts, vph in ((10, 5), (30, 5)):
            cfg = RunConfig(seed=seed, day=TUNING_DAY, n_hosts=n_hosts, vms_per_host=vph, forecaster="last", allocator="milp")
            env = Environment(cfg)
            for k in INTERVALS:
                uhat = env.F[:, k]
                demand = np.ceil(cfg.arrival_margin * CLASS_PROB * env.lambda0 * STEP_SECONDS).astype(np.int64)
                inst = build_instance(env, uhat, np.zeros(env.vm_mips.size), demand, cfg.weights)
                res = solve_milp(inst, time_limit=30.0, mip_rel_gap=1e-4)
                ref = float(objective(inst, res.n, res.overflow, capacity_penalty=CAPACITY_PENALTY))
                out.append((f"s{seed}_h{n_hosts}_k{k}", inst, ref))
    return out


_INSTANCES = None


def _init():
    global _INSTANCES
    _INSTANCES = tuning_instances()


def _score(job):
    algo, cid, params = job
    fn = {"pso": solve_pso, "ga": solve_ga, "aco": solve_aco}[algo]
    gaps, rts = [], []
    for label, inst, ref in _INSTANCES:
        for r in range(RESTARTS):
            rng = np.random.default_rng([7, cid, r, hash(label) % 100000])
            res = fn(inst, rng, evaluations=EVALUATIONS, **params)
            val = float(objective(inst, res.n, None, capacity_penalty=CAPACITY_PENALTY))
            gaps.append((val - ref) / abs(ref)); rts.append(res.runtime)
    return dict(algo=algo, config_id=cid, params=params, mean_gap=float(np.mean(gaps)),
                median_gap=float(np.median(gaps)), max_gap=float(np.max(gaps)), mean_runtime=float(np.mean(rts)))


def all_jobs():
    jobs = []
    for algo in ("pso", "ga", "aco"):
        rng = np.random.default_rng({"pso": 11, "ga": 12, "aco": 13}[algo])
        for cid in range(N_CONFIGS):
            jobs.append((algo, cid, sample_config(algo, rng)))
    return jobs


def worker(index: int, stride: int):
    """Score jobs index, index+stride, ... and write a partial result file."""
    _init()
    rows = []
    for j, job in enumerate(all_jobs()):
        if j % stride != index:
            continue
        r = _score(job); rows.append(r)
        print(f"{r['algo']} config {r['config_id']} mean gap {r['mean_gap']:.4f}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"tuning_part{index}.json"), "w") as fh:
        json.dump(rows, fh)


def merge(t0=None):
    rows = []
    for p in sorted(os.listdir(RESULTS_DIR)):
        if p.startswith("tuning_part") and p.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, p)) as fh:
                rows += json.load(fh)
    best = {}
    for algo in ("pso", "ga", "aco"):
        cand = sorted([r for r in rows if r["algo"] == algo], key=lambda r: r["mean_gap"])
        if cand:
            best[algo] = cand[0]
    inst = tuning_instances()
    summary = dict(tuning_day=TUNING_DAY, n_instances=len(inst), n_configs=N_CONFIGS, restarts=RESTARTS,
                   evaluations_per_search=EVALUATIONS, searches_per_algorithm=N_CONFIGS * len(inst) * RESTARTS,
                   instances=[dict(label=l, C=i.C, V=i.V, slots=int(i.demand.sum()), exact=ref) for l, i, ref in inst],
                   best=best, all=rows, wall_minutes=((time.time() - t0) / 60) if t0 else None)
    with open(os.path.join(RESULTS_DIR, "tuning.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for algo, b in best.items():
        print(algo, b["params"], "mean gap", round(b["mean_gap"], 4), "median", round(b["median_gap"], 4), flush=True)
    return summary


def main(processes: int = 24):
    """Launch `processes` worker subprocesses, wait, merge."""
    t0 = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = os.path.join(root, "logs"); os.makedirs(logs, exist_ok=True)

    def run(i):
        with open(os.path.join(logs, f"tuning_worker{i}.log"), "w", encoding="utf-8") as fh:
            return subprocess.call([sys.executable, "-m", "farda.tuning", "worker", str(i), str(processes)],
                                   cwd=root, stdout=fh, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    with ThreadPoolExecutor(processes) as ex:
        rcs = list(ex.map(run, range(processes)))
    print("worker return codes:", rcs, flush=True)
    merge(t0)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
