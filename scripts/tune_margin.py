# -*- coding: utf-8 -*-
"""Select the quantile level of the calibrated forecast margin on the validation day.

FARDA (LSTM forecast + exact allocator) is run for full days on 20110403, the validation
day of fold 0 and a day never used for evaluation, with the margin quantile level alpha in
{none, 0.5, 0.75, 0.9, 0.95} and three seeds each, at 10 hosts.  The level with the lowest
mean latency is written to results/margin_alpha.json and used for every experiment.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from farda.simulator import RunConfig, simulate, RESULTS_DIR  # noqa: E402

ALPHAS = (0.0, 0.5, 0.75, 0.9, 0.95)
SEEDS = (100, 101, 102)
RUNS = os.path.join(RESULTS_DIR, "runs")


def cfg_for(alpha, seed):
    return RunConfig(seed=seed, day="20110403", n_hosts=10, vms_per_host=5, forecaster="lstm_m" if alpha > 0 else "lstm",
                     allocator="milp", margin_alpha=alpha, keep_series=False, tag="tunemargin")


def one(args):
    alpha, seed = args
    cfg = cfg_for(alpha, seed)
    path = os.path.join(RUNS, cfg.name() + ".json")
    if not os.path.exists(path):
        out = simulate(cfg)
        with open(path, "w") as fh:
            json.dump(out, fh)
    with open(path) as fh:
        return alpha, seed, json.load(fh)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        alpha, seed = float(sys.argv[2]), int(sys.argv[3])
        one((alpha, seed))
        sys.exit(0)
    t0 = time.time()
    os.makedirs(RUNS, exist_ok=True)
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]

    def run(job):
        a, s = job
        return subprocess.call([sys.executable, os.path.abspath(__file__), "worker", str(a), str(s)], cwd=ROOT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    with ThreadPoolExecutor(len(jobs)) as ex:
        list(ex.map(run, jobs))
    table = []
    for a in ALPHAS:
        outs = [one((a, s))[2] for s in SEEDS]
        table.append(dict(alpha=a, margin_q=outs[0]["margin_q"],
                          mean_latency_ms=sum(o["mean_latency_ms"] for o in outs) / len(outs),
                          p99_latency_ms=sum(o["p99_latency_ms"] for o in outs) / len(outs),
                          sla_violation_pct=sum(o["sla_violation_pct"] for o in outs) / len(outs),
                          energy_kwh=sum(o["energy_kwh"] for o in outs) / len(outs),
                          exposure=sum(o["exposure_sensitive"] for o in outs) / len(outs)))
    best = min(table, key=lambda r: r["mean_latency_ms"])
    with open(os.path.join(RESULTS_DIR, "margin_alpha.json"), "w") as fh:
        json.dump(dict(alpha=best["alpha"], day="20110403", seeds=list(SEEDS), table=table, wall_minutes=(time.time() - t0) / 60), fh, indent=1)
    for r in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("selected alpha:", best["alpha"])
