# -*- coding: utf-8 -*-
"""Re-measure exact-solver wall-clock time in isolation (one process, quiet machine).

The main experiment measured solve times with up to 28 concurrent single-threaded
solver processes.  This script solves the same kind of planning instances (LSTM
forecast, no backlog, two seeds, every twelfth control interval) sequentially in a
single process and records runtime, gap and time-limit hits per host count, so that
the manuscript can quote an uncontended reference distribution.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from farda.simulator import RunConfig, Environment, build_instance, STEP_SECONDS, CLASS_PROB, RESULTS_DIR  # noqa: E402
from farda.allocators import solve_milp  # noqa: E402
from farda.experiments import day_for, HOSTS, load_margin_alpha  # noqa: E402

rows = []
t0 = time.time()
for nh, vph in HOSTS:
    for seed in (0, 1):
        cfg = RunConfig(seed=seed, day=day_for(seed), n_hosts=nh, vms_per_host=vph, forecaster="lstm_m", allocator="milp",
                        margin_alpha=load_margin_alpha())
        env = Environment(cfg, with_arrivals=False)
        demand = np.ceil(cfg.arrival_margin * CLASS_PROB * env.lambda0 * STEP_SECONDS).astype(np.int64)
        for k in range(0, 288, 12):
            inst = build_instance(env, env.F[:, k], np.zeros(env.vm_mips.size), demand, cfg.weights)
            res = solve_milp(inst, time_limit=cfg.time_limit, mip_rel_gap=cfg.mip_rel_gap)
            rows.append(dict(n_hosts=nh, seed=seed, interval=k, runtime=res.runtime, mip_gap=res.mip_gap,
                             time_limit_hit=bool(res.extra["time_limit_hit"]), int_vars=res.extra["n_int"]))
        print(f"hosts {nh} seed {seed}: median {np.median([r['runtime'] for r in rows if r['n_hosts'] == nh]):.2f} s", flush=True)
with open(os.path.join(RESULTS_DIR, "solve_time_isolated.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
print(f"done: {len(rows)} solves in {(time.time() - t0) / 60:.1f} min")
