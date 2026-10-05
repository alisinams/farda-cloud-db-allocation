# -*- coding: utf-8 -*-
"""Launch the rolling-origin forecasting study: 4 folds x 5 hidden sizes + baselines."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)
CONCURRENCY = int(os.environ.get("FARDA_FORECAST_JOBS", "5"))

jobs = []
for fold in range(4):
    jobs.append(("baselines", fold, None, [sys.executable, "-m", "farda.forecasting", "--fold", str(fold), "--baselines"]))
for hidden in (256, 128, 64, 32, 16):
    for fold in range(4):
        jobs.append(("lstm", fold, hidden, [sys.executable, "-m", "farda.forecasting", "--fold", str(fold), "--hidden", str(hidden)]))


def run(job):
    kind, fold, hidden, cmd = job
    name = f"forecast_{kind}_fold{fold}" + (f"_h{hidden}" if hidden else "")
    t0 = time.time()
    with open(os.path.join(LOGS, name + ".log"), "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return name, rc, time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for name, rc, dt in ex.map(run, jobs):
            print(f"{name}: rc={rc} {dt/60:.1f} min", flush=True)
    print(f"all done in {(time.time()-t0)/60:.1f} min", flush=True)
