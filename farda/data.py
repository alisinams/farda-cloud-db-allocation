# -*- coding: utf-8 -*-
"""PlanetLab trace loading.

The trace archive distributed with CloudSim / CloudSim Plus
(https://github.com/cloudsimplus/planetlab-workload-traces) holds ten
directories named by date.  Every file inside a directory is one PlanetLab
virtual machine and contains 288 integers, the CPU utilisation percentage
sampled every 300 s over one day.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

STEP_SECONDS = 300
STEPS_PER_DAY = 288

DEFAULT_ROOT = os.environ.get(
    "PLANETLAB_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "planetlab-workload-traces-master"),
)


def list_days(root: str = DEFAULT_ROOT) -> list[str]:
    days = sorted(d for d in os.listdir(root)
                  if d.startswith("2011") and os.path.isdir(os.path.join(root, d)))
    if not days:
        raise FileNotFoundError(f"no PlanetLab day directories under {root}")
    return days


@lru_cache(maxsize=None)
def load_day(day: str, root: str = DEFAULT_ROOT) -> tuple[list[str], np.ndarray]:
    """Return (vm_names, matrix) with matrix of shape (n_vm, 288) in percent."""
    d = os.path.join(root, day)
    names, rows = [], []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            vals = [float(x.strip()) for x in fh if x.strip()]
        if len(vals) != STEPS_PER_DAY:
            continue
        names.append(f)
        rows.append(vals)
    return names, np.asarray(rows, dtype=np.float64)


def load_all(root: str = DEFAULT_ROOT) -> dict[str, tuple[list[str], np.ndarray]]:
    return {d: load_day(d, root) for d in list_days(root)}


def common_with_previous_day(day: str, root: str = DEFAULT_ROOT) -> list[str]:
    """VM names present both on `day` and on the previous recorded day."""
    days = list_days(root)
    i = days.index(day)
    if i == 0:
        return []
    prev_names = set(load_day(days[i - 1], root)[0])
    return [n for n in load_day(day, root)[0] if n in prev_names]
