# -*- coding: utf-8 -*-
"""Refit one LSTM fold on the CPU (used when the GPU is saturated by other jobs)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from farda.data import list_days  # noqa: E402
from farda.forecasting import folds, run_lstm  # noqa: E402

fold = int(sys.argv[1]); hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 64
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 12
torch.set_num_threads(threads)
fit, val, test = folds(list_days())[fold]
print(f"fold {fold} on CPU with {threads} threads: fit {fit} val {val} test {test}", flush=True)
rows, cost = run_lstm(fold, hidden, fit, val, test, device="cpu")
for r in rows:
    print(r, flush=True)
print(cost, flush=True)
