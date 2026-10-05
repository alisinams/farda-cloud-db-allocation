# -*- coding: utf-8 -*-
"""Rolling-origin forecasting study on the PlanetLab traces.

Protocol
--------
The ten recorded days are ordered chronologically.  Fold k fits on days
1..k-1, early-stops on day k and tests on day k+1 (expanding window, four
folds, test days 20110409, 20110411, 20110412 and 20110420).  A forecaster
sees the five preceding observations of one virtual machine (window L = 5)
and emits the next 30 steps directly; error is reported per horizon at
h = 1, 5, 15 and 30 steps (5, 25, 75 and 150 minutes).

All models are scored on the same targets: forecast origins t = 5..258 of
every test-day VM that is also present on the previous recorded day, so that
the seasonal-naive baseline (same time of day on the previous recorded day)
is defined for every target.

Metrics are in utilisation percentage points.  MAPE excludes targets below
one percentage point and the excluded share is reported.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .data import STEPS_PER_DAY, list_days, load_day, common_with_previous_day

L = 5
H_MAX = 30
HORIZONS = (1, 5, 15, 30)
HIDDEN_SIZES = (16, 32, 64, 128, 256)
ORIGINS = np.arange(L, STEPS_PER_DAY - H_MAX + 1)          # 5..258 inclusive
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FORECAST_DIR = os.path.join(RESULTS_DIR, "forecasts")


# ------------------------------------------------------------------ windows
def windows(mat: np.ndarray, origins=ORIGINS):
    """mat: (n_vm, 288) scaled to [0,1].  Returns X (N, L), Y (N, H_MAX)."""
    X = np.stack([mat[:, t - L:t] for t in origins], axis=1)
    Y = np.stack([mat[:, t:t + H_MAX] for t in origins], axis=1)
    return X.reshape(-1, L).astype(np.float32), Y.reshape(-1, H_MAX).astype(np.float32)


def fold_days(k: int, days: list[str]):
    """k = index of the test day in `days` (7..9 for four folds when 10 days)."""
    return days[:k - 1], days[k - 1], days[k]


def folds(days: list[str]):
    return [fold_days(k, days) for k in range(6, len(days))]      # 4 folds for 10 days


def eval_set(test_day: str):
    names, mat = load_day(test_day)
    keep = set(common_with_previous_day(test_day))
    idx = [i for i, n in enumerate(names) if n in keep]
    prev_day = list_days()[list_days().index(test_day) - 1]
    pnames, pmat = load_day(prev_day)
    pidx = {n: i for i, n in enumerate(pnames)}
    prev = np.stack([pmat[pidx[names[i]]] for i in idx])
    return [names[i] for i in idx], mat[idx] / 100.0, prev / 100.0


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """pred, truth in [0,1]; returns percentage-point metrics."""
    e = (pred - truth) * 100.0
    t = truth * 100.0
    mask = t >= 1.0
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mape": float(np.mean(np.abs(e[mask]) / t[mask]) * 100.0) if mask.any() else float("nan"),
        "mape_excluded_share": float(1.0 - mask.mean()),
        "n": int(e.size),
    }


def score_multi(pred: np.ndarray, Y: np.ndarray) -> list[dict]:
    """pred, Y of shape (N, H_MAX).  One metric dict per reported horizon."""
    out = []
    for h in HORIZONS:
        m = metrics(pred[:, h - 1], Y[:, h - 1])
        m["h"] = h
        out.append(m)
    return out


# ------------------------------------------------------------------ baselines
def run_baselines(fold_id: int, fit_days, val_day, test_day, out_dir=RESULTS_DIR):
    import lightgbm as lgb
    names, mat, prev = eval_set(test_day)
    X, Y = windows(mat)
    rows = []

    # naive: last observed value for every horizon
    pred = np.repeat(X[:, -1:], H_MAX, axis=1)
    for m in score_multi(pred, Y):
        rows.append(dict(fold=fold_id, test_day=test_day, model="naive", **m))

    # seasonal naive: same slot on the previous recorded day
    pred = np.stack([prev[:, t:t + H_MAX] for t in ORIGINS], axis=1).reshape(-1, H_MAX)
    for m in score_multi(pred, Y):
        rows.append(dict(fold=fold_id, test_day=test_day, model="seasonal_naive", **m))

    # LightGBM on the same five lags plus time-of-day, one model per horizon
    def feats(days_):
        Xs, Ys = [], []
        for d in days_:
            _, m_ = load_day(d)
            x, y = windows(m_ / 100.0)
            tod = np.tile(ORIGINS, m_.shape[0]).astype(np.float32)
            x = np.column_stack([x, np.sin(2 * np.pi * tod / STEPS_PER_DAY), np.cos(2 * np.pi * tod / STEPS_PER_DAY)])
            Xs.append(x); Ys.append(y)
        return np.vstack(Xs), np.vstack(Ys)
    Xf, Yf = feats(fit_days)
    Xv, Yv = feats([val_day])
    tod = np.tile(ORIGINS, mat.shape[0]).astype(np.float32)
    Xt = np.column_stack([X, np.sin(2 * np.pi * tod / STEPS_PER_DAY), np.cos(2 * np.pi * tod / STEPS_PER_DAY)])
    t0 = time.time()
    for h in HORIZONS:
        params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                      feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0, num_threads=8)
        dtrain = lgb.Dataset(Xf, Yf[:, h - 1]); dval = lgb.Dataset(Xv, Yv[:, h - 1])
        bst = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        p = bst.predict(Xt, num_iteration=bst.best_iteration)
        m = metrics(np.clip(p, 0, 1), Y[:, h - 1]); m["h"] = h
        m["best_iteration"] = int(bst.best_iteration)
        rows.append(dict(fold=fold_id, test_day=test_day, model="lightgbm", **m))
    rows[-1]["train_seconds_all_horizons"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"forecast_baselines_fold{fold_id}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


# ------------------------------------------------------------------ LSTM
def run_lstm(fold_id: int, hidden: int, fit_days, val_day, test_day, out_dir=RESULTS_DIR,
             layers=2, batch_size=512, lr=1e-3, max_epochs=100, patience=5, seed=0, device=None):
    import torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def stack(days_):
        Xs, Ys = [], []
        for d in days_:
            _, m_ = load_day(d)
            x, y = windows(m_ / 100.0)
            Xs.append(x); Ys.append(y)
        return np.vstack(Xs), np.vstack(Ys)
    Xf, Yf = stack(fit_days)
    Xv, Yv = stack([val_day])
    names, mat, prev = eval_set(test_day)
    Xt, Yt = windows(mat)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, H_MAX)

        def forward(self, x):
            out, _ = self.lstm(x.unsqueeze(-1))
            return self.head(out[:, -1])

    net = Net().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xf_t = torch.tensor(Xf, device=device); Yf_t = torch.tensor(Yf, device=device)
    Xv_t = torch.tensor(Xv, device=device); Yv_t = torch.tensor(Yv, device=device)
    n = Xf_t.shape[0]
    best, best_state, bad, history = float("inf"), None, 0, []
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xf_t[idx]), Yf_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.functional.mse_loss(net(Xv_t[i:i + 65536]), Yv_t[i:i + 65536]).item()
                                for i in range(0, Xv_t.shape[0], 65536)]))
        history.append(vl)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_seconds = time.time() - t0
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        out = []
        with torch.no_grad():
            Xt_ = torch.tensor(X, device=device)
            for i in range(0, Xt_.shape[0], 65536):
                out.append(net(Xt_[i:i + 65536]).cpu().numpy())
        return np.clip(np.vstack(out), 0, 1)

    pred = predict(Xt)
    rows = []
    for m in score_multi(pred, Yt):
        rows.append(dict(fold=fold_id, test_day=test_day, model=f"lstm_{hidden}", hidden=hidden, **m))

    # inference cost: one forecast for 1000 virtual machines
    xb = torch.tensor(Xt[:1000], device=device)
    with torch.no_grad():
        for _ in range(3):
            net(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(20):
            net(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms_gpu = (time.time() - t1) / 20 * 1000
    net_cpu = Net().to("cpu"); net_cpu.load_state_dict({k: v.cpu() for k, v in best_state.items()}); net_cpu.eval()
    xb_cpu = torch.tensor(Xt[:1000])
    with torch.no_grad():
        net_cpu(xb_cpu)
        t1 = time.time()
        for _ in range(5):
            net_cpu(xb_cpu)
        infer_ms_cpu = (time.time() - t1) / 5 * 1000
    cost = dict(fold=fold_id, test_day=test_day, hidden=hidden, layers=layers, params=int(n_params),
                epochs=len(history), best_epoch=int(np.argmin(history)) + 1, train_seconds=train_seconds,
                val_mse=best, infer_ms_per_1000_vms_gpu=infer_ms_gpu, infer_ms_per_1000_vms_cpu=infer_ms_cpu,
                n_fit=int(n), device=device, batch_size=batch_size, lr=lr)

    # one-step-ahead forecasts for every VM of the test day (used by the simulator) and of the
    # validation day (used to calibrate the forecast margin and to select its level)
    def one_step(day):
        all_names, all_mat = load_day(day)
        A = all_mat / 100.0
        F = np.empty_like(A)
        F[:, 0] = A[:, 0]
        F[:, 1:L] = A[:, :L - 1]                       # too little history: last value
        origins_all = np.arange(L, STEPS_PER_DAY)
        Xa = np.stack([A[:, t - L:t] for t in origins_all], axis=1).reshape(-1, L).astype(np.float32)
        F[:, L:] = predict(Xa)[:, 0].reshape(A.shape[0], len(origins_all))
        return all_names, A, F
    os.makedirs(FORECAST_DIR, exist_ok=True)
    tn, tA, tF = one_step(test_day)
    np.savez_compressed(os.path.join(FORECAST_DIR, f"{test_day}_lstm{hidden}.npz"), names=np.array(tn), forecast=tF.astype(np.float32))
    vn, vA, vF = one_step(val_day)
    np.savez_compressed(os.path.join(FORECAST_DIR, f"{val_day}_lstm{hidden}_val.npz"), names=np.array(vn), forecast=vF.astype(np.float32))
    resid_lstm = (vA[:, L:] - vF[:, L:]).ravel()
    resid_last = (vA[:, 1:] - vA[:, :-1]).ravel()
    residual_quantiles = {f"q{int(round(q * 100))}": {"lstm": float(np.quantile(resid_lstm, q)), "last": float(np.quantile(resid_last, q))}
                          for q in (0.5, 0.75, 0.9, 0.95)}
    cost["val_day"] = val_day
    cost["residual_quantiles"] = residual_quantiles

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"forecast_lstm_fold{fold_id}_h{hidden}.json"), "w") as fh:
        json.dump({"metrics": rows, "cost": cost, "val_history": history, "val_day": val_day,
                   "residual_quantiles": residual_quantiles}, fh, indent=1)
    return rows, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="fold id 0..3")
    ap.add_argument("--hidden", type=int, default=None, help="hidden units; omit for baselines")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    days = list_days()
    fit_days, val_day, test_day = folds(days)[a.fold]
    print(f"fold {a.fold}: fit {fit_days} val {val_day} test {test_day}", flush=True)
    if a.baselines:
        rows = run_baselines(a.fold, fit_days, val_day, test_day)
        for r in rows:
            print(r, flush=True)
    if a.hidden is not None:
        rows, cost = run_lstm(a.fold, a.hidden, fit_days, val_day, test_day)
        for r in rows:
            print(r, flush=True)
        print(cost, flush=True)


if __name__ == "__main__":
    main()
