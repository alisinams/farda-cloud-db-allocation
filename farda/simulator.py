# -*- coding: utf-8 -*-
"""Discrete-event simulation of the FARDA control loop on PlanetLab traces.

Model
-----
* Hosts: n_h physical hosts with capacity drawn from {4000, ..., 8000} MIPS and a
  SPECpower-style power curve (two archetypes, tabulated at 10 % utilisation
  steps and linearly interpolated).  Hosts differ in energy per instruction.
* Virtual machines: n_v = vms_per_host * n_h, placed round-robin, each with an
  equal share of its host's MIPS.  Every VM is bound to one PlanetLab trace
  sampled without replacement from the simulated day: the trace value at
  step k is the background utilisation of that VM's own tenant during the
  300 s step, so the capacity left for admitted requests is
  MIPS_v * max(1 - u_v(k), 0.02).
* Requests: a Poisson stream of database requests in six classes (three
  sizes x two security levels).  A request must be served by exactly one VM.
  Each VM serves its admitted requests under processor sharing (the
  time-shared scheduler): every active request receives an equal share of
  the VM's free capacity.  Latency = completion - arrival + a per-VM network
  delay.  The queue is simulated exactly with the virtual-time method and its
  state is carried across control intervals.
* Control loop: once per 300 s interval the allocator receives a forecast of
  every VM's background utilisation (LSTM, last observed value, or oracle),
  the observed backlog and a forecast of per-class arrivals, and returns a
  slot plan.  Admission is constant-time: an arriving request takes the next
  planned slot of its class; when slots are exhausted, or before the plan has
  been committed, the standard path assigns it in proportion to residual
  observed capacity.
* Accountability layer: the plan commit is modelled as an additive delay
  OH after the interval boundary during which the standard path serves
  arrivals; per-request records are committed asynchronously in batches.  In
  the adversarial regime a fraction of requests carry a tampered record; the
  record fails validation after OH and the request is rolled back and
  rescheduled through the standard path.
* Energy: host utilisation per step = background + processed request work,
  mapped through the power curve and integrated over the day.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
from numba import njit

from .data import load_day, list_days, STEP_SECONDS, STEPS_PER_DAY
import json
from .allocators import Instance, SOLVERS

DAY_SECONDS = STEP_SECONDS * STEPS_PER_DAY
SIZE_CLASSES = np.array([100.0, 400.0, 1600.0])          # million instructions
SIZE_PROBS = np.array([0.6, 0.3, 0.1])
P_SENSITIVE = 0.3
SLA_FACTOR = 4.0                                          # SLA = 4 x service time at 1000 MIPS
SLA_REF_MIPS = 1000.0
THETA = 0.8                                               # planning utilisation threshold
MIN_FREE_FRACTION = 0.02
FREE_FRACTION_NOMINAL = 0.88                              # 1 - mean PlanetLab utilisation, for load scaling
HOST_MIPS_CHOICES = np.array([4000, 5000, 6000, 7000, 8000])
POWER_CURVES = {                                          # watts at 0, 10, ..., 100 % utilisation
    "G4": np.array([86.0, 89.4, 92.6, 96.0, 99.5, 102.0, 106.0, 108.0, 112.0, 114.0, 117.0]),
    "G5": np.array([93.7, 97.0, 101.0, 105.0, 110.0, 116.0, 121.0, 125.0, 129.0, 133.0, 135.0]),
}
BATCH_COMMIT = 64                                         # records per ledger commit
CLASS_SIZE = np.repeat(SIZE_CLASSES, 2)                   # c = 2*size_idx + sec
CLASS_SEC = np.tile([0, 1], 3).astype(float)
CLASS_PROB = np.repeat(SIZE_PROBS, 2) * np.tile([1 - P_SENSITIVE, P_SENSITIVE], 3)
CLASS_SLA = SLA_FACTOR * CLASS_SIZE / SLA_REF_MIPS       # seconds
MEAN_SIZE = float(SIZE_PROBS @ SIZE_CLASSES)
N_CLASSES = 6
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FORECAST_DIR = os.path.join(RESULTS_DIR, "forecasts")


@dataclass
class RunConfig:
    seed: int
    day: str
    n_hosts: int = 10
    vms_per_host: int = 5
    forecaster: str = "lstm"            # lstm | last | oracle | lstm_m | last_m (calibrated margin)
    allocator: str = "milp"             # milp | pso | ga | aco | spread
    margin_alpha: float = 0.0           # quantile level of the forecast margin for the *_m forecasters
    weights: tuple = (1 / 3, 1 / 3, 1 / 3)
    regime: str = "steady"              # steady | surge | adversarial
    commit_delay: float = 0.05          # OH, seconds
    load_fraction: float = 0.25
    surge_factor: float = 3.0
    surge_step: int = 144
    tamper_fraction: float = 0.05
    time_limit: float = 5.0
    mip_rel_gap: float = 1e-3
    evaluations: int = 20000
    meta_params: dict = field(default_factory=dict)
    lstm_hidden: int = 64
    arrival_margin: float = 1.15
    ewma_alpha: float = 0.5
    keep_series: bool = True
    tag: str = ""

    @property
    def n_vms(self):
        return self.n_hosts * self.vms_per_host

    def name(self):
        w = "-".join(f"{x:.2f}" for x in self.weights)
        a = f"_a{self.margin_alpha:g}" if self.margin_alpha > 0 else ""
        return (f"{self.regime}_h{self.n_hosts}_{self.forecaster}_{self.allocator}_w{w}"
                f"_oh{self.commit_delay:g}{a}_s{self.seed}{('_' + self.tag) if self.tag else ''}")


def power_watts(curve: np.ndarray, util: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(util, 0, 1) * 10.0, np.arange(11), curve)


def largest_remainder(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Split integer counts (C,) over V bins in proportion to weights (V,)."""
    w = np.clip(weights, 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    x = counts[:, None] * w[None, :] / w.sum()
    fl = np.floor(x).astype(np.int64)
    rem = counts - fl.sum(axis=1)
    frac = x - fl
    order = np.argsort(-frac, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(w.size), order.shape), axis=1)
    return fl + (ranks < rem[:, None])


@njit(cache=True)
def _sift_up(keys, ids, i):
    while i > 0:
        p = (i - 1) // 2
        if keys[i] < keys[p]:
            keys[i], keys[p] = keys[p], keys[i]
            ids[i], ids[p] = ids[p], ids[i]
            i = p
        else:
            break


@njit(cache=True)
def _sift_down(keys, ids, n, i):
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < n and keys[l] < keys[m]:
            m = l
        if r < n and keys[r] < keys[m]:
            m = r
        if m == i:
            break
        keys[i], keys[m] = keys[m], keys[i]
        ids[i], ids[m] = ids[m], ids[i]
        i = m


@njit(cache=True)
def ps_serve(arr_t, arr_s, arr_id, rate, t_end, V, t, act_vdep, act_id, n_act, dep_out):
    """Processor-sharing queue of one VM over [t, t_end) at constant rate (MI/s).

    arr_t sorted arrival times, arr_s sizes (MI), arr_id global request ids.
    act_vdep / act_id hold the active jobs as a binary min-heap on the virtual
    departure time and must have room for n_act + len(arr_t) entries.
    Departure times are written to dep_out[id].  Returns (V, t, n_act, busy_time).
    """
    i = 0
    m = arr_t.shape[0]
    busy = 0.0
    while True:
        next_arr = arr_t[i] if i < m else 1e300
        if n_act > 0:
            vmin = act_vdep[0]
            t_dep = t + (vmin - V) * n_act / rate
        else:
            vmin = 1e300
            t_dep = 1e300
        if next_arr < t_end and next_arr <= t_dep:
            if n_act > 0:
                V += (next_arr - t) * rate / n_act
                busy += next_arr - t
            t = next_arr
            act_vdep[n_act] = V + arr_s[i]
            act_id[n_act] = arr_id[i]
            n_act += 1
            _sift_up(act_vdep, act_id, n_act - 1)
            i += 1
        elif t_dep < t_end:
            busy += t_dep - t
            V = vmin
            t = t_dep
            dep_out[act_id[0]] = t
            n_act -= 1
            act_vdep[0] = act_vdep[n_act]
            act_id[0] = act_id[n_act]
            if n_act > 0:
                _sift_down(act_vdep, act_id, n_act, 0)
        else:
            if n_act > 0:
                V += (t_end - t) * rate / n_act
                busy += t_end - t
            t = t_end
            break
    return V, t, n_act, busy


class Environment:
    """Everything that depends on the seed but not on the allocator."""

    def __init__(self, cfg: RunConfig, with_arrivals: bool = True):
        self.cfg = cfg
        rng = np.random.default_rng([cfg.seed, 1000])
        n_h, n_v = cfg.n_hosts, cfg.n_vms
        self.host_mips = rng.choice(HOST_MIPS_CHOICES, n_h).astype(float)
        self.host_curve = np.array(["G4" if m <= 6000 else "G5" for m in self.host_mips])
        self.vm_host = np.arange(n_v) % n_h
        self.vm_mips = self.host_mips[self.vm_host] / cfg.vms_per_host
        self.vm_shared = rng.random(n_v) < 0.5
        self.kappa = np.bincount(self.vm_host, weights=self.vm_shared.astype(float), minlength=n_h) / cfg.vms_per_host
        self.vm_net = rng.uniform(0.5e-3, 3.0e-3, n_v)                      # seconds
        p_idle = np.array([POWER_CURVES[c][0] for c in self.host_curve])
        p_max = np.array([POWER_CURVES[c][-1] for c in self.host_curve])
        self.energy_per_mi = (p_max - p_idle) / self.host_mips                 # J per MI
        cost = CLASS_SIZE[:, None] * self.energy_per_mi[self.vm_host][None, :]
        self.cost_norm = cost / cost.max()
        self.risk = CLASS_SEC[:, None] * self.kappa[self.vm_host][None, :]
        # traces
        names, mat = load_day(cfg.day)
        # sampled without replacement; only the planning-only scaling study (500 and 1000 hosts)
        # needs more machines than a PlanetLab day holds, and then samples with replacement
        idx = rng.choice(len(names), n_v, replace=n_v > len(names))
        self.vm_trace_names = [names[i] for i in idx]
        self.U = mat[idx] / 100.0                                              # (n_v, 288)
        free = np.maximum(1.0 - self.U, MIN_FREE_FRACTION)
        self.speed = self.vm_mips[:, None] * free                              # MI/s per step
        # arrivals
        rng_a = np.random.default_rng([cfg.seed, 2000])
        self.lambda0 = cfg.load_fraction * FREE_FRACTION_NOMINAL * self.vm_mips.sum() / MEAN_SIZE
        if not with_arrivals:                      # planning instances only (solver timing studies)
            t = np.empty(0)
        elif cfg.regime == "surge":
            t_step = cfg.surge_step * STEP_SECONDS
            n1 = rng_a.poisson(self.lambda0 * t_step)
            n2 = rng_a.poisson(self.lambda0 * cfg.surge_factor * (DAY_SECONDS - t_step))
            t = np.concatenate([rng_a.uniform(0, t_step, n1), rng_a.uniform(t_step, DAY_SECONDS, n2)])
        else:
            n = rng_a.poisson(self.lambda0 * DAY_SECONDS)
            t = rng_a.uniform(0, DAY_SECONDS, n)
        self.arrival = np.sort(t)
        N = self.arrival.size
        self.cls = rng_a.choice(N_CLASSES, N, p=CLASS_PROB).astype(np.int8)
        self.tamper = (rng_a.random(N) < cfg.tamper_fraction) if cfg.regime == "adversarial" else np.zeros(N, bool)
        self.bounds = np.searchsorted(self.arrival, np.arange(STEPS_PER_DAY + 1) * STEP_SECONDS)
        # forecasts of background utilisation, one per VM and step
        base = cfg.forecaster.replace("_m", "")
        self.margin_q = 0.0
        if base == "lstm":
            fold, kind = fold_for_day(cfg.day)
            fname = f"{cfg.day}_lstm{cfg.lstm_hidden}.npz" if kind == "test" else f"{cfg.day}_lstm{cfg.lstm_hidden}_val.npz"
            z = np.load(os.path.join(FORECAST_DIR, fname))
            pos = {n: i for i, n in enumerate(z["names"])}
            F = np.stack([z["forecast"][pos[n]] for n in self.vm_trace_names]).astype(float)
        elif base == "oracle":
            F = self.U.copy()
        elif base == "last":
            F = np.column_stack([self.U[:, :1], self.U[:, :-1]])
        else:
            raise ValueError(cfg.forecaster)
        if cfg.forecaster.endswith("_m"):
            if cfg.margin_alpha <= 0:
                raise ValueError("margin forecaster needs margin_alpha > 0")
            fold, kind = fold_for_day(cfg.day)
            with open(os.path.join(RESULTS_DIR, f"forecast_lstm_fold{fold}_h{cfg.lstm_hidden}.json")) as fh:
                rq = json.load(fh)["residual_quantiles"]
            self.margin_q = float(rq[f"q{int(round(cfg.margin_alpha * 100))}"][base])
            F = np.clip(F + self.margin_q, 0.0, 1.0)
        self.F = F


def fold_for_day(day: str):
    """(fold id, 'test'|'val') of a PlanetLab day under the rolling-origin folds."""
    from .forecasting import folds
    for k, (fit, val, test) in enumerate(folds(list_days())):
        if test == day:
            return k, "test"
    for k, (fit, val, test) in enumerate(folds(list_days())):
        if val == day:
            return k, "val"
    raise ValueError(f"day {day} is neither a test nor a validation day of any fold")


def build_instance(env: Environment, uhat: np.ndarray, backlog_mi: np.ndarray, demand: np.ndarray,
                   weights=(1 / 3, 1 / 3, 1 / 3)) -> Instance:
    """Planning instance for one control interval from forecast utilisation and observed backlog."""
    w1, w2, w3 = weights
    n_v = env.vm_mips.size
    b_rate = backlog_mi / STEP_SECONDS
    base_util = np.clip(uhat + b_rate / env.vm_mips, 0, 1.5)
    cap = np.clip(THETA * env.vm_mips - uhat * env.vm_mips - b_rate, 0, None)
    free_hat = env.vm_mips * np.maximum(1.0 - uhat, MIN_FREE_FRACTION)
    lat_service = CLASS_SIZE[:, None] / free_hat[None, :] + env.vm_net[None, :]
    allowed = lat_service <= CLASS_SLA[:, None]
    lat_norm = np.minimum(lat_service / CLASS_SLA[:, None], 1.0)
    coef = w1 * env.cost_norm + w2 * env.risk + w3 * lat_norm
    with np.errstate(invalid="ignore"):
        mean_lat = np.where(allowed.sum(0) > 0, (lat_norm * allowed).sum(0) / np.maximum(allowed.sum(0), 1), 0.0)
    gamma = w3 * mean_lat * demand.sum() / n_v
    return Instance(coef=coef, allowed=allowed, demand=demand, load=CLASS_SIZE / STEP_SECONDS,
                    cap=cap, mips=env.vm_mips, base_util=base_util, gamma=gamma)


def simulate(cfg: RunConfig) -> dict:
    t_wall = time.perf_counter()
    env = Environment(cfg)
    n_v = cfg.n_vms
    solver = SOLVERS.get(cfg.allocator)
    if cfg.allocator != "spread" and solver is None:
        raise ValueError(cfg.allocator)
    N = env.arrival.size
    vm_of = np.full(N, -1, dtype=np.int32)
    path = np.zeros(N, dtype=np.int8)                       # 0 plan, 1 standard path, 2 tampered rework
    dep = np.zeros(N)
    eff_all = env.arrival.copy()
    sizes_all = CLASS_SIZE[env.cls]
    # processor-sharing state per VM
    ps_V = np.zeros(n_v); ps_t = np.zeros(n_v); ps_n = np.zeros(n_v, dtype=np.int64)
    ps_vdep = [np.empty(0) for _ in range(n_v)]; ps_id = [np.empty(0, dtype=np.int64) for _ in range(n_v)]
    work = np.zeros((n_v, STEPS_PER_DAY))
    pending = []                                            # tampered requests re-entering in a later interval
    ewma = CLASS_PROB * env.lambda0 * STEP_SECONDS
    solver_rt = np.zeros(STEPS_PER_DAY); solver_gap = np.full(STEPS_PER_DAY, np.nan)
    solver_hit = np.zeros(STEPS_PER_DAY, bool); solver_obj = np.zeros(STEPS_PER_DAY)
    overflow_share = np.zeros(STEPS_PER_DAY); solver_nint = np.zeros(STEPS_PER_DAY, dtype=np.int64)
    interval_arrivals = np.zeros(STEPS_PER_DAY, dtype=np.int64)
    rng_env = np.random.default_rng([cfg.seed, 4000])

    for k in range(STEPS_PER_DAY):
        t_start, t_end = k * STEP_SECONDS, (k + 1) * STEP_SECONDS
        u_obs = env.U[:, k - 1] if k > 0 else env.U[:, 0]
        uhat = env.F[:, k]
        backlog = np.array([ps_vdep[v][:ps_n[v]].sum() - ps_n[v] * ps_V[v] for v in range(n_v)])  # MI remaining
        backlog = np.clip(backlog, 0, None)
        b_rate = backlog / STEP_SECONDS
        demand = np.ceil(cfg.arrival_margin * ewma).astype(np.int64)
        inst = build_instance(env, uhat, backlog, demand, cfg.weights)
        if cfg.allocator == "milp":
            res = solver(inst, time_limit=cfg.time_limit, mip_rel_gap=cfg.mip_rel_gap)
            solver_gap[k] = res.mip_gap; solver_hit[k] = res.extra["time_limit_hit"]; solver_nint[k] = res.extra["n_int"]
        elif cfg.allocator == "spread":
            from .allocators import PlanResult
            res = PlanResult(n=np.zeros((N_CLASSES, n_v), dtype=np.int64), overflow=demand.copy(), objective=float("nan"), runtime=0.0, status="none")
        else:
            res = solver(inst, np.random.default_rng([cfg.seed, 3000, k]), evaluations=cfg.evaluations, **cfg.meta_params)
        solver_rt[k] = res.runtime; solver_obj[k] = res.objective
        n_plan = res.n
        overflow_share[k] = res.overflow.sum() / max(demand.sum(), 1)

        # ---- admission for arrivals in this interval
        i0, i1 = env.bounds[k], env.bounds[k + 1]
        interval_arrivals[k] = i1 - i0
        a = env.arrival[i0:i1]; c = env.cls[i0:i1]
        assign = np.full(i1 - i0, -1, dtype=np.int32)
        p = np.zeros(i1 - i0, dtype=np.int8)
        active = a >= t_start + cfg.commit_delay
        planned_load = np.zeros(n_v)
        for cc in range(N_CLASSES):
            members = np.where((c == cc) & active)[0]
            slots = np.repeat(np.arange(n_v, dtype=np.int32), n_plan[cc])
            if slots.size and members.size:
                slots = rng_env.permutation(slots)
                m = min(slots.size, members.size)
                assign[members[:m]] = slots[:m]
            planned_load += n_plan[cc] * CLASS_SIZE[cc] / STEP_SECONDS
        residual = np.clip(THETA * env.vm_mips - u_obs * env.vm_mips - b_rate - planned_load, 1e-6, None)
        unassigned = np.where(assign < 0)[0]
        if unassigned.size:
            counts = np.bincount(c[unassigned], minlength=N_CLASSES)
            split = largest_remainder(counts, residual)
            for cc in range(N_CLASSES):
                members = unassigned[c[unassigned] == cc]
                if members.size:
                    assign[members] = rng_env.permutation(np.repeat(np.arange(n_v, dtype=np.int32), split[cc]))
            p[unassigned] = 1
        eff = a.copy()
        tam = np.where(env.tamper[i0:i1])[0]
        if tam.size:
            counts = np.bincount(c[tam], minlength=N_CLASSES)
            split = largest_remainder(counts, residual)
            for cc in range(N_CLASSES):
                members = tam[c[tam] == cc]
                if members.size:
                    assign[members] = rng_env.permutation(np.repeat(np.arange(n_v, dtype=np.int32), split[cc]))
            p[tam] = 2
            eff[tam] = a[tam] + cfg.commit_delay
        vm_of[i0:i1] = assign; path[i0:i1] = p; eff_all[i0:i1] = eff
        # batch for service: this interval's arrivals plus tampered carry-overs from earlier intervals
        ids = np.arange(i0, i1)
        if pending:
            ids = np.concatenate([ids, np.array(pending, dtype=np.int64)]); pending = []
        eff_b = eff_all[ids]
        late = eff_b >= t_end
        if late.any():
            pending = list(ids[late]); ids = ids[~late]; eff_b = eff_b[~late]
        vm_b = vm_of[ids]; s_b = sizes_all[ids]
        order = np.lexsort((eff_b, vm_b))
        ids, eff_b, vm_b, s_b = ids[order], eff_b[order], vm_b[order], s_b[order]
        starts = np.searchsorted(vm_b, np.arange(n_v + 1))
        # ---- processor-sharing service on every VM until the end of the interval
        for v in range(n_v):
            lo, hi = starts[v], starts[v + 1]
            m = hi - lo
            need = ps_n[v] + m
            if ps_vdep[v].size < need:
                nv_ = np.empty(max(need, 2 * ps_vdep[v].size)); nv_[:ps_n[v]] = ps_vdep[v][:ps_n[v]]; ps_vdep[v] = nv_
                ni_ = np.empty(nv_.size, dtype=np.int64); ni_[:ps_n[v]] = ps_id[v][:ps_n[v]]; ps_id[v] = ni_
            V_, t_, n_, busy = ps_serve(eff_b[lo:hi], s_b[lo:hi], ids[lo:hi], env.speed[v, k], float(t_end),
                                        ps_V[v], max(ps_t[v], float(t_start)), ps_vdep[v], ps_id[v], int(ps_n[v]), dep)
            ps_V[v], ps_t[v], ps_n[v] = V_, t_, n_
            work[v, k] = busy * env.speed[v, k]
        observed = np.bincount(c, minlength=N_CLASSES)
        ewma = cfg.ewma_alpha * observed + (1 - cfg.ewma_alpha) * ewma

    # ---- flush: jobs still active at midnight finish at the last step's rate
    if pending:
        ids = np.array(pending, dtype=np.int64); vm_b = vm_of[ids]; order = np.argsort(vm_b, kind="stable")
        ids, vm_b = ids[order], vm_b[order]
        starts = np.searchsorted(vm_b, np.arange(n_v + 1))
    else:
        starts = np.zeros(n_v + 1, dtype=np.int64); ids = np.empty(0, dtype=np.int64)
    extra_work = np.zeros(n_v)
    for v in range(n_v):
        lo, hi = starts[v], starts[v + 1]
        need = ps_n[v] + (hi - lo)
        if ps_vdep[v].size < need:
            nv_ = np.empty(need); nv_[:ps_n[v]] = ps_vdep[v][:ps_n[v]]; ps_vdep[v] = nv_
            ni_ = np.empty(need, dtype=np.int64); ni_[:ps_n[v]] = ps_id[v][:ps_n[v]]; ps_id[v] = ni_
        arr = np.sort(eff_all[ids[lo:hi]]) if hi > lo else np.empty(0)
        sz = sizes_all[ids[lo:hi]][np.argsort(eff_all[ids[lo:hi]])] if hi > lo else np.empty(0)
        idd = ids[lo:hi][np.argsort(eff_all[ids[lo:hi]])] if hi > lo else np.empty(0, dtype=np.int64)
        V_, t_, n_, busy = ps_serve(arr, sz, idd, env.speed[v, STEPS_PER_DAY - 1], 1e300,
                                    ps_V[v], max(ps_t[v], float(DAY_SECONDS)), ps_vdep[v], ps_id[v], int(ps_n[v]), dep)
        extra_work[v] = busy * env.speed[v, STEPS_PER_DAY - 1]

    # a tampered request re-enters at arrival + OH (validation fails after OH), so dep - arrival already includes OH once
    lat = dep - env.arrival + env.vm_net[vm_of]
    # ---- energy
    host_util = np.zeros((cfg.n_hosts, STEPS_PER_DAY))
    bg = env.U * env.vm_mips[:, None]
    for h in range(cfg.n_hosts):
        sel = env.vm_host == h
        host_util[h] = (bg[sel].sum(0) + work[sel].sum(0) / STEP_SECONDS) / env.host_mips[h]
    host_util = np.clip(host_util, 0, 1)
    watts = np.stack([power_watts(POWER_CURVES[c], host_util[h]) for h, c in enumerate(env.host_curve)])
    energy_kwh = float(watts.sum() * STEP_SECONDS / 3.6e6)
    energy_idle_kwh = float(sum(POWER_CURVES[c][0] for c in env.host_curve) * DAY_SECONDS / 3.6e6)
    # work processed after midnight is charged at the marginal energy of its host
    energy_kwh += float((extra_work * env.energy_per_mi[env.vm_host]).sum() / 3.6e6)

    sla = CLASS_SLA[env.cls]
    sens = CLASS_SEC[env.cls] > 0
    interval_mean_lat = np.array([lat[env.bounds[k]:env.bounds[k + 1]].mean() if env.bounds[k + 1] > env.bounds[k] else np.nan
                                  for k in range(STEPS_PER_DAY)])
    out = dict(
        config=asdict(cfg), name=cfg.name(), n_requests=int(N), lambda0=float(env.lambda0), margin_q=float(env.margin_q),
        mean_latency_ms=float(lat.mean() * 1e3), p50_latency_ms=float(np.percentile(lat, 50) * 1e3),
        p95_latency_ms=float(np.percentile(lat, 95) * 1e3), p99_latency_ms=float(np.percentile(lat, 99) * 1e3),
        sla_violation_pct=float((lat > sla).mean() * 100), standard_path_pct=float((path >= 1).mean() * 100),
        energy_kwh=energy_kwh, energy_idle_kwh=energy_idle_kwh,
        exposure_sensitive=float(env.kappa[env.vm_host[vm_of[sens]]].mean()) if sens.any() else float("nan"),
        commit_cost_index=float((N / BATCH_COMMIT) * cfg.commit_delay / DAY_SECONDS),
        mean_host_util=float(host_util.mean()), mean_bg_util=float(env.U.mean()),
        solver_runtime_median=float(np.median(solver_rt)), solver_runtime_p95=float(np.percentile(solver_rt, 95)),
        solver_runtime_p99=float(np.percentile(solver_rt, 99)), solver_runtime_max=float(solver_rt.max()),
        solver_runtime_mean=float(solver_rt.mean()),
        solver_gap_mean=float(np.nanmean(solver_gap)) if np.isfinite(solver_gap).any() else float("nan"),
        solver_gap_max=float(np.nanmax(solver_gap)) if np.isfinite(solver_gap).any() else float("nan"),
        solver_time_limit_hits=int(solver_hit.sum()), solver_int_vars_mean=float(solver_nint.mean()),
        overflow_share_mean=float(overflow_share.mean()), wall_seconds=time.perf_counter() - t_wall,
        mean_latency_by_class_ms=[float(lat[env.cls == cc].mean() * 1e3) if (env.cls == cc).any() else None for cc in range(N_CLASSES)],
        sla_violation_by_class_pct=[float((lat[env.cls == cc] > CLASS_SLA[cc]).mean() * 100) if (env.cls == cc).any() else None for cc in range(N_CLASSES)],
    )
    if cfg.regime == "surge":
        b = env.bounds[cfg.surge_step]
        out["mean_latency_before_ms"] = float(lat[:b].mean() * 1e3)
        out["mean_latency_after_ms"] = float(lat[b:].mean() * 1e3)
        out["sla_violation_before_pct"] = float((lat[:b] > sla[:b]).mean() * 100)
        out["sla_violation_after_pct"] = float((lat[b:] > sla[b:]).mean() * 100)
        e_b = watts[:, :cfg.surge_step].sum() * STEP_SECONDS / 3.6e6
        out["energy_before_kwh"] = float(e_b); out["energy_after_kwh"] = float(energy_kwh - e_b)
    if cfg.regime == "adversarial":
        out["mean_latency_tampered_ms"] = float(lat[path == 2].mean() * 1e3) if (path == 2).any() else float("nan")
        out["mean_latency_clean_ms"] = float(lat[path != 2].mean() * 1e3)
    if cfg.keep_series:
        out["series"] = dict(interval_mean_latency_ms=np.nan_to_num(interval_mean_lat * 1e3).round(3).tolist(),
                             interval_arrivals=interval_arrivals.tolist(),
                             solver_runtime=solver_rt.round(5).tolist(),
                             solver_gap=[None if not np.isfinite(g) else float(g) for g in solver_gap],
                             solver_time_limit_hit=solver_hit.astype(int).tolist(),
                             solver_objective=solver_obj.round(4).tolist(),
                             solver_int_vars=solver_nint.tolist(),
                             overflow_share=overflow_share.round(4).tolist())
    return out
