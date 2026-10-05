# -*- coding: utf-8 -*-
"""Allocation layer: the exact mixed-integer programme and three metaheuristic comparators.

Planning instance
-----------------
At the start of every control interval the controller has, for each request
class c (size class x security level, C = 6) and virtual machine v:

  coef[c, v]   per-slot objective coefficient  w1*Cost + w2*Risk + w3*Latency, in [0, 1]
  allowed[c,v] False when the predicted service time on v exceeds the SLA of class c
  demand[c]    forecast number of class-c requests in the interval (slots to plan)
  load[c]      MIPS a class-c slot consumes on average over the interval (size / dt)
  cap[v]       MIPS available for new slots: theta*MIPS_v - forecast background - backlog
  base_util[v] forecast utilisation before new slots (background + backlog)/MIPS_v
  gamma[v]     scale of the convex congestion surcharge on planned utilisation

The plan is an integer matrix n[c, v] (slots of class c on VM v).  The exact
programme is Eq. (1)-(4) of the manuscript aggregated over interchangeable
requests in a class, plus a convex piecewise-linear surcharge g_v >= f(rho_v)
that keeps queueing pressure visible to a linear model.  Slots the programme
cannot place feasibly are left to the standard admission path at a penalty M.

The comparators optimise exactly the same objective.  PSO and the GA use a
proportional real-coded encoding (a non-negative share per class and VM,
decoded by largest-remainder rounding to integer slots); ACO samples slot
counts from pheromone-weighted class distributions.  Capacity violations are
penalised because the comparators have no constraint handling of their own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# convex piecewise-linear surcharge f(rho) ~ rho/(1-rho) - 1 above rho = 0.5
PWL_SLOPES = np.array([0.0, 5.0, 8.33, 16.7, 50.0, 100.0])
PWL_INTERCEPTS = np.array([0.0, -2.5, -4.498, -10.36, -37.0, -82.0])
OVERFLOW_PENALTY = 10.0          # M, per slot left to the standard path (exact programme)
CAPACITY_PENALTY = 50.0          # per unit of utilisation above capacity (comparators)


@dataclass
class Instance:
    coef: np.ndarray        # (C, V)
    allowed: np.ndarray     # (C, V) bool
    demand: np.ndarray      # (C,) int
    load: np.ndarray        # (C,) MIPS per slot
    cap: np.ndarray         # (V,) MIPS
    mips: np.ndarray        # (V,)
    base_util: np.ndarray   # (V,)
    gamma: np.ndarray       # (V,)

    @property
    def C(self):
        return self.coef.shape[0]

    @property
    def V(self):
        return self.coef.shape[1]


@dataclass
class PlanResult:
    n: np.ndarray                    # (C, V) int
    overflow: np.ndarray             # (C,) int slots left to the standard path
    objective: float
    runtime: float
    status: str = ""
    mip_gap: float = float("nan")
    evaluations: int = 0
    extra: dict = field(default_factory=dict)


def pwl(rho: np.ndarray) -> np.ndarray:
    """Convex surcharge f(rho), vectorised over any shape."""
    return np.max(PWL_SLOPES * rho[..., None] + PWL_INTERCEPTS, axis=-1)


def objective(inst: Instance, n: np.ndarray, overflow=None, capacity_penalty=0.0) -> np.ndarray:
    """Objective for plans n of shape (..., C, V).  Returns shape (...)."""
    lin = np.sum(inst.coef * n, axis=(-2, -1))
    load_v = np.sum(inst.load[:, None] * n, axis=-2)                      # (..., V)
    rho = inst.base_util + load_v / inst.mips
    cong = np.sum(inst.gamma * pwl(rho), axis=-1)
    val = lin + cong
    if overflow is not None:
        val = val + OVERFLOW_PENALTY * np.sum(overflow, axis=-1)
    if capacity_penalty:
        viol = np.clip(load_v - inst.cap, 0, None) / inst.mips
        val = val + capacity_penalty * np.sum(viol, axis=-1)
    return val


# ------------------------------------------------------------------ exact MILP
def solve_milp(inst: Instance, time_limit: float = 5.0, threads: int = 1, mip_rel_gap: float = 1e-3) -> PlanResult:
    import highspy
    from scipy import sparse

    C, V = inst.C, inst.V
    pairs = np.argwhere(inst.allowed)                     # (P, 2) of (c, v)
    P = len(pairs)
    K = len(PWL_SLOPES)
    # columns: n (P integer) | overflow o (C) | g (V)
    ncol = P + C + V
    cost = np.concatenate([inst.coef[pairs[:, 0], pairs[:, 1]],
                           np.full(C, OVERFLOW_PENALTY),
                           inst.gamma.astype(float)])
    lower = np.zeros(ncol)
    upper = np.concatenate([inst.demand[pairs[:, 0]].astype(float), inst.demand.astype(float),
                            np.full(V, 1e9)])
    rows_i, cols_i, vals = [], [], []
    row_lo, row_hi = [], []
    r = 0
    # demand equalities: sum_v n[c,v] + o_c = demand_c
    for c in range(C):
        sel = np.where(pairs[:, 0] == c)[0]
        rows_i += [r] * (len(sel) + 1); cols_i += list(sel) + [P + c]; vals += [1.0] * (len(sel) + 1)
        row_lo.append(float(inst.demand[c])); row_hi.append(float(inst.demand[c])); r += 1
    # capacity: sum_c load_c n[c,v] <= cap_v
    for v in range(V):
        sel = np.where(pairs[:, 1] == v)[0]
        if len(sel) == 0:
            continue
        rows_i += [r] * len(sel); cols_i += list(sel); vals += list(inst.load[pairs[sel, 0]])
        row_lo.append(-1e30); row_hi.append(float(max(inst.cap[v], 0.0))); r += 1
    # congestion: g_v - slope_k * sum_c (load_c/mips_v) n[c,v] >= slope_k*base_v + intercept_k
    for v in range(V):
        sel = np.where(pairs[:, 1] == v)[0]
        for k in range(1, K):
            s = PWL_SLOPES[k]
            rows_i += [r] * (len(sel) + 1)
            cols_i += list(sel) + [P + C + v]
            vals += list(-s * inst.load[pairs[sel, 0]] / inst.mips[v]) + [1.0]
            row_lo.append(float(s * inst.base_util[v] + PWL_INTERCEPTS[k])); row_hi.append(1e30); r += 1
    A = sparse.csc_matrix((vals, (rows_i, cols_i)), shape=(r, ncol))

    lp = highspy.HighsLp()
    lp.num_col_ = ncol
    lp.num_row_ = r
    lp.col_cost_ = cost
    lp.col_lower_ = lower
    lp.col_upper_ = upper
    lp.row_lower_ = np.array(row_lo)
    lp.row_upper_ = np.array(row_hi)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data
    integ = [highspy.HighsVarType.kInteger] * P + [highspy.HighsVarType.kContinuous] * (C + V)
    lp.integrality_ = integ
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
    h.setOptionValue("random_seed", 0)
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    runtime = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = np.array(h.getSolution().col_value)
    n = np.zeros((C, V), dtype=np.int64)
    if sol.size == ncol and np.all(np.isfinite(sol[:P])):
        n[pairs[:, 0], pairs[:, 1]] = np.rint(sol[:P]).astype(np.int64)
    else:                                                   # no incumbent at all: standard path
        n[:] = 0
    n = np.clip(n, 0, None)
    # repair rounding so that no class exceeds its demand
    for c in range(C):
        excess = n[c].sum() - inst.demand[c]
        while excess > 0:
            v = np.argmax(n[c]); take = min(excess, n[c, v]); n[c, v] -= take; excess -= take
    overflow = inst.demand - n.sum(axis=1)
    gap = float(info.mip_gap) if np.isfinite(info.mip_gap) else float("nan")
    return PlanResult(n=n, overflow=overflow, objective=float(objective(inst, n, overflow)),
                      runtime=runtime, status=status, mip_gap=gap,
                      extra=dict(n_int=P, n_rows=r, time_limit_hit=(status == "Time limit reached"),
                                 solver_objective=float(info.objective_function_value),
                                 dual_bound=float(info.mip_dual_bound)))


# ------------------------------------------------------------------ shared decoding
def decode_shares(shares: np.ndarray, inst: Instance) -> np.ndarray:
    """shares (..., C, V) >= 0 -> integer plans (..., C, V) by largest-remainder rounding."""
    s = np.where(inst.allowed, np.clip(shares, 0, None), 0.0)
    tot = s.sum(axis=-1, keepdims=True)
    # classes with an all-zero share vector: fall back to the cheapest allowed VM
    fallback = np.where(inst.allowed, -inst.coef, -np.inf).argmax(axis=-1)            # (C,)
    empty = (tot[..., 0] <= 0)
    if empty.any():
        onehot = np.zeros(inst.coef.shape); onehot[np.arange(inst.C), fallback] = 1.0
        s = np.where(empty[..., None], onehot, s)
        tot = s.sum(axis=-1, keepdims=True)
    x = inst.demand[:, None] * s / tot
    fl = np.floor(x)
    rem = (inst.demand - fl.sum(axis=-1)).astype(np.int64)                 # (..., C)
    frac = x - fl
    order = np.argsort(-frac, axis=-1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(inst.V), order.shape), axis=-1)
    add = (ranks < rem[..., None]).astype(np.int64)
    return fl.astype(np.int64) + add


def _fitness(inst: Instance, n: np.ndarray) -> np.ndarray:
    return objective(inst, n, capacity_penalty=CAPACITY_PENALTY)


# ------------------------------------------------------------------ PSO
def solve_pso(inst: Instance, rng: np.random.Generator, evaluations: int = 20000,
              swarm: int = 50, w: float = 0.7, c1: float = 1.5, c2: float = 1.5, vmax: float = 0.2) -> PlanResult:
    t0 = time.perf_counter()
    C, V = inst.C, inst.V
    iters = max(1, evaluations // swarm)
    x = rng.random((swarm, C, V))
    vel = np.zeros_like(x)
    n = decode_shares(x, inst)
    f = _fitness(inst, n)
    pbest_x, pbest_f = x.copy(), f.copy()
    gi = np.argmin(f); gbest_x, gbest_f, gbest_n = x[gi].copy(), f[gi], n[gi].copy()
    for _ in range(iters - 1):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        vel = w * vel + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        vel = np.clip(vel, -vmax, vmax)
        x = np.clip(x + vel, 0.0, 1.0)
        n = decode_shares(x, inst)
        f = _fitness(inst, n)
        better = f < pbest_f
        pbest_x[better], pbest_f[better] = x[better], f[better]
        gi = np.argmin(pbest_f)
        if pbest_f[gi] < gbest_f:
            gbest_f, gbest_x, gbest_n = pbest_f[gi], pbest_x[gi].copy(), decode_shares(pbest_x[gi], inst)
    return PlanResult(n=gbest_n, overflow=np.zeros(C, dtype=np.int64), objective=float(gbest_f),
                      runtime=time.perf_counter() - t0, status="budget", evaluations=iters * swarm)


# ------------------------------------------------------------------ GA
def solve_ga(inst: Instance, rng: np.random.Generator, evaluations: int = 20000,
             pop: int = 50, tournament: int = 3, pc: float = 0.9, pm: float = 0.01,
             sigma: float = 0.1, blx_alpha: float = 0.5, elites: int = 2) -> PlanResult:
    t0 = time.perf_counter()
    C, V = inst.C, inst.V
    gens = max(1, evaluations // pop)
    x = rng.random((pop, C, V))
    f = _fitness(inst, decode_shares(x, inst))
    for _ in range(gens - 1):
        order = np.argsort(f)
        elite = x[order[:elites]].copy()
        # tournament selection
        cand = rng.integers(0, pop, size=(pop, tournament))
        parents = x[cand[np.arange(pop), np.argmin(f[cand], axis=1)]]
        # BLX-alpha crossover between consecutive parents
        p1, p2 = parents, parents[rng.permutation(pop)]
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        span = hi - lo
        child = rng.uniform(lo - blx_alpha * span, hi + blx_alpha * span)
        do_cx = rng.random((pop, 1, 1)) < pc
        child = np.where(do_cx, child, p1)
        # gaussian mutation
        mut = rng.random(child.shape) < pm
        child = child + mut * rng.normal(0, sigma, child.shape)
        child = np.clip(child, 0.0, 1.0)
        child[:elites] = elite
        x = child
        f = _fitness(inst, decode_shares(x, inst))
    gi = np.argmin(f)
    return PlanResult(n=decode_shares(x[gi], inst), overflow=np.zeros(C, dtype=np.int64),
                      objective=float(f[gi]), runtime=time.perf_counter() - t0, status="budget",
                      evaluations=gens * pop)


# ------------------------------------------------------------------ ACO
def solve_aco(inst: Instance, rng: np.random.Generator, evaluations: int = 20000,
              ants: int = 30, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
              q: float = 1.0, tau_min: float = 0.01, tau_max: float = 10.0) -> PlanResult:
    t0 = time.perf_counter()
    C, V = inst.C, inst.V
    iters = max(1, evaluations // ants)
    eta = np.where(inst.allowed, 1.0 / (inst.coef + 1e-3), 0.0)
    tau = np.where(inst.allowed, 1.0, 0.0)
    best_f, best_n = np.inf, None
    for _ in range(iters):
        p = (tau ** alpha) * (eta ** beta)
        p = np.where(inst.allowed, p, 0.0)
        tot = p.sum(axis=1, keepdims=True)
        p = np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), inst.allowed / inst.allowed.sum(axis=1, keepdims=True))
        n = rng.multinomial(inst.demand, np.broadcast_to(p, (ants, C, V)))              # (ants, C, V)
        f = _fitness(inst, n)
        gi = np.argmin(f)
        if f[gi] < best_f:
            best_f, best_n = f[gi], n[gi].copy()
        # evaporation and deposit from the iteration-best and global-best ants
        tau = (1 - rho) * tau
        share_it = n[gi] / np.maximum(inst.demand[:, None], 1)
        share_gb = best_n / np.maximum(inst.demand[:, None], 1)
        tau = tau + q * (share_it / (1 + f[gi]) + share_gb / (1 + best_f))
        tau = np.where(inst.allowed, np.clip(tau, tau_min, tau_max), 0.0)
    return PlanResult(n=best_n, overflow=np.zeros(C, dtype=np.int64), objective=float(best_f),
                      runtime=time.perf_counter() - t0, status="budget", evaluations=iters * ants)


SOLVERS = {"milp": solve_milp, "pso": solve_pso, "ga": solve_ga, "aco": solve_aco}
