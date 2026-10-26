"""Shared parts of the plan-based baselines (SGA, B&B, ALNS): the instance view, the recharging
dynamic program of Appendix B.2, and the replay that scores a plan.

A plan assigns each robot an order of tasks. For a fixed order, the dynamic program inserts the
recharging stops that minimize the robot's completion time: its stages are the tasks, its labels the
Pareto-optimal (elapsed time, energy on arrival) pairs, and between two tasks the robot either travels
directly or recharges along the cheapest chain of stations (Floyd-Warshall over the pairs of stations
that one full charge connects). A move is allowed only if the energy covers the travel and, on arrival
at a task, the travel from that task to its nearest station -- the same rule as the policy's mask.
`route_time` (the call every search makes) runs the compiled port in rc_kernels.py; the Python version
below builds the concrete route of a final plan, which `replay` then times as the simulator does.
"""
from __future__ import annotations

import math

import numpy as np

from . import rc_kernels as _rck

_RC_EPS = 1e-9


class Instance:
    """Float64 view of an instance (slcap.instances) for the plan-based methods."""

    def __init__(self, inst):
        self.raw = inst
        self.K, self.N, self.M = int(inst["K"]), int(inst["N"]), int(inst["M"])
        self.station_pos = np.asarray(inst["stations"], dtype=np.float64)    # (M, 2)
        self.task_pos = np.asarray(inst["tasks"], dtype=np.float64)          # (N, 2)
        self.robot_pos = np.asarray(inst["robots"], dtype=np.float64)        # (K, 2)
        self.energy0 = np.asarray(inst["energy0"], dtype=np.float64)         # (K,)
        self.vel = float(inst["speed"])
        self.max_time = float(inst["energy"])
        self.charge_time = float(inst["charge_time"])
        d_ts = np.linalg.norm(self.task_pos[:, None, :] - self.station_pos[None, :, :], axis=-1)
        self.task_near_dist = d_ts.min(axis=1)                               # (N,)


def _station_paths(inst: Instance):
    """All-pairs cheapest recharging chain between stations (cached on the instance).

    Returns (Dt, cost, nxt): Dt[s, s'] travel time; cost[s, s'] = travel plus a recharge at every
    station entered after s (inf if unreachable), a direct leg allowed iff one full charge covers it;
    nxt[s, s'] = next station on that chain."""
    cache = getattr(inst, "_sp_cache", None)
    if cache is not None:
        return cache
    M = inst.M
    sp = inst.station_pos
    Dt = np.linalg.norm(sp[:, None, :] - sp[None, :, :], axis=-1) / inst.vel
    ct, mt = inst.charge_time, inst.max_time
    INF = np.inf
    cost = np.full((M, M), INF)
    nxt = -np.ones((M, M), dtype=np.int64)
    for s in range(M):
        cost[s, s] = 0.0
        nxt[s, s] = s
        for s2 in range(M):
            if s2 != s and Dt[s, s2] <= mt + _RC_EPS:
                cost[s, s2] = Dt[s, s2] + ct
                nxt[s, s2] = s2
    for k in range(M):                                       # Floyd-Warshall
        ck = cost[k]
        for i in range(M):
            cik = cost[i, k]
            if cik == INF:
                continue
            row = cost[i]
            nik = nxt[i, k]
            for j in range(M):
                nd = cik + ck[j]
                if nd < row[j] - _RC_EPS:
                    row[j] = nd
                    nxt[i, j] = nik
    cache = (Dt, cost, nxt)
    inst._sp_cache = cache
    return cache


def _ts_time(inst: Instance):
    """(N, M) task-to-station travel times (cached)."""
    cache = getattr(inst, "_ts_cache", None)
    if cache is None:
        cache = np.linalg.norm(inst.task_pos[:, None, :] - inst.station_pos[None, :, :], axis=-1) / inst.vel
        inst._ts_cache = cache
    return cache


def _hyp(p, q):
    return math.hypot(float(q[0] - p[0]), float(q[1] - p[1]))


def _pareto(labels):
    """Keep the (time, energy) skyline: sort by time (energy descending on ties), keep increasing energy."""
    labels.sort(key=lambda L: (L[0], -L[1]))
    keep = []
    best_e = -np.inf
    for L in labels:
        if L[1] > best_e + _RC_EPS:
            keep.append(L)
            best_e = L[1]
    return keep


def _recon_path(s_in, s_out, nxt):
    path = [int(s_in)]
    s = int(s_in)
    while s != s_out:
        s = int(nxt[s, s_out])
        path.append(s)
    return path


def _solve_recharge_dp(inst: Instance, rid: int, task_order, reconstruct: bool):
    """The dynamic program for a fixed order whose all-direct route is infeasible.

    Stage j = task_order[j] served; a label = [time, energy on arrival, min energy, parent, seg],
    seg = the stations entered on the way into this stage. A recharge restores full energy, so the
    state after it depends only on the station and the time: the Pareto labels suffice."""
    v, mt, ct = inst.vel, inst.max_time, inst.charge_time
    M = inst.M
    sp, tp = inst.station_pos, inst.task_pos
    near_t = inst.task_near_dist / v
    eps = _RC_EPS
    _, SPcost, SPnext = _station_paths(inst)
    ts = _ts_time(inst)
    INF = np.inf

    start = inst.robot_pos[rid]
    e0 = float(inst.energy0[rid])

    def loc_station_times(prev_task):
        if prev_task is None:
            return np.array([_hyp(start, sp[s]) for s in range(M)]) / v
        return ts[prev_task]

    stages = [[[0.0, e0, e0, -1, []]]]
    labels = stages[0]
    loc_pos = start
    prev_task = None

    for b in task_order:
        loc_ts = loc_station_times(prev_task)
        bnear = near_t[b]
        new = []
        for li, lab in enumerate(labels):
            t_a, e_a, me_a = lab[0], lab[1], lab[2]
            dd = _hyp(loc_pos, tp[b]) / v                    # direct
            e_arr = e_a - dd
            if e_arr >= bnear - eps:
                new.append([t_a + dd, e_arr, min(me_a, e_arr), li, []])
            for s_out in range(M):                           # via a chain of stations ending at s_out
                d_out = ts[b, s_out]
                eb = mt - d_out
                if eb < bnear - eps:
                    continue
                best_c, best_sin = INF, -1
                for s_in in range(M):
                    d_in = loc_ts[s_in]
                    if d_in <= e_a + eps and SPcost[s_in, s_out] < INF:
                        c = d_in + ct + SPcost[s_in, s_out]
                        if c < best_c - eps:
                            best_c, best_sin = c, s_in
                if best_sin >= 0:
                    seg = _recon_path(best_sin, s_out, SPnext) if reconstruct else []
                    new.append([t_a + best_c + d_out, eb, min(me_a, eb), li, seg])
        if not new:
            return None, INF, False
        new = _pareto(new)
        stages.append(new)
        labels = new
        loc_pos = tp[b]
        prev_task = b

    loc_ts = loc_station_times(prev_task)                    # park at the best reachable station
    best = None
    for li, lab in enumerate(labels):
        t_a, e_a = lab[0], lab[1]
        for s in range(M):
            ds = loc_ts[s]
            if ds <= e_a + eps:
                tf = t_a + ds
                if best is None or tf < best[0] - eps:
                    best = (tf, li, s)
    if best is None:
        return None, INF, False
    t_final, li, s_park = best
    if not reconstruct:
        return None, t_final, True

    order_list = list(task_order)                            # walk the parents back
    route_rev = [("park", int(s_park))]
    cur = li
    for si in range(len(stages) - 1, 0, -1):
        lab = stages[si][cur]
        route_rev.append(("task", order_list[si - 1]))
        for s in reversed(lab[4]):
            route_rev.append(("station", int(s)))
        cur = lab[3]
    return list(reversed(route_rev)), t_final, True


def _solve_recharge_py(inst: Instance, rid: int, task_order, reconstruct: bool):
    """Reference implementation: the all-direct route when it is feasible (any recharge only adds
    time), else the dynamic program. Returns (route or None, time, feasible)."""
    v = inst.vel
    sp, tp = inst.station_pos, inst.task_pos
    near_t = inst.task_near_dist / v
    eps = _RC_EPS
    pos = inst.robot_pos[rid]
    e = float(inst.energy0[rid])
    t = 0.0
    ok = True
    for j in task_order:
        dd = _hyp(pos, tp[j]) / v
        e -= dd
        t += dd
        if e < near_t[j] - eps:
            ok = False
            break
        pos = tp[j]
    if ok:
        best_s, best_d = -1, np.inf
        for s in range(inst.M):
            ds = _hyp(pos, sp[s]) / v
            if ds <= e + eps and ds < best_d:
                best_s, best_d = s, ds
        if best_s >= 0:
            route = [("task", j) for j in task_order] + [("park", best_s)] if reconstruct else None
            return route, t + best_d, True
    return _solve_recharge_dp(inst, rid, task_order, reconstruct)


def _nb_ctx(inst: Instance):
    """(packed arrays, workspace) for the compiled kernels, cached on the instance."""
    ctx = getattr(inst, "_nbctx", None)
    if ctx is None:
        pk = _rck.pack_instance(inst)
        ctx = inst._nbctx = (pk, _rck.make_workspace(pk))
    return ctx


def route_time_uncached(inst: Instance, rid: int, task_order):
    """Minimum completion time of robot rid serving task_order (inf if infeasible)."""
    pk, ws = _nb_ctx(inst)
    tmp = ws[11]
    for L, j in enumerate(task_order):
        tmp[L] = j
    return float(_rck.route_time_nb(pk, ws, rid, tmp, len(task_order)))


def route_time(inst: Instance, rid: int, task_order):
    """route_time_uncached, memoized per instance (the searches re-evaluate the same orders often)."""
    cache = getattr(inst, "_rt_cache", None)
    if cache is None:
        cache = inst._rt_cache = {}
    key = (rid, tuple(task_order))
    val = cache.get(key)
    if val is None:
        val = cache[key] = route_time_uncached(inst, rid, task_order)
    return val


def makespan_of(inst: Instance, task_orders):
    return max(route_time(inst, r, task_orders[r]) for r in range(inst.K))


def build_routes(inst: Instance, task_orders):
    """Per-robot routes [("task", j) | ("station", s) | ("park", s), ...] with the optimal recharging stops."""
    routes = []
    for r in range(inst.K):
        route, _, feasible = _solve_recharge_py(inst, r, list(task_orders[r]), reconstruct=True)
        if not feasible:
            raise ValueError(f"robot {r}: no energy-feasible route for its task order")
        routes.append(route)
    return routes


def replay(inst, routes):
    """Score per-robot routes as the simulator does (float32 positions, travel time = distance / v,
    a recharge adds tau and restores full energy). Returns (makespan, complete, min energy)."""
    sp, tp = inst["stations"], inst["tasks"]
    v, mt, ct = float(inst["speed"]), float(inst["energy"]), float(inst["charge_time"])
    served, parked, min_e = set(), [], math.inf
    for r, route in enumerate(routes):
        pos, e, t = inst["robots"][r], float(inst["energy0"][r]), 0.0
        for kind, j in route:
            target = tp[j] if kind == "task" else sp[j]
            d = float(np.linalg.norm(target - pos)) / v
            e -= d
            t += d
            min_e = min(min_e, e)
            pos = target
            if kind == "task":
                served.add(j)
            elif kind == "station":
                t += ct
                e = mt
            else:
                parked.append(t)
                break
    complete = len(served) == inst["N"] and len(parked) == inst["K"]
    return max(parked), complete, min_e
