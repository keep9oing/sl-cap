"""Compiled (numba) kernels for the fixed-order optimal recharge insertion and for the
insertion scans of the ALNS baseline.

`route_time_nb` is a line-by-line port of `common._solve_recharge_py` (all-direct fast path) and
`common._solve_recharge_dp` (stage-wise Pareto-label RCSP over the station graph) for the
`reconstruct=False` case, i.e. it returns the completion time only.  The Python versions stay the
reference implementation (test oracle, and the `reconstruct=True` path that builds the concrete
route for simulator replay).  Arithmetic mirrors the Python code expression by expression
(same accumulation order, same `math.hypot` vs `np.linalg.norm` distances, same 1e-9 epsilons)
so that both agree to floating-point noise; `tests/test_baselines.py` checks this.

Data layout
-----------
`pack_instance(inst)` -> `pk`, a tuple of scalars and C-contiguous float64 arrays cached on the
`Instance` as `inst._pk`:
    T, S, R, vel, mt, ct        problem sizes, speed, energy capacity (time units), recharge duration
    e0[R]                       initial energy per robot
    near_t[T]                   nearest-station travel time per task (reach-with-return threshold)
    d_rt[R,T], d_tt[T,T], rs[R,S], ts_h[T,S]   travel times via math.hypot (as `_hyp`)
    ts[T,S]                     task->station travel times via np.linalg.norm (as `_ts_time`)
    sp_cost[S,S]                charged-at-s -> charged-at-s' cost (Floyd-Warshall, `_station_paths`)
    dmax                        max pairwise distance over all locations (noise / Shaw scale)
`make_workspace(pk)` -> `ws`, scratch arrays allocated once per solve (frontier buffers sized by the
exact bound |frontier at stage k| <= 1 + k*S).

Routes are int64 arrays padded with -1; `L` is the number of tasks in the route.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

INF = np.inf
EPS = 1e-9          # == common._RC_EPS
_CACHE = True


# ------------------------------------------------------------------ packing
def pack_arrays(task_pos, station_pos, robot_pos, energy0, vel, max_time, charge_time, sp_cost):
    """Build the packed tuple from raw arrays (used by pack_instance and by the JIT warm-up)."""
    tp = np.ascontiguousarray(task_pos, dtype=np.float64)
    sp = np.ascontiguousarray(station_pos, dtype=np.float64)
    rp = np.ascontiguousarray(robot_pos, dtype=np.float64)
    T, S, R = tp.shape[0], sp.shape[0], rp.shape[0]
    v = float(vel)

    def hyp_mat(a, b):
        out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                out[i, j] = math.hypot(float(b[j, 0] - a[i, 0]), float(b[j, 1] - a[i, 1])) / v
        return out

    d_rt = hyp_mat(rp, tp)
    d_tt = hyp_mat(tp, tp)
    rs = hyp_mat(rp, sp)
    ts_h = hyp_mat(tp, sp)
    # norm-based, exactly as common._ts_time / Instance.task_near_dist
    ts = np.ascontiguousarray(np.linalg.norm(tp[:, None, :] - sp[None, :, :], axis=-1) / v)
    d_ts = np.linalg.norm(tp[:, None, :] - sp[None, :, :], axis=-1)
    near_t = np.ascontiguousarray(d_ts.min(axis=1) / v) if T > 0 else np.zeros(0)
    allp = np.concatenate([tp, sp, rp], axis=0)
    dmax = float(np.linalg.norm(allp[:, None, :] - allp[None, :, :], axis=-1).max()) if allp.shape[0] else 1.0
    if dmax <= 0.0:
        dmax = 1.0
    e0 = np.ascontiguousarray(energy0, dtype=np.float64)
    return (int(T), int(S), int(R), v, float(max_time), float(charge_time), e0, near_t,
            d_rt, d_tt, rs, ts, ts_h, np.ascontiguousarray(sp_cost, dtype=np.float64), dmax)


def pack_instance(inst):
    """Packed arrays for a `common.Instance` (cached as inst._pk)."""
    pk = getattr(inst, "_pk", None)
    if pk is None:
        from .common import _station_paths
        _, sp_cost, _ = _station_paths(inst)
        pk = pack_arrays(inst.task_pos, inst.station_pos, inst.robot_pos, inst.energy0,
                         inst.vel, inst.max_time, inst.charge_time, sp_cost)
        inst._pk = pk
    return pk


def make_workspace(pk):
    """Scratch arrays for one solve (sized by T, S, R)."""
    T, S, R = pk[0], pk[1], pk[2]
    maxl = 1 + S * T                      # exact frontier bound after Pareto pruning
    cap = maxl * (S + 1)                  # candidates before pruning
    stages = T + 2
    ws = (
        np.zeros((stages, maxl)),                     # 0  lab_t   stage frontiers (forward_nb)
        np.zeros((stages, maxl)),                     # 1  lab_e
        np.zeros((stages, maxl), dtype=np.int64),     # 2  lab_par
        np.zeros(stages, dtype=np.int64),             # 3  lab_n
        np.zeros((2, maxl)),                          # 4  sc_t    ping-pong scratch frontiers
        np.zeros((2, maxl)),                          # 5  sc_e
        np.zeros((2, maxl), dtype=np.int64),          # 6  sc_par
        np.zeros(cap),                                # 7  cand_t
        np.zeros(cap),                                # 8  cand_e
        np.zeros(cap, dtype=np.int64),                # 9  cand_par
        np.zeros(cap, dtype=np.int64),                # 10 cand_idx (sorting scratch)
        np.full(T + 1, -1, dtype=np.int64),           # 11 tmp_route
        np.zeros(T + 2),                              # 12 pre_t   direct-prefix travel time
        np.zeros(T + 2),                              # 13 pre_e   direct-prefix energy on arrival
        np.zeros(T + 2),                              # 14 suf_slack
        np.zeros(4),                                  # 15 meta: [direct_ok, direct_total, L, unused]
        np.zeros((T, max(R, 1))),                     # 16 tab_time  (repair table)
        np.zeros((T, max(R, 1))),                     # 17 tab_delta
        np.zeros((T, max(R, 1)), dtype=np.int64),     # 18 tab_pos
        np.zeros(T),                                  # 19 f1: per-task float scratch (visit / t_without)
        np.zeros(T),                                  # 20 f2: per-task float scratch (gain / relatedness)
        np.zeros(T, dtype=np.int64),                  # 21 i1: per-task int scratch
        np.zeros(T, dtype=np.int64),                  # 22 i2: per-task int scratch
        np.zeros(max(R, 1), dtype=np.int64),          # 23 r1: per-robot int scratch
        np.zeros(max(R, 1)),                          # 24 rf: per-robot float scratch
    )
    return ws


# ------------------------------------------------------------------ DP pieces
@njit(cache=_CACHE)
def _pareto(cand_t, cand_e, cand_par, cand_idx, nc, out_t, out_e, out_par):
    """common._pareto: sort by (time asc, energy desc), keep strictly increasing energy (eps)."""
    if nc <= 32:
        # insertion sort on (t, -e)
        for i in range(nc):
            cand_idx[i] = i
        for i in range(1, nc):
            v = cand_idx[i]
            tv = cand_t[v]
            ev = cand_e[v]
            k = i - 1
            while k >= 0:
                u = cand_idx[k]
                tu = cand_t[u]
                if tu > tv or (tu == tv and cand_e[u] < ev):
                    cand_idx[k + 1] = u
                    k -= 1
                else:
                    break
            cand_idx[k + 1] = v
        order = cand_idx[:nc]
    else:
        idx1 = np.argsort(-cand_e[:nc], kind="mergesort")
        t1 = cand_t[:nc][idx1]
        idx2 = np.argsort(t1, kind="mergesort")
        order = idx1[idx2]
    n_out = 0
    best_e = -INF
    for k in range(nc):
        i = order[k]
        if cand_e[i] > best_e + EPS:
            out_t[n_out] = cand_t[i]
            out_e[n_out] = cand_e[i]
            out_par[n_out] = cand_par[i]
            n_out += 1
            best_e = cand_e[i]
    return n_out


@njit(cache=_CACHE)
def _extend(S, mt, ct, near_t, d_rt, d_tt, rs, ts, sp_cost,
            prev, rid, b, in_t, in_e, n_in, out_t, out_e, out_par,
            cand_t, cand_e, cand_par, cand_idx):
    """One DP stage: serve task b from the frontier `in_*` (location = task `prev`, or the robot
    start if prev < 0).  Mirrors the loop body of common._solve_recharge_dp.  Returns the size of
    the pruned output frontier (0 = infeasible)."""
    bnear = near_t[b]
    nc = 0
    for li in range(n_in):
        t_a = in_t[li]
        e_a = in_e[li]
        # ---- direct (no charge) ----
        if prev < 0:
            dd = d_rt[rid, b]
        else:
            dd = d_tt[prev, b]
        e_arr = e_a - dd
        if e_arr >= bnear - EPS:
            cand_t[nc] = t_a + dd
            cand_e[nc] = e_arr
            cand_par[nc] = li
            nc += 1
        # ---- via a charger chain ending at s_out ----
        for s_out in range(S):
            d_out = ts[b, s_out]
            eb = mt - d_out
            if eb < bnear - EPS:
                continue
            best_c = INF
            best_sin = -1
            for s_in in range(S):
                if prev < 0:
                    d_in = rs[rid, s_in]
                else:
                    d_in = ts[prev, s_in]
                if d_in <= e_a + EPS and sp_cost[s_in, s_out] < INF:
                    c = d_in + ct + sp_cost[s_in, s_out]
                    if c < best_c - EPS:
                        best_c = c
                        best_sin = s_in
            if best_sin >= 0:
                cand_t[nc] = t_a + best_c + d_out
                cand_e[nc] = eb
                cand_par[nc] = li
                nc += 1
    if nc == 0:
        return 0
    return _pareto(cand_t, cand_e, cand_par, cand_idx, nc, out_t, out_e, out_par)


@njit(cache=_CACHE)
def _park(S, rs, ts, prev, rid, in_t, in_e, n_in):
    """Park at the nearest reachable station from task `prev` (robot start if prev < 0).
    Returns (t_final, label index) or (inf, -1)."""
    best_t = INF
    best_li = -1
    for li in range(n_in):
        t_a = in_t[li]
        e_a = in_e[li]
        for s in range(S):
            if prev < 0:
                ds = rs[rid, s]
            else:
                ds = ts[prev, s]
            if ds <= e_a + EPS:
                tf = t_a + ds
                if tf < best_t - EPS:
                    best_t = tf
                    best_li = li
    return best_t, best_li


@njit(cache=_CACHE)
def _fast_direct(S, e0, near_t, d_rt, d_tt, rs, ts_h, rid, route, L):
    """common._solve_recharge_py fast path: all-direct route, park at the nearest reachable station.
    Returns (time, feasible)."""
    e = e0[rid]
    t = 0.0
    prev = -1
    for k in range(L):
        j = route[k]
        if prev < 0:
            dd = d_rt[rid, j]
        else:
            dd = d_tt[prev, j]
        e -= dd
        t += dd
        if e < near_t[j] - EPS:
            return INF, False
        prev = j
    best_d = INF
    found = False
    for s in range(S):
        if prev < 0:
            ds = rs[rid, s]
        else:
            ds = ts_h[prev, s]
        if ds <= e + EPS and ds < best_d:
            best_d = ds
            found = True
    if not found:
        return INF, False
    return t + best_d, True


@njit(cache=_CACHE)
def route_time_nb(pk, ws, rid, route, L):
    """Completion time of robot `rid` serving route[:L] in that order with optimal recharging
    (inf if no feasible recharge plan exists).  == common.route_time without the memo."""
    T, S, R, vel, mt, ct, e0, near_t, d_rt, d_tt, rs, ts, ts_h, sp_cost, dmax = pk
    t, ok = _fast_direct(S, e0, near_t, d_rt, d_tt, rs, ts_h, rid, route, L)
    if ok:
        return t
    sc_t = ws[4]
    sc_e = ws[5]
    sc_par = ws[6]
    cand_t = ws[7]
    cand_e = ws[8]
    cand_par = ws[9]
    cand_idx = ws[10]
    cur = 0
    n = 1
    sc_t[0, 0] = 0.0
    sc_e[0, 0] = e0[rid]
    prev = -1
    for k in range(L):
        b = route[k]
        n = _extend(S, mt, ct, near_t, d_rt, d_tt, rs, ts, sp_cost, prev, rid, b,
                    sc_t[cur], sc_e[cur], n, sc_t[1 - cur], sc_e[1 - cur], sc_par[1 - cur],
                    cand_t, cand_e, cand_par, cand_idx)
        if n == 0:
            return INF
        cur = 1 - cur
        prev = b
    t, li = _park(S, rs, ts, prev, rid, sc_t[cur], sc_e[cur], n)
    return t


@njit(cache=_CACHE)
def forward_nb(pk, ws, rid, route, L):
    """Prepare route `rid` for insertion scans: DP frontiers after k served tasks (k = 0..L) in
    ws.lab_*, plus the all-direct prefix arrays (pre_t, pre_e, suf_slack) and meta =
    [direct_ok, direct_total (time incl. park), L, 0].  Returns the canonical route time."""
    T, S, R, vel, mt, ct, e0, near_t, d_rt, d_tt, rs, ts, ts_h, sp_cost, dmax = pk
    lab_t = ws[0]
    lab_e = ws[1]
    lab_par = ws[2]
    lab_n = ws[3]
    cand_t = ws[7]
    cand_e = ws[8]
    cand_par = ws[9]
    cand_idx = ws[10]
    pre_t = ws[12]
    pre_e = ws[13]
    suf_slack = ws[14]
    meta = ws[15]
    # ---- direct prefix ----
    e = e0[rid]
    t = 0.0
    pre_t[0] = 0.0
    pre_e[0] = e
    prev = -1
    direct_ok = True
    for k in range(L):
        j = route[k]
        if prev < 0:
            dd = d_rt[rid, j]
        else:
            dd = d_tt[prev, j]
        e -= dd
        t += dd
        pre_t[k + 1] = t
        pre_e[k + 1] = e
        if e < near_t[j] - EPS:
            direct_ok = False
            break
        prev = j
    direct_total = INF
    if direct_ok:
        best_d = INF
        found = False
        for s in range(S):
            if prev < 0:
                ds = rs[rid, s]
            else:
                ds = ts_h[prev, s]
            if ds <= e + EPS and ds < best_d:
                best_d = ds
                found = True
        if found:
            direct_total = t + best_d
        else:
            direct_ok = False
    suf_slack[L] = INF
    if direct_ok:
        m = INF
        for p in range(L - 1, -1, -1):
            v = pre_e[p + 1] - near_t[route[p]]
            if v < m:
                m = v
            suf_slack[p] = m
    meta[0] = 1.0 if direct_ok else 0.0
    meta[1] = direct_total
    meta[2] = L
    # ---- DP frontiers for every stage ----
    lab_n[0] = 1
    lab_t[0, 0] = 0.0
    lab_e[0, 0] = e0[rid]
    lab_par[0, 0] = -1
    prev = -1
    n = 1
    for k in range(L):
        b = route[k]
        n = _extend(S, mt, ct, near_t, d_rt, d_tt, rs, ts, sp_cost, prev, rid, b,
                    lab_t[k], lab_e[k], n, lab_t[k + 1], lab_e[k + 1], lab_par[k + 1],
                    cand_t, cand_e, cand_par, cand_idx)
        lab_n[k + 1] = n
        if n == 0:
            for kk in range(k + 2, L + 1):
                lab_n[kk] = 0
            break
        prev = b
    if direct_ok:
        return direct_total
    if n == 0:
        return INF
    tf, li = _park(S, rs, ts, prev, rid, lab_t[L], lab_e[L], n)
    return tf


@njit(cache=_CACHE)
def route_visit_nb(pk, ws, rid, route, L, visit):
    """Arrival time at every task of the route under the optimal recharge plan (DP with
    backtracking); written into visit[task].  Returns the route time (inf if infeasible)."""
    T, S, R, vel, mt, ct, e0, near_t, d_rt, d_tt, rs, ts, ts_h, sp_cost, dmax = pk
    tf = forward_nb(pk, ws, rid, route, L)
    if L == 0:
        return tf
    lab_t = ws[0]
    lab_e = ws[1]
    lab_par = ws[2]
    lab_n = ws[3]
    meta = ws[15]
    if meta[0] > 0.5:
        pre_t = ws[12]
        for k in range(L):
            visit[route[k]] = pre_t[k + 1]
        return tf
    n = lab_n[L]
    if n == 0:
        for k in range(L):
            visit[route[k]] = INF
        return INF
    t_end, li = _park(S, rs, ts, route[L - 1], rid, lab_t[L], lab_e[L], n)
    if li < 0:
        for k in range(L):
            visit[route[k]] = INF
        return INF
    for k in range(L, 0, -1):
        visit[route[k - 1]] = lab_t[k, li]
        li = lab_par[k, li]
    return t_end


@njit(cache=_CACHE)
def insertion_scan_nb(pk, ws, rid, route, L, t_r, j, noise_on, max_noise, rng):
    """Best position of task j in the route prepared by forward_nb (same rid/route/L).
    Returns (new_route_time, delta_noisy, pos); pos = -1 if j fits nowhere.
    delta_noisy = max(0, (new_time - t_r) + U(-max_noise, max_noise)) when noise_on, else the
    plain increase; the position with the smallest (noisy) increase is returned together with its
    plain new route time."""
    T, S, R, vel, mt, ct, e0, near_t, d_rt, d_tt, rs, ts, ts_h, sp_cost, dmax = pk
    lab_t = ws[0]
    lab_e = ws[1]
    lab_n = ws[3]
    sc_t = ws[4]
    sc_e = ws[5]
    sc_par = ws[6]
    cand_t = ws[7]
    cand_e = ws[8]
    cand_par = ws[9]
    cand_idx = ws[10]
    pre_t = ws[12]
    pre_e = ws[13]
    suf_slack = ws[14]
    meta = ws[15]
    direct_ok = meta[0] > 0.5
    direct_total = meta[1]
    best_dn = INF
    best_pos = -1
    best_time = INF
    for p in range(L + 1):
        if p > 0:
            prev = route[p - 1]
        else:
            prev = -1
        new_t = INF
        done = False
        if direct_ok:
            if prev < 0:
                d1 = d_rt[rid, j]
            else:
                d1 = d_tt[prev, j]
            e_j = pre_e[p] - d1
            if e_j >= near_t[j] - EPS:
                if p < L:
                    nxt = route[p]
                    if prev < 0:
                        d0 = d_rt[rid, nxt]
                    else:
                        d0 = d_tt[prev, nxt]
                    delta = d1 + d_tt[j, nxt] - d0
                    if delta <= suf_slack[p] + EPS:
                        new_t = direct_total + delta
                        done = True
                else:
                    # append: park from j at its nearest station
                    best_d = INF
                    for s in range(S):
                        ds = ts_h[j, s]
                        if ds <= e_j + EPS and ds < best_d:
                            best_d = ds
                    if best_d < INF:
                        # travel to the old last task is pre_t[L]; new time = pre_t[L] + d1 + park
                        new_t = pre_t[L] + d1 + best_d
                        done = True
        if not done:
            n = lab_n[p]
            if n == 0:
                continue
            n = _extend(S, mt, ct, near_t, d_rt, d_tt, rs, ts, sp_cost, prev, rid, j,
                        lab_t[p], lab_e[p], n, sc_t[0], sc_e[0], sc_par[0],
                        cand_t, cand_e, cand_par, cand_idx)
            if n == 0:
                continue
            cur = 0
            prev2 = j
            feas = True
            for k in range(p, L):
                b = route[k]
                n = _extend(S, mt, ct, near_t, d_rt, d_tt, rs, ts, sp_cost, prev2, rid, b,
                            sc_t[cur], sc_e[cur], n, sc_t[1 - cur], sc_e[1 - cur], sc_par[1 - cur],
                            cand_t, cand_e, cand_par, cand_idx)
                if n == 0:
                    feas = False
                    break
                cur = 1 - cur
                prev2 = b
            if not feas:
                continue
            new_t, li = _park(S, rs, ts, prev2, rid, sc_t[cur], sc_e[cur], n)
            if li < 0:
                continue
        delta = new_t - t_r
        if delta < 0.0:
            delta = 0.0
        if noise_on:
            dn = delta + (2.0 * rng.random() - 1.0) * max_noise
            if dn < 0.0:
                dn = 0.0
        else:
            dn = delta
        if dn < best_dn:
            best_dn = dn
            best_pos = p
            best_time = new_t
    return best_time, best_dn, best_pos


# ------------------------------------------------------------------ warm-up
def warmup():
    """Compile every kernel on a tiny synthetic instance (kept out of any timer)."""
    from .common import _station_paths  # noqa: F401  (import check only)
    rng = np.random.default_rng(0)
    tp = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9], [0.2, 0.6]])
    sp = np.array([[0.5, 0.5], [0.15, 0.15]])
    rp = np.array([[0.5, 0.45], [0.6, 0.6]])
    e0 = np.array([0.6, 0.4])
    vel, mt, ct = 1.0, 0.8, 0.2
    S = sp.shape[0]
    Dt = np.linalg.norm(sp[:, None, :] - sp[None, :, :], axis=-1) / vel
    cost = np.full((S, S), INF)
    for s in range(S):
        cost[s, s] = 0.0
        for s2 in range(S):
            if s2 != s and Dt[s, s2] <= mt + EPS:
                cost[s, s2] = Dt[s, s2] + ct
    pk = pack_arrays(tp, sp, rp, e0, vel, mt, ct, cost)
    ws = make_workspace(pk)
    route = np.array([0, 1, 2, 3, -1], dtype=np.int64)
    visit = np.zeros(4)
    route_time_nb(pk, ws, 0, route, 4)
    forward_nb(pk, ws, 1, route, 3)
    insertion_scan_nb(pk, ws, 1, route, 3, 1.0, 3, False, 0.0, rng)
    insertion_scan_nb(pk, ws, 1, route, 3, 1.0, 3, True, 0.01, rng)
    route_visit_nb(pk, ws, 0, route, 4, visit)
