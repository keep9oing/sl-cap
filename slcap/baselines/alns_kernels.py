"""Compiled destroy / repair operators of the ALNS baseline (Ropke & Pisinger 2006), operating on
the padded solution layout used by baselines/alns.py:

    routes[R, T+1]  int64, task ids in service order, -1 padded
    lens[R]         int64, number of tasks per robot
    times[R]        float64, canonical route time (rc_kernels.route_time_nb) per robot
    where[T, 2]     int64, (robot, position) of every task; robot = -1 while removed

Search objective (min-max with a tie-breaker, see alns.py):  F = C_max + eps * mean_r times[r].
Insertion cost of task u into route r with route-time increase d:
    c(u, r) = max(times[r] + d, M_{-r}) - C_max + eps * d / R,      M_{-r} = max_{r' != r} times[r']
(monotone in d, so the best position of u in r is the one with the smallest d -- what
rc_kernels.insertion_scan_nb returns).  Worst-removal gain is the same expression with the sign
reversed.  All random draws come from the numpy Generator passed in, shared with the `alns` package.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .rc_kernels import route_time_nb, forward_nb, route_visit_nb, insertion_scan_nb

INF = np.inf
BIG = 1e6                      # cost of an impossible insertion (RP: "cannot be inserted")
_CACHE = True


@njit(cache=_CACHE)
def _remove_at(routes, lens, where, r, pos):
    L = lens[r]
    for k in range(pos, L - 1):
        routes[r, k] = routes[r, k + 1]
        where[routes[r, k], 1] = k
    routes[r, L - 1] = -1
    lens[r] = L - 1


@njit(cache=_CACHE)
def _insert_at(routes, lens, where, r, pos, j):
    L = lens[r]
    for k in range(L, pos, -1):
        routes[r, k] = routes[r, k - 1]
        where[routes[r, k], 1] = k
    routes[r, pos] = j
    where[j, 0] = r
    where[j, 1] = pos
    lens[r] = L + 1


@njit(cache=_CACHE)
def _remove_set(pk, ws, routes, lens, times, where, removed, n_rem):
    """Remove removed[:n_rem] from their routes and refresh the affected route times."""
    R = routes.shape[0]
    affected = ws[23]
    for r in range(R):
        affected[r] = 0
    for i in range(n_rem):
        j = removed[i]
        r = where[j, 0]
        _remove_at(routes, lens, where, r, where[j, 1])
        where[j, 0] = -1
        where[j, 1] = -1
        affected[r] = 1
    for r in range(R):
        if affected[r] == 1:
            times[r] = route_time_nb(pk, ws, r, routes[r], lens[r])


@njit(cache=_CACHE)
def _top2(times, R):
    """(cmax, argmax, second largest)."""
    cmax = -INF
    r1 = -1
    second = -INF
    for r in range(R):
        v = times[r]
        if v > cmax:
            second = cmax
            cmax = v
            r1 = r
        elif v > second:
            second = v
    return cmax, r1, second


@njit(cache=_CACHE)
def _pick_index(y, p, n):
    """RP's randomised choice L[floor(y^p * |L|)]."""
    idx = int(y ** p * n)
    if idx >= n:
        idx = n - 1
    if idx < 0:
        idx = 0
    return idx


# ------------------------------------------------------------------ destroy
@njit(cache=_CACHE)
def destroy_random_nb(pk, ws, routes, lens, times, where, q, rng, removed):
    """Random removal: q tasks chosen uniformly."""
    T = where.shape[0]
    perm = rng.permutation(T)
    for i in range(q):
        removed[i] = perm[i]
    _remove_set(pk, ws, routes, lens, times, where, removed, q)
    return q


@njit(cache=_CACHE)
def _t_without_route(pk, ws, routes, lens, r, t_without):
    """Route time of route r without each of its tasks (one evaluation per task)."""
    tmp = ws[11]
    L = lens[r]
    for pos in range(L):
        j = routes[r, pos]
        m = 0
        for k in range(L):
            if k != pos:
                tmp[m] = routes[r, k]
                m += 1
        t_without[j] = route_time_nb(pk, ws, r, tmp, L - 1)


@njit(cache=_CACHE)
def destroy_worst_nb(pk, ws, routes, lens, times, where, q, p_worst, eps, rng, removed):
    """Worst removal (RP Alg. 3): repeatedly remove a task with a large removal gain,
    gain(i) = F(s) - F(s without i), randomised with parameter p_worst."""
    T = where.shape[0]
    R = routes.shape[0]
    t_without = ws[19]
    gain = ws[20]
    cand = ws[22]
    for r in range(R):
        _t_without_route(pk, ws, routes, lens, r, t_without)
    n_rem = 0
    for it in range(q):
        cmax, r1, second = _top2(times, R)
        n_c = 0
        for j in range(T):
            r = where[j, 0]
            if r < 0:
                continue
            M = second if r == r1 else cmax
            tw = t_without[j]
            new_c = tw if tw > M else M
            gain[j] = (cmax - new_c) + eps * (times[r] - tw) / R
            cand[n_c] = j
            n_c += 1
        if n_c == 0:
            break
        keys = np.empty(n_c)
        for i in range(n_c):
            keys[i] = -gain[cand[i]]
        order = np.argsort(keys, kind="mergesort")
        pick = cand[order[_pick_index(rng.random(), p_worst, n_c)]]
        r = where[pick, 0]
        _remove_at(routes, lens, where, r, where[pick, 1])
        where[pick, 0] = -1
        where[pick, 1] = -1
        times[r] = route_time_nb(pk, ws, r, routes[r], lens[r])
        removed[n_rem] = pick
        n_rem += 1
        _t_without_route(pk, ws, routes, lens, r, t_without)
    return n_rem


@njit(cache=_CACHE)
def destroy_shaw_nb(pk, ws, routes, lens, times, where, q, p_shaw, phi, chi, rng, removed):
    """Shaw (related) removal (RP Alg. 2) with relatedness
    R(i, j) = phi * d_ij / d_max + chi * |visit_i - visit_j| / visit_max."""
    T = where.shape[0]
    R = routes.shape[0]
    d_tt = pk[9]
    vel = pk[3]
    tmax = pk[14] / vel                       # max pairwise travel time
    visit = ws[19]
    rel = ws[20]
    in_d = ws[21]
    cand = ws[22]
    for j in range(T):
        visit[j] = 0.0
        in_d[j] = 0
    for r in range(R):
        if lens[r] > 0:
            route_visit_nb(pk, ws, r, routes[r], lens[r], visit)
    vmax = 0.0
    for j in range(T):
        if visit[j] < INF and visit[j] > vmax:
            vmax = visit[j]
    if vmax <= 0.0:
        vmax = 1.0
    first = rng.integers(0, T)
    removed[0] = first
    in_d[first] = 1
    n_d = 1
    while n_d < q:
        rsel = removed[rng.integers(0, n_d)]
        n_c = 0
        for j in range(T):
            if in_d[j] == 1:
                continue
            dv = visit[rsel] - visit[j]
            if dv < 0.0:
                dv = -dv
            rel[j] = phi * d_tt[rsel, j] / tmax + chi * dv / vmax
            cand[n_c] = j
            n_c += 1
        if n_c == 0:
            break
        keys = np.empty(n_c)
        for i in range(n_c):
            keys[i] = rel[cand[i]]
        order = np.argsort(keys, kind="mergesort")
        pick = cand[order[_pick_index(rng.random(), p_shaw, n_c)]]
        removed[n_d] = pick
        in_d[pick] = 1
        n_d += 1
    _remove_set(pk, ws, routes, lens, times, where, removed, n_d)
    return n_d


# ------------------------------------------------------------------ repair
@njit(cache=_CACHE)
def _scan_column(pk, ws, routes, lens, times, removed, n_rem, placed, r, noise_on, max_noise, rng,
                 tab_time, tab_delta, tab_pos):
    forward_nb(pk, ws, r, routes[r], lens[r])
    for i in range(n_rem):
        if placed[i] == 1:
            continue
        t_new, dn, pos = insertion_scan_nb(pk, ws, r, routes[r], lens[r], times[r], removed[i],
                                           noise_on, max_noise, rng)
        tab_time[i, r] = t_new
        tab_delta[i, r] = dn
        tab_pos[i, r] = pos


@njit(cache=_CACHE)
def repair_nb(pk, ws, routes, lens, times, where, removed, n_rem, k_regret, noise_on, max_noise,
              eps, rng):
    """Regret-k insertion (k = 1 is RP's basic greedy).  Returns the number of tasks that could
    not be inserted anywhere (0 for a feasible repair)."""
    R = routes.shape[0]
    tab_time = ws[16]
    tab_delta = ws[17]
    tab_pos = ws[18]
    placed = ws[21]
    costs = ws[24]
    for i in range(n_rem):
        placed[i] = 0
    for r in range(R):
        _scan_column(pk, ws, routes, lens, times, removed, n_rem, placed, r, noise_on, max_noise, rng,
                     tab_time, tab_delta, tab_pos)
    n_placed = 0
    for step in range(n_rem):
        cmax, r1, second = _top2(times, R)
        best_i = -1
        best_val = -INF
        best_c1 = INF
        best_r = -1
        for i in range(n_rem):
            if placed[i] == 1:
                continue
            # insertion cost per route
            arg_r = -1
            c1 = INF
            for r in range(R):
                if tab_pos[i, r] < 0:
                    c = BIG
                else:
                    M = second if r == r1 else cmax
                    d = tab_delta[i, r]
                    tn = times[r] + d
                    c = (tn if tn > M else M) - cmax + eps * d / R
                costs[r] = c
                if c < c1:
                    c1 = c
                    arg_r = r
            if c1 >= BIG:
                continue                        # fits nowhere (for now)
            # sort costs ascending (R is small)
            for a in range(1, R):
                v = costs[a]
                b = a - 1
                while b >= 0 and costs[b] > v:
                    costs[b + 1] = costs[b]
                    b -= 1
                costs[b + 1] = v
            regret = 0.0
            for m in range(1, k_regret):
                cm = costs[m] if m < R else BIG
                regret += cm - c1
            take = False
            if best_i < 0:
                take = True
            elif regret > best_val + 1e-12:
                take = True
            elif regret >= best_val - 1e-12:
                if c1 < best_c1 - 1e-12:
                    take = True
                elif c1 <= best_c1 + 1e-12 and removed[i] < removed[best_i]:
                    take = True
            if take:
                best_i = i
                best_val = regret
                best_c1 = c1
                best_r = arg_r
        if best_i < 0:
            break
        u = removed[best_i]
        r = best_r
        _insert_at(routes, lens, where, r, tab_pos[best_i, r], u)
        times[r] = route_time_nb(pk, ws, r, routes[r], lens[r])
        placed[best_i] = 1
        n_placed += 1
        if n_placed < n_rem:
            _scan_column(pk, ws, routes, lens, times, removed, n_rem, placed, r, noise_on, max_noise,
                         rng, tab_time, tab_delta, tab_pos)
    return n_rem - n_placed


def warmup(pk, ws, rng):
    """Compile every operator kernel on the given (tiny) packed instance."""
    T, R = pk[0], pk[2]
    routes = np.full((R, T + 1), -1, dtype=np.int64)
    lens = np.zeros(R, dtype=np.int64)
    where = np.zeros((T, 2), dtype=np.int64)
    for j in range(T):
        r = j % R
        routes[r, lens[r]] = j
        where[j, 0] = r
        where[j, 1] = lens[r]
        lens[r] += 1
    times = np.array([route_time_nb(pk, ws, r, routes[r], lens[r]) for r in range(R)])
    removed = np.zeros(T, dtype=np.int64)
    for fn in (lambda: destroy_random_nb(pk, ws, routes.copy(), lens.copy(), times.copy(), where.copy(), 1, rng, removed),
               lambda: destroy_worst_nb(pk, ws, routes.copy(), lens.copy(), times.copy(), where.copy(), 1, 3.0, 1e-3, rng, removed),
               lambda: destroy_shaw_nb(pk, ws, routes.copy(), lens.copy(), times.copy(), where.copy(), 1, 6.0, 9.0, 3.0, rng, removed)):
        fn()
    r2, l2, t2, w2 = routes.copy(), lens.copy(), times.copy(), where.copy()
    n = destroy_random_nb(pk, ws, r2, l2, t2, w2, 1, rng, removed)
    repair_nb(pk, ws, r2, l2, t2, w2, removed, n, 2, False, 0.0, 1e-3, rng)
    n = destroy_random_nb(pk, ws, r2, l2, t2, w2, 1, rng, removed)
    repair_nb(pk, ws, r2, l2, t2, w2, removed, n, 1, True, 0.01, 1e-3, rng)
