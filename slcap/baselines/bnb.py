"""Exact branch-and-bound over task-to-robot assignments (Sec. 7.2).

Each robot's best order for its assigned tasks is found by enumeration (memoized per robot and task
subset). A partial assignment is pruned once some robot's minimum completion time reaches the best
makespan found so far; the SGA makespan is the initial bound. This is exact because adding a task to
a robot never shortens its best route. Tractable for small instances only (Table 2: N <= 15).
"""
from __future__ import annotations

import itertools
import time

from .common import Instance, route_time_uncached
from .sga import sga_solve


class _TimeUp(Exception):
    pass


def bnb_solve(inst: Instance, time_limit=None):
    """-> {"orders", "makespan", "optimal"}. With `time_limit` (s) the search may stop early and
    return the best assignment found so far (the SGA solution if none), with optimal=False."""
    deadline = None if time_limit is None else time.perf_counter() + time_limit

    def check_time():
        if deadline is not None and time.perf_counter() > deadline:
            raise _TimeUp

    K, N = inst.K, inst.N
    memo = {}

    def best_order(r, subset):
        key = (r, subset)
        if key not in memo:
            if not subset:
                memo[key] = ([], route_time_uncached(inst, r, []))
            else:
                best = None
                for n, perm in enumerate(itertools.permutations(subset)):
                    if not n & 1023:
                        check_time()
                    t = route_time_uncached(inst, r, perm)     # every (robot, order) is seen once
                    if best is None or t < best[1]:
                        best = (list(perm), t)
                memo[key] = best
        return memo[key]

    sga = sga_solve(inst)
    ub = sga["makespan"] * (1 + 1e-9) + 1e-12        # prune only; float noise must not cut the optimum
    incumbent = [None, float("inf")]
    subsets = [frozenset() for _ in range(K)]
    times = [best_order(r, subsets[r])[1] for r in range(K)]

    def rec(i):
        check_time()
        if i == N:
            incumbent[0] = [best_order(r, subsets[r])[0] for r in range(K)]
            incumbent[1] = max(times)
            return
        for r in range(K):
            s_old, t_old = subsets[r], times[r]
            s_new = s_old | {i}
            t_new = best_order(r, s_new)[1]
            bound = max([t_new] + [times[k] for k in range(K) if k != r])
            if bound >= incumbent[1] or bound > ub:
                continue
            subsets[r], times[r] = s_new, t_new
            rec(i + 1)
            subsets[r], times[r] = s_old, t_old

    try:
        rec(0)
        optimal = True
    except _TimeUp:
        optimal = False
    if incumbent[0] is None:
        return {"orders": sga["orders"], "makespan": sga["makespan"], "optimal": False}
    return {"orders": incumbent[0], "makespan": incumbent[1], "optimal": optimal}
