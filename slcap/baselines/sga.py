"""Sequential greedy algorithm (SGA; Choi, Brunet & How 2009) with a min-max rule (Sec. 7.2).

One task per round: every unassigned task is tentatively inserted into each robot's order at the
position that minimizes that robot's completion time; among the insertions with the lowest resulting
makespan, the one that gives its robot the earliest completion time is kept.
"""
from __future__ import annotations

from .common import Instance, route_time

_EPS = 1e-9


def _best_insertion(inst: Instance, rid: int, order, j):
    """(order with task j at its best position for robot rid, the resulting completion time)."""
    best = None
    for pos in range(len(order) + 1):
        cand = order[:pos] + [j] + order[pos:]
        tr = route_time(inst, rid, cand)
        if best is None or tr < best[1] - _EPS:
            best = (cand, tr)
    return best


def sga_solve(inst: Instance):
    """-> {"orders": per-robot task orders, "makespan": their makespan}."""
    K = inst.K
    orders = [[] for _ in range(K)]
    times = [route_time(inst, r, []) for r in range(K)]
    unassigned = set(range(inst.N))
    while unassigned:
        best = None   # (resulting makespan, robot's new time, task, robot, new order)
        for j in unassigned:
            for r in range(K):
                cand, tr = _best_insertion(inst, r, orders[r], j)
                res = max(tr, max((times[k] for k in range(K) if k != r), default=0.0))
                if best is None or res < best[0] - _EPS or (abs(res - best[0]) <= _EPS and tr < best[1] - _EPS):
                    best = (res, tr, j, r, cand)
        _, tr, j, r, cand = best
        orders[r] = cand
        times[r] = tr
        unassigned.discard(j)
    return {"orders": orders, "makespan": max(times)}
