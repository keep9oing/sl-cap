"""Adaptive large neighborhood search (ALNS; Ropke & Pisinger 2006) on top of the `alns` package
(Wouda & Lan 2023), Sec. 7.2 and Appendix B.3.

Each iteration removes q tasks by Shaw, random or worst removal, 4 <= q <= min(100, floor(0.4 N)),
and reinserts them by basic greedy or regret-2/3/4 insertion, each with and without noise; roulette
wheels choose the operators (weights updated every 100 iterations, reaction 0.1, scores 33/9/13) and
simulated annealing accepts (start temperature: a solution 5 % worse than the SGA start is accepted
with probability 0.5; cooling 0.99975 per iteration over 25,000 iterations). The parameter values are
those of Ropke & Pisinger, not tuned. Deviations: the search minimizes C_max + 1e-3 * mean robot
completion time and reports the accepted solution with the smallest makespan; the Shaw load and
vehicle terms are dropped (identical robots, no loads); the regret insertion over all robots is not
used. A budget other than 25,000 iterations keeps the start and end temperatures and adjusts the
cooling factor (`schedule_iters`). The operators are compiled (alns_kernels.py).
"""
from __future__ import annotations

import math
import time

import numpy as np
from alns import ALNS
from alns.Outcome import Outcome
from alns.accept import SimulatedAnnealing
from alns.select import OperatorSelectionScheme

from .common import Instance, makespan_of, _nb_ctx
from .sga import sga_solve
from . import rc_kernels as K
from . import alns_kernels as AK

N_DEFAULT = 25000
PHI, CHI = 9.0, 3.0            # Shaw relatedness weights (distance, arrival time)
P_SHAW, P_WORST = 6.0, 3.0     # randomization of Shaw / worst removal
W_START = 0.05                 # start temperature: a 5 %-worse solution accepted with probability 0.5
C_END_RATIO = 0.99975 ** 25000 # end/start temperature ratio of the 25,000-iteration schedule
SCORES = [33.0, 9.0, 13.0, 0.0]
REACTION = 0.1
SEG_LEN = 100
ETA = 0.025                    # noise amplitude (times the largest pairwise distance)
XI = 0.4                       # destroy size fraction
Q_MIN, Q_CAP = 4, 100
EPS_F = 1e-3                   # weight of the mean completion time in the search objective

_NAMES_D = ("shaw", "random", "worst")
_NAMES_R = ("greedy", "regret2", "regret3", "regret4",
            "greedy_noise", "regret2_noise", "regret3_noise", "regret4_noise")


class _Solution:
    """Padded-array solution; `objective()` is the search objective F."""
    __slots__ = ("inst", "pk", "ws", "routes", "lens", "times", "where", "removed", "n_removed",
                 "cmax", "F", "n_unplaced")

    def __init__(self, inst, pk, ws, routes, lens, times, where):
        self.inst, self.pk, self.ws = inst, pk, ws
        self.routes, self.lens, self.times, self.where = routes, lens, times, where
        self.removed = np.zeros(where.shape[0], dtype=np.int64)
        self.n_removed = 0
        self.n_unplaced = 0
        self.finalize(0)

    @classmethod
    def from_orders(cls, inst, orders):
        pk, ws = _nb_ctx(inst)
        routes = np.full((inst.K, inst.N + 1), -1, dtype=np.int64)
        lens = np.zeros(inst.K, dtype=np.int64)
        where = np.full((inst.N, 2), -1, dtype=np.int64)
        for r, order in enumerate(orders):
            for pos, j in enumerate(order):
                routes[r, pos] = j
                where[j, 0] = r
                where[j, 1] = pos
            lens[r] = len(order)
        times = np.array([K.route_time_nb(pk, ws, r, routes[r], lens[r]) for r in range(inst.K)], dtype=np.float64)
        return cls(inst, pk, ws, routes, lens, times, where)

    def copy(self):
        s = _Solution.__new__(_Solution)
        s.inst, s.pk, s.ws = self.inst, self.pk, self.ws
        s.routes = self.routes.copy()
        s.lens = self.lens.copy()
        s.times = self.times.copy()
        s.where = self.where.copy()
        s.removed = np.zeros(self.where.shape[0], dtype=np.int64)
        s.n_removed = 0
        s.n_unplaced = 0
        s.cmax, s.F = self.cmax, self.F
        return s

    def finalize(self, n_unplaced):
        self.n_unplaced = int(n_unplaced)
        if n_unplaced > 0 or not np.all(np.isfinite(self.times)):
            self.cmax = math.inf
            self.F = math.inf
        else:
            self.cmax = float(self.times.max())
            self.F = self.cmax + EPS_F * float(self.times.mean())

    def objective(self):
        return self.F

    def key(self):
        """Hashable identity (Ropke & Pisinger's visited-solution hash key)."""
        return hash((self.routes.tobytes(), self.lens.tobytes()))


class _RPRouletteWheel(OperatorSelectionScheme):
    """Ropke & Pisinger's adaptive weights: at the end of every segment w <- (1-r) w + r pi / theta
    (pi = score collected, theta = times used); sigma2 and sigma3 only for unvisited solutions.
    The package's RouletteWheel neither divides by theta nor remembers visited solutions."""
    __name__ = "rp_roulette_wheel"

    def __init__(self, scores, reaction, seg_length, num_destroy, num_repair):
        super().__init__(num_destroy, num_repair)
        self.scores = [float(s) for s in scores]
        self.r = float(reaction)
        self.seg = int(seg_length)
        self.d_w = np.ones(num_destroy)
        self.r_w = np.ones(num_repair)
        self.d_pi = np.zeros(num_destroy)
        self.d_theta = np.zeros(num_destroy)
        self.r_pi = np.zeros(num_repair)
        self.r_theta = np.zeros(num_repair)
        self.iter = 0
        self.visited = set()

    @property
    def destroy_weights(self):
        return self.d_w

    @property
    def repair_weights(self):
        return self.r_w

    def _end_segment(self):
        for w, pi, theta in ((self.d_w, self.d_pi, self.d_theta), (self.r_w, self.r_pi, self.r_theta)):
            used = theta > 0
            w[used] = (1.0 - self.r) * w[used] + self.r * pi[used] / theta[used]
            pi[:] = 0.0
            theta[:] = 0.0

    def __call__(self, rng, best, curr):
        self.iter += 1
        if self.iter % self.seg == 0:
            self._end_segment()
        d = int(rng.choice(self.num_destroy, p=self.d_w / self.d_w.sum()))
        r = int(rng.choice(self.num_repair, p=self.r_w / self.r_w.sum()))
        return d, r

    def update(self, cand, d_idx, r_idx, outcome):
        self.d_theta[d_idx] += 1
        self.r_theta[r_idx] += 1
        key = cand.key()
        unvisited = key not in self.visited
        self.visited.add(key)
        if outcome == Outcome.BEST:
            s = self.scores[0]
        elif outcome == Outcome.BETTER and unvisited:
            s = self.scores[1]
        elif outcome == Outcome.ACCEPT and unvisited:
            s = self.scores[2]
        else:
            s = 0.0
        self.d_pi[d_idx] += s
        self.r_pi[r_idx] += s


def _draw_q(rng, N):
    """4 <= q <= min(100, floor(0.4 N)); for N < 10 the interval is empty and q = 4."""
    q_hi = max(Q_MIN, min(Q_CAP, int(math.floor(XI * N))))
    q_lo = min(Q_MIN, max(1, N - 1))
    return int(rng.integers(q_lo, q_hi + 1))


def _destroy_ops(inst):
    pk, ws = _nb_ctx(inst)
    N = inst.N

    def shaw(state, rng):
        s = state.copy()
        s.n_removed = AK.destroy_shaw_nb(pk, ws, s.routes, s.lens, s.times, s.where, _draw_q(rng, N),
                                         P_SHAW, PHI, CHI, rng, s.removed)
        return s

    def random(state, rng):
        s = state.copy()
        s.n_removed = AK.destroy_random_nb(pk, ws, s.routes, s.lens, s.times, s.where, _draw_q(rng, N),
                                           rng, s.removed)
        return s

    def worst(state, rng):
        s = state.copy()
        s.n_removed = AK.destroy_worst_nb(pk, ws, s.routes, s.lens, s.times, s.where, _draw_q(rng, N),
                                          P_WORST, EPS_F, rng, s.removed)
        return s

    return [("shaw", shaw), ("random", random), ("worst", worst)]


def _repair_ops(inst):
    pk, ws = _nb_ctx(inst)
    max_noise = ETA * pk[14] / pk[3]          # eta * largest pairwise distance, in time units

    def make(k, noise):
        def repair(state, rng):
            n_unplaced = AK.repair_nb(pk, ws, state.routes, state.lens, state.times, state.where,
                                      state.removed, state.n_removed, k, noise, max_noise, EPS_F, rng)
            state.n_removed = 0
            state.finalize(n_unplaced)
            return state
        return repair

    return list(zip(_NAMES_R, [make(k, noise) for noise in (False, True) for k in (1, 2, 3, 4)]))


class _BestTracker:
    """Smallest makespan among the accepted solutions (the reported result)."""
    __name__ = "best_tracker"

    def __init__(self, sol):
        self.best_cmax = sol.cmax
        self.routes = sol.routes.copy()
        self.lens = sol.lens.copy()

    def __call__(self, cand, rng, **kwargs):
        if cand.cmax < self.best_cmax - 1e-12:
            self.best_cmax = cand.cmax
            self.routes[...] = cand.routes
            self.lens[...] = cand.lens

    def orders(self):
        return [[int(x) for x in self.routes[r, :self.lens[r]]] for r in range(self.routes.shape[0])]


class _Stop:
    """Stop after `iters` iterations or `time_limit` seconds of search, whichever comes first."""

    def __init__(self, iters, time_limit):
        self.iters, self.time_limit = int(iters), time_limit
        self.calls = 0
        self.t0 = time.perf_counter()

    def __call__(self, rng, best, curr):
        self.calls += 1
        if self.calls - 1 >= self.iters:
            return True
        return self.time_limit is not None and time.perf_counter() - self.t0 >= self.time_limit


def alns_solve(inst: Instance, iters=N_DEFAULT, seed=0, time_limit=None, schedule_iters=None):
    """ALNS from the SGA solution. `iters` caps the loop, `time_limit` (s) caps the search (the SGA
    start is not included), and the cooling schedule spans `schedule_iters` (default `iters`) iterations.
    -> {"orders", "makespan", "iters_done"}"""
    n_sched = int(schedule_iters) if schedule_iters else int(iters)
    sol0 = _Solution.from_orders(inst, sga_solve(inst)["orders"])
    rng = np.random.default_rng(seed)
    alg = ALNS(rng)
    for name, fn in _destroy_ops(inst):
        alg.add_destroy_operator(fn, name=name)
    for name, fn in _repair_ops(inst):
        alg.add_repair_operator(fn, name=name)
    tracker = _BestTracker(sol0)
    alg.on_best(tracker)
    alg.on_better(tracker)
    alg.on_accept(tracker)
    select = _RPRouletteWheel(SCORES, REACTION, SEG_LEN, len(_NAMES_D), len(_NAMES_R))
    if not math.isfinite(sol0.F):
        raise ValueError("initial (SGA) solution infeasible")
    t_start = -W_START * sol0.F / math.log(0.5)
    accept = SimulatedAnnealing(t_start, t_start * C_END_RATIO, C_END_RATIO ** (1.0 / n_sched), method="exponential")
    stop = _Stop(iters, time_limit)
    alg.iterate(sol0, select, accept, stop)
    orders = tracker.orders()
    return {"orders": orders, "makespan": makespan_of(inst, orders), "iters_done": stop.calls - 1}


def warmup():
    """Compile the kernels on a tiny instance (keeps compilation out of every timer)."""
    K.warmup()
    rng = np.random.default_rng(0)
    tp = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9], [0.2, 0.6], [0.7, 0.7], [0.3, 0.5]])
    sp = np.array([[0.5, 0.5], [0.15, 0.15]])
    rp = np.array([[0.5, 0.45], [0.6, 0.6]])
    e0 = np.array([0.6, 0.4])
    vel, mt, ct = 1.0, 0.8, 0.2
    M = sp.shape[0]
    Dt = np.linalg.norm(sp[:, None, :] - sp[None, :, :], axis=-1) / vel
    cost = np.full((M, M), np.inf)
    for s in range(M):
        cost[s, s] = 0.0
        for s2 in range(M):
            if s2 != s and Dt[s, s2] <= mt + 1e-9:
                cost[s, s2] = Dt[s, s2] + ct
    pk = K.pack_arrays(tp, sp, rp, e0, vel, mt, ct, cost)
    AK.warmup(pk, K.make_workspace(pk), rng)


warmup()
