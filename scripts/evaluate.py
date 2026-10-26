"""Evaluate SL-CAP, RL-CAP and the baselines on test cases (Sec. 7).

    python scripts/evaluate.py data/testcases/K4_N30_M3.json --methods slcap,rlcap,sga,alns
    python scripts/evaluate.py data/testcases/K3_N10_M3.json --methods bnb
    python scripts/evaluate.py data/testcases/K4_N120_M3.json --methods slcap,rlcap --time
    python scripts/evaluate.py data/testcases/K4_N120_M3.json --methods alns --alns-time 1.36 --workers 8

Methods
  slcap, rlcap   best of --samples sampled solutions (0: greedy decoding), on the GPU in full fp32
  sga            sequential greedy algorithm
  bnb            exact branch-and-bound (--bnb-time-limit stops it early; column `optimal`)
  alns           ALNS with --alns-iters iterations, or with a wall-clock budget --alns-time that includes
                 the SGA start solution (the equal-time comparison of Sec. 7.4)
Every plan is replayed in the simulator to score it. One CSV per test case and method is written to
<out>/<case>/<method>.csv. time_s is the solve time of the classical methods (one thread per worker:
give every worker its own core when timing). For the policies it is the batch-of-1 latency
(one instance, all its samples in one batch), measured in a separate pass with --time.
The policies' makespans reproduce the paper only for the full 32 instances with the default seed, since
the samples are drawn for chunks of instances at a time.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                      # noqa: E402
import csv                           # noqa: E402
import math                          # noqa: E402
import multiprocessing as mp         # noqa: E402
import statistics                    # noqa: E402
import time                          # noqa: E402

import torch                         # noqa: E402

from slcap import instances          # noqa: E402
from slcap.baselines.alns import N_DEFAULT, alns_solve   # noqa: E402  (importing compiles the kernels)
from slcap.baselines.bnb import bnb_solve                 # noqa: E402
from slcap.baselines.common import Instance, build_routes, replay   # noqa: E402
from slcap.baselines.sga import sga_solve                 # noqa: E402
from slcap.policy import load_policy                      # noqa: E402
from slcap.rollout import best_of, collate, rollout       # noqa: E402

POLICIES = ("slcap", "rlcap")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ classical methods
def _score(inst, I, orders):
    mks, complete, min_e = replay(inst, build_routes(I, orders))
    assert min_e > -1e-6, "infeasible plan"
    return mks, complete


def _solve(job):
    method, idx, inst, a, rate = job
    I = Instance(inst)                               # fresh per run: no memoized route times
    extra = {}
    if method == "alns" and a.alns_time is not None:
        return {"idx": idx, **_alns_budget(inst, a.alns_time, rate, a.seed)}
    t0 = time.perf_counter()
    if method == "sga":
        out = sga_solve(I)
    elif method == "bnb":
        out = bnb_solve(I, a.bnb_time_limit)
        extra["optimal"] = int(out["optimal"])
    else:
        out = alns_solve(I, iters=a.alns_iters, seed=a.seed)
        extra["iters"] = out["iters_done"]
    dt = time.perf_counter() - t0
    mks, complete = _score(inst, I, out["orders"])
    return {"idx": idx, "makespan": mks, "complete": int(complete), "time_s": f"{dt:.6f}", **extra}


def _cold_sga(inst):
    """Median time of three SGA solves on fresh instances, and the solution."""
    ts = []
    for _ in range(3):
        I = Instance(inst)
        t0 = time.perf_counter()
        sol = sga_solve(I)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), sol, I


def _alns_budget(inst, budget, rate, seed):
    """ALNS within `budget` seconds including its SGA start: the cooling schedule spans the iterations
    expected to fit (rate = iterations/s of a pilot run); SGA's own solution if nothing fits after it."""
    t_sga, sol, I = _cold_sga(inst)
    search = budget - t_sga
    if search <= 1e-3:
        mks, complete = _score(inst, I, sol["orders"])
        return {"makespan": mks, "complete": int(complete), "time_s": f"{t_sga:.6f}", "iters": 0}
    n_sched = max(5, int(round(rate * search)))
    I = Instance(inst)
    t0 = time.perf_counter()
    out = alns_solve(I, iters=max(N_DEFAULT, 10 * n_sched), seed=seed, time_limit=search, schedule_iters=n_sched)
    dt = time.perf_counter() - t0
    mks, complete = _score(inst, I, out["orders"])
    return {"makespan": mks, "complete": int(complete), "time_s": f"{dt:.6f}", "iters": out["iters_done"]}


def _pilot_rate(insts, budget):
    """ALNS iterations per second on this test case (pilot runs of max(1, budget) s on two instances)."""
    rates = []
    for inst in insts[:2]:
        t0 = time.perf_counter()
        sga_solve(Instance(inst))
        t_sga = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = alns_solve(Instance(inst), iters=N_DEFAULT, time_limit=max(1.0, budget))
        rates.append(out["iters_done"] / max(time.perf_counter() - t0 - t_sga, 1e-9))
    return statistics.median(rates)


def run_classical(method, insts, idx, a):
    rate = _pilot_rate(insts, a.alns_time) if method == "alns" and a.alns_time is not None else None
    jobs = [(method, i, insts[k], a, rate) for k, i in enumerate(idx)]
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            return pool.map(_solve, jobs, chunksize=1)
    return [_solve(j) for j in jobs]


# ------------------------------------------------------------------ policies
@torch.no_grad()
def run_policy(method, insts, idx, a):
    dev = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("the policies are evaluated on the GPU; CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    policy = load_policy(a.slcap if method == "slcap" else a.rlcap, dev)
    best = best_of(policy, insts, a.samples, dev, seed=a.seed)
    times = latency(policy, insts, a.samples, dev) if a.time else [None] * len(insts)
    return [{"idx": i, "makespan": float(b) if math.isfinite(b) else "inf", "complete": int(math.isfinite(b)),
             "time_s": "" if t is None else f"{t:.6f}"} for i, b, t in zip(idx, best.tolist(), times)]


def latency(policy, insts, K, dev, warmup=3):
    """Wall-clock time to solve each instance alone (collate + rollout of its K samples), after warm-up."""
    def once(x):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rollout(policy, collate([x] * max(K, 1), dev), "sampling" if K else "greedy")
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    for _ in range(warmup):
        once(insts[0])
    return [once(x) for x in insts]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", help="test-case files (data/testcases/*.json)")
    ap.add_argument("--methods", default="slcap,rlcap,sga,alns")
    ap.add_argument("--instances", default=None, help="slice of the instances, e.g. 0:4 (default: all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--slcap", default=os.path.join(ROOT, "checkpoints", "slcap.pth"))
    ap.add_argument("--rlcap", default=os.path.join(ROOT, "checkpoints", "rlcap.pth"))
    ap.add_argument("--samples", type=int, default=128, help="sampled solutions per instance (0: greedy)")
    ap.add_argument("--time", action="store_true", help="also measure the policies' batch-of-1 latency")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed of the policies, ALNS seed")
    ap.add_argument("--alns-iters", type=int, default=N_DEFAULT)
    ap.add_argument("--alns-time", type=float, default=None, help="ALNS wall-clock budget (s), SGA start included")
    ap.add_argument("--bnb-time-limit", type=float, default=None, help="seconds per instance")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for the classical methods")
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in POLICIES + ("sga", "bnb", "alns"):
            raise SystemExit(f"unknown method {m!r}")
    methods.sort(key=lambda m: m in POLICIES)       # classical methods first: their workers fork before CUDA

    for path in a.cases:
        data = instances.load(path)
        idx = list(range(len(data["instances"])))
        if a.instances:
            lo, hi = (int(x) if x else None for x in a.instances.split(":"))
            idx = idx[lo:hi]
        insts = [data["instances"][i] for i in idx]
        for m in methods:
            t0 = time.time()
            rows = run_policy(m, insts, idx, a) if m in POLICIES else run_classical(m, insts, idx, a)
            name = f"alns_t{a.alns_time:g}" if m == "alns" and a.alns_time is not None else m
            out = os.path.join(a.out, data["name"], name + ".csv")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            ok = [r["makespan"] for r in rows if r["complete"]]
            print(f"{data['name']:12s} {name:10s} mean makespan {statistics.mean(ok) if ok else math.nan:.4f} "
                  f"({len(ok)}/{len(rows)} complete)  {time.time() - t0:.1f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
