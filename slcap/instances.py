"""Problem instances: the generator of Appendix A, the training distribution of Table 1, and JSON files.

An instance is a dict
    K, N, M        numbers of robots, tasks and stations
    energy         energy capacity E (a full charge lasts E time units and carries a robot v*E far)
    charge_time    recharge duration tau
    speed          speed v
    stations (M, 2), tasks (N, 2), robots (K, 2)    positions in the unit square W (float32)
    energy0 (K,)   initial energy levels (float32)

The generator draws everything from one numpy RandomState, in a fixed order, so a seed determines
an instance. The test cases of the paper are `generate(K, N, M, TEST_ENERGY, RandomState(i))` for
instance i = 0..31 (scripts/gen_testcases.py).
"""
import json
import math

import numpy as np
from scipy.spatial import distance_matrix

SPEED = 1.0
CHARGE_TIME = 0.2
EPS = 0.005            # margin of the generator's distance tests
GEN_CAP = 2000         # rejection-sampling rounds before a task placement counts as failed

# training distribution (Table 1); the operation range v*E is 40-80 % of the diagonal of W
TRAIN_K, TRAIN_N, TRAIN_M = (2, 5), (10, 50), (2, 5)
TRAIN_RANGE = (0.4, 0.8)


def capacity(range_frac, speed=SPEED):
    """Energy capacity E whose operation range v*E is `range_frac` times the diagonal of W."""
    return range_frac * math.sqrt(2.0) / speed


TEST_ENERGY = capacity(0.6)    # E = 0.85, the middle of the training range (Sec. 7.3)


def generate(K, N, M, energy, rs, charge_time=CHARGE_TIME, speed=SPEED):
    """One instance with K robots, N tasks and M stations (Appendix A), drawn with RandomState rs.
    Every instance admits a feasible solution: the stations form a connected network under one full
    charge, every task lies within half the operation range of a station, and every robot can reach a
    station on its initial energy."""
    half = energy * speed / 2                  # a fully charged robot can go this far and back
    for _ in range(10):
        stations = _stations(M, half, rs)
        tasks = _tasks(N, stations, half, rs)
        if tasks is None:
            continue
        robots = _robots(K, stations, energy, speed, rs)
        if robots is None:
            continue
        return {"K": K, "N": N, "M": M, "energy": float(energy), "charge_time": float(charge_time),
                "speed": float(speed), "stations": stations, "tasks": tasks,
                "robots": robots[0], "energy0": robots[1]}
    raise ValueError("instance generation failed 10 times")


def _stations(M, half, rs):
    """A random tree: each new station attaches to a random placed one, at 55-95 % of one full-charge
    hop. Clipping to W only shortens the edge to the parent, so the network stays connected."""
    hop = 2.0 * half - EPS
    lo, hi = 0.55 * hop, 0.95 * hop
    pts = [rs.uniform(0.0, 1.0, size=2).tolist()]
    while len(pts) < M:
        base = np.array(pts[rs.randint(len(pts))])
        step = rs.uniform(lo, hi)
        ang = rs.uniform(0.0, 2.0 * np.pi)
        pts.append(np.clip(base + step * np.array([np.cos(ang), np.sin(ang)]), 0.0, 1.0).tolist())
    return np.array(pts, dtype=np.float64).astype(np.float32)


def _tasks(N, stations, half, rs):
    """Uniform in W, kept if the nearest station lies within half the operation range."""
    hi = half - EPS
    out, tries = [], 0
    while len(out) < N:
        cand = rs.uniform(low=0, high=1, size=(N - len(out), 2))
        near = distance_matrix(stations, cand).min(axis=0)
        out.extend(cand[np.logical_and(near > 0.0, near <= hi)].tolist())
        tries += 1
        if tries > GEN_CAP:
            return None
    return np.array(out[:N]).astype(np.float32)


def _robots(K, stations, energy, speed, rs):
    """Initial energy U[0.01, E]; position uniform in W, kept if a station is within reach of that energy."""
    e0 = rs.uniform(low=0.01, high=energy, size=(1, K, 1)).astype(np.float32)
    reach = (e0 * speed)[0].T
    pos = np.zeros((K, 2)).astype(np.float32)
    ids = np.arange(K)
    placed, rounds = 0, 0
    while placed < K:
        cand = rs.uniform(low=0, high=1, size=(K - placed, 2))
        dist = distance_matrix(stations, cand)
        idx = np.where(np.any(np.logical_and(dist > 0.001, dist < reach - EPS), axis=0))[0]
        pos[ids[idx]] = np.array(cand[idx])
        ids = np.delete(ids, idx)
        reach = np.delete(reach, idx)
        placed += len(idx)
        rounds += 1
        if rounds > 100 * K:
            return None
    return pos, e0[0, :, 0]


def sample_instance(rs):
    """One instance of the training distribution, sizes and E drawn with rs (the validation set)."""
    N = rs.randint(TRAIN_N[0], TRAIN_N[1] + 1)
    K = rs.randint(TRAIN_K[0], TRAIN_K[1] + 1)
    E = capacity(rs.uniform(*TRAIN_RANGE))
    M = rs.randint(TRAIN_M[0], TRAIN_M[1] + 1)
    return generate(K, N, M, E, rs)


def sample_batch(rs, B):
    """B instances of the training distribution that share K, N, M and E (one training batch)."""
    K = rs.randint(TRAIN_K[0], TRAIN_K[1] + 1)
    N = rs.randint(TRAIN_N[0], TRAIN_N[1] + 1)
    E = capacity(rs.uniform(*TRAIN_RANGE))
    M = rs.randint(TRAIN_M[0], TRAIN_M[1] + 1)
    return [generate(K, N, M, E, np.random.RandomState(rs.randint(2 ** 31))) for _ in range(B)]


def validation_set(n=1024, seed=12345):
    """The fixed validation set used for model selection: instance i drawn with RandomState(seed + i)."""
    return [sample_instance(np.random.RandomState(seed + i)) for i in range(n)]


_ARRAYS = ("stations", "tasks", "robots", "energy0")


def save(path, instances, **meta):
    rows = [{k: (v.tolist() if k in _ARRAYS else v) for k, v in x.items()} for x in instances]
    with open(path, "w") as f:
        json.dump({**meta, "instances": rows}, f)


def load(path):
    """A test-case file -> dict with its metadata and `instances` (arrays restored to float32)."""
    with open(path) as f:
        data = json.load(f)
    for x in data["instances"]:
        for k in _ARRAYS:
            x[k] = np.asarray(x[k], dtype=np.float32)
    return data
