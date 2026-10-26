"""Quick checks (under a minute; the last test needs a GPU).

    pytest tests
"""
import os
import random

import numpy as np
import pytest
import torch

from slcap import instances
from slcap.baselines.alns import alns_solve
from slcap.baselines.bnb import bnb_solve
from slcap.baselines.common import Instance, _solve_recharge_py, build_routes, replay, route_time_uncached
from slcap.baselines.sga import sga_solve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def case(name):
    return instances.load(os.path.join(ROOT, "data", "testcases", name + ".json"))["instances"]


def test_generator_reproduces_test_cases():
    for i, x in enumerate(case("K4_N30_M3")[:8]):
        y = instances.generate(4, 30, 3, instances.TEST_ENERGY, np.random.RandomState(i))
        assert x["energy"] == y["energy"]
        for k in ("stations", "tasks", "robots", "energy0"):
            assert np.array_equal(x[k], y[k])


def test_compiled_dp_matches_reference():
    rng = random.Random(0)
    for x in case("K3_N20_M3")[:4]:
        inst = Instance(x)
        for _ in range(50):
            order = rng.sample(range(inst.N), rng.randint(0, 12))
            r = rng.randrange(inst.K)
            _, t_ref, ok = _solve_recharge_py(inst, r, order, reconstruct=False)
            t = route_time_uncached(inst, r, order)
            assert (not ok and t == np.inf) or abs(t - t_ref) < 1e-9


def test_plans_replay_to_their_makespan():
    for x in case("K3_N10_M3")[:2]:
        results = {name: solve(Instance(x)) for name, solve in
                   (("sga", sga_solve), ("bnb", bnb_solve), ("alns", lambda i: alns_solve(i, iters=500)))}
        for out in results.values():
            mks, complete, min_e = replay(x, build_routes(Instance(x), out["orders"]))
            assert complete and min_e > -1e-6 and abs(mks - out["makespan"]) < 1e-5
        assert results["bnb"]["makespan"] <= results["alns"]["makespan"] + 1e-9
        assert results["alns"]["makespan"] <= results["sga"]["makespan"] + 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_released_models_reproduce_table2():
    """Mean best-of-128 makespan on K3_N10_M3 (Table 2: SL-CAP 1.667). The exact values were produced on
    an RTX 3090 with PyTorch 2.8 / CUDA 12.9; another GPU or library version may change the samples."""
    from slcap.policy import load_policy
    from slcap.rollout import best_of
    insts = case("K3_N10_M3")
    for name, expected in (("slcap", 1.666974), ("rlcap", 1.730463)):
        policy = load_policy(os.path.join(ROOT, "checkpoints", name + ".pth"), "cuda")
        best = best_of(policy, insts, 128, "cuda")
        assert torch.isfinite(best).all()
        assert abs(best.mean().item() - expected) < 1e-5
