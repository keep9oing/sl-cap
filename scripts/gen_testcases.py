"""Write test cases (Sec. 7.3): 32 instances that share K, N and M, instance i drawn with
numpy.random.RandomState(i), energy capacity E = 0.85.

    python scripts/gen_testcases.py                  # the 12 test cases of Tables 2 and 3
    python scripts/gen_testcases.py --case 4 60 3    # another case: K N M
"""
import argparse
import os

import numpy as np

from slcap import instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = [(3, 10, 3), (3, 15, 3), (3, 20, 3),                      # Table 2
         (4, 30, 3), (4, 90, 3), (4, 150, 3),                     # Table 3, tasks
         (4, 120, 3), (4, 120, 6), (4, 120, 9),                   # stations
         (3, 120, 3), (6, 120, 3), (9, 120, 3)]                   # robots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, nargs=3, metavar=("K", "N", "M"), default=None)
    ap.add_argument("-n", type=int, default=32, help="instances per case")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "testcases"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for K, N, M in [tuple(a.case)] if a.case else PAPER:
        name = f"K{K}_N{N}_M{M}"
        insts = [instances.generate(K, N, M, instances.TEST_ENERGY, np.random.RandomState(i)) for i in range(a.n)]
        path = os.path.join(a.out, name + ".json")
        instances.save(path, insts, name=name, K=K, N=N, M=M, energy=instances.TEST_ENERGY, seeds=f"0..{a.n - 1}")
        print(f"{name}: {a.n} instances -> {path}")


if __name__ == "__main__":
    main()
