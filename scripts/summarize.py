"""Summarize the results of evaluate.py: per test case and method, the mean makespan, the mean gap to
a reference method and the median time (the quantities of Tables 2 and 3).

    python scripts/summarize.py                  # reference: bnb where it proved every optimum, else alns
    python scripts/summarize.py --ref sga results

The gap of a solution with makespan C to the reference makespan C_ref of the same instance is
(C - C_ref) / C_ref, averaged over the instances that both methods completed.
"""
import argparse
import csv
import glob
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["bnb", "alns", "sga", "rlcap", "slcap"]


def read(path):
    with open(path) as f:
        return {int(r["idx"]): r for r in csv.DictReader(f)}


def case_key(name):
    m = re.fullmatch(r"K(\d+)_N(\d+)_M(\d+)", name)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default=os.path.join(ROOT, "results"))
    ap.add_argument("--ref", default=None, help="reference method (default: bnb if every instance is optimal, else alns)")
    a = ap.parse_args()
    print("| case | method | n | mean makespan | gap to ref (%) | median time (s) |")
    print("|---|---|---:|---:|---:|---:|")
    for case in sorted(os.listdir(a.results), key=case_key):
        files = glob.glob(os.path.join(a.results, case, "*.csv"))
        res = {os.path.splitext(os.path.basename(f))[0]: read(f) for f in files}
        if not res:
            continue
        ref = a.ref
        if ref is None:
            bnb = res.get("bnb", {})
            ref = "bnb" if bnb and all(r.get("optimal") == "1" for r in bnb.values()) else "alns"
        base = res.get(ref, {})
        names = sorted(res, key=lambda m: (ORDER.index(m.split("_t")[0]) if m.split("_t")[0] in ORDER else 9, m))
        for m in names:
            rows = res[m]
            done = [r for r in rows.values() if r["complete"] == "1"]
            mks = statistics.mean(float(r["makespan"]) for r in done) if done else float("nan")
            gaps = [100.0 * (float(r["makespan"]) / float(base[i]["makespan"]) - 1.0)
                    for i, r in rows.items() if r["complete"] == "1" and i in base and base[i]["complete"] == "1"]
            times = [float(r["time_s"]) for r in rows.values() if r.get("time_s")]
            gap = f"{statistics.mean(gaps):.1f}" if gaps and m != ref else ("ref" if m == ref else "")
            med = f"{statistics.median(times):.3g}" if times else ""
            print(f"| {case} | {m} | {len(done)}/{len(rows)} | {mks:.3f} | {gap} | {med} |")


if __name__ == "__main__":
    main()
