"""Train RL-CAP: the same policy, trained by REINFORCE (Sec. 7.2 and Appendix B.4).

Each step samples one trajectory for each of --instances instances; its return is
-(C_max + 10 * unserved tasks), and the advantage is the return minus that of the greedy trajectory of
the current policy on the same instance, normalized over the batch. Every --val-every steps the policy
is validated by greedy decoding on the 1,024 validation instances of SL-CAP; best.pth keeps the
parameters with the lowest mean makespan among the validations with at least 99 % complete trajectories.

    python scripts/train_rl.py --out runs/rlcap
    python scripts/train_rl.py --out runs/rlcap --resume     # continue an interrupted run

Writes <out>/best.pth, <out>/latest.pth (resume state), <out>/train.csv and <out>/val.csv. The released
checkpoints/rlcap.pth is step 95,000 of a run with these defaults.
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from slcap import fastpath
from slcap.policy import Policy
from slcap.rollout import replay_loss, rollout
from slcap.train import CSVLog, GracefulStop, Prefetcher, Validation, atomic_save, require_cuda, rng_state, set_rng_state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--instances", type=int, default=512, help="instances per batch (one trajectory each)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--entropy", type=float, default=0.01, help="entropy weight")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="batch-generation processes")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/latest.pth")
    a = ap.parse_args()

    dev = require_cuda()
    os.makedirs(a.out, exist_ok=True)
    latest, best_path = os.path.join(a.out, "latest.pth"), os.path.join(a.out, "best.pth")
    ck = torch.load(latest, map_location="cpu", weights_only=False) if a.resume and os.path.exists(latest) else None
    start = ck["step"] if ck else 0
    pf = Prefetcher(a.instances, 1, a.seed + 100003 * (start + 1) if ck else a.seed, a.workers)   # before CUDA init
    stopper = GracefulStop()

    fastpath.enable(True, tf32=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    policy = Policy().to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr)
    best, wall0 = math.inf, 0.0
    if ck:
        policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best, wall0 = ck["best"], ck["wall"]
        set_rng_state(ck["rng"])
        print(f"resumed at step {start}")
    val = Validation(dev)
    train_log, val_log = CSVLog(os.path.join(a.out, "train.csv")), CSVLog(os.path.join(a.out, "val.csv"))
    t_start = time.time() - wall0

    def save_latest(step):
        atomic_save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                     "wall": time.time() - t_start, "rng": rng_state()}, latest)

    for step in range(start + 1, a.steps + 1):
        state = pf.get(dev)
        mks, comp, reward, traj = rollout(policy, state, "sampling", record=True)
        mks_g, comp_g, reward_g, _ = rollout(policy, state, "greedy")
        adv = reward - reward_g
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        opt.zero_grad(set_to_none=True)
        loss, ent = replay_loss(policy, traj, adv, a.entropy)
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), a.clip)
        opt.step()

        if step % a.log_every == 0:
            row = {"step": step, "loss": loss, "entropy": ent, "grad_norm": float(gnorm),
                   "greedy_makespan": mks_g.mean().item(), "sample_makespan": mks.mean().item(),
                   "greedy_complete": comp_g.float().mean().item(), "K": state["K"], "N": state["N"],
                   "M": state["M"], "wall_h": (time.time() - t_start) / 3600}
            train_log.write(row)
            print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

        if step % a.val_every == 0:
            g_mean, g_comp = val.greedy(policy)
            saved = g_comp >= 0.99 and g_mean < best
            if saved:
                best = g_mean
                atomic_save({"model": policy.state_dict(), "step": step}, best_path)
            val_log.write({"step": step, "greedy_makespan": g_mean, "greedy_complete": g_comp, "saved": int(saved)})
            print(f"[val] step {step} | greedy {g_mean:.4f} ({g_comp:.3f} complete)"
                  + (" -> best.pth" if saved else ""), flush=True)
            save_latest(step)
        if stopper.requested:
            save_latest(step)
            print(f"stopped; resume with --resume (step {step})")
            break
    pf.close()


if __name__ == "__main__":
    main()
