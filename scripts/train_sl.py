"""Train SL-CAP by self-labeling (Sec. 6; the defaults are those of Table 1).

Each step samples --samples trajectories for every instance of a batch of --instances and trains, by
cross-entropy, on the complete trajectory with the lowest makespan of each instance (self-labeling,
Corsini et al., NeurIPS 2024). Every --val-every steps the policy is validated on 1,024 fixed instances;
best.pth keeps the parameters with the lowest mean best-of-128 makespan among the validations in which
every instance had a complete sample and greedy decoding completed at least 99 % of them (ties: lower
greedy mean, then the later step).

    python scripts/train_sl.py --out runs/slcap
    python scripts/train_sl.py --out runs/slcap --resume     # continue an interrupted run

Writes <out>/best.pth, <out>/latest.pth (resume state), <out>/train.csv and <out>/val.csv. The released
checkpoints/slcap.pth is step 97,500 of a run with these defaults.
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from slcap import fastpath
from slcap.policy import Policy
from slcap.rollout import replay_loss, rollout, select
from slcap.train import CSVLog, GracefulStop, Prefetcher, Validation, atomic_save, require_cuda, rng_state, set_rng_state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--instances", type=int, default=256, help="instances per batch")
    ap.add_argument("--samples", type=int, default=8, help="sampled trajectories per instance")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--entropy", type=float, default=0.01, help="entropy weight")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--val-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4, help="batch-generation processes")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/latest.pth")
    a = ap.parse_args()

    dev = require_cuda()
    os.makedirs(a.out, exist_ok=True)
    latest, best_path = os.path.join(a.out, "latest.pth"), os.path.join(a.out, "best.pth")
    ck = torch.load(latest, map_location="cpu", weights_only=False) if a.resume and os.path.exists(latest) else None
    start = ck["step"] if ck else 0
    I, S = a.instances, a.samples
    # a resumed run continues with a new instance stream instead of repeating the batches it has seen
    pf = Prefetcher(I, S, a.seed + 100003 * (start + 1) if ck else a.seed, a.workers)   # before CUDA init
    stopper = GracefulStop()

    fastpath.enable(True, tf32=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    policy = Policy().to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr)
    best_key, wall0 = (math.inf, math.inf), 0.0
    if ck:
        policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best_key, wall0 = tuple(ck["best_key"]), ck["wall"]
        set_rng_state(ck["rng"])
        print(f"resumed at step {start}")
    val = Validation(dev)
    train_log, val_log = CSVLog(os.path.join(a.out, "train.csv")), CSVLog(os.path.join(a.out, "val.csv"))
    t_start = time.time() - wall0

    def save_latest(step):
        atomic_save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": step, "best_key": best_key,
                     "wall": time.time() - t_start, "rng": rng_state()}, latest)

    for step in range(start + 1, a.steps + 1):
        state = pf.get(dev)
        mks, comp, _, traj = rollout(policy, state, "sampling", record=True)
        cand = mks.masked_fill(~comp, math.inf).view(I, S)
        has = comp.view(I, S).any(dim=1)
        labels = (torch.arange(I, device=dev) * S + cand.argmin(dim=1))[has]     # best complete sample
        opt.zero_grad(set_to_none=True)
        ce, ent = replay_loss(policy, select(traj, labels), torch.ones(labels.numel(), device=dev), a.entropy)
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), a.clip)
        opt.step()

        if step % a.log_every == 0:
            row = {"step": step, "ce": ce, "entropy": ent, "grad_norm": float(gnorm),
                   "label_makespan": cand.min(dim=1).values[has].mean().item() if has.any() else math.nan,
                   "sample_complete": comp.float().mean().item(), "K": state["K"], "N": state["N"], "M": state["M"],
                   "wall_h": (time.time() - t_start) / 3600}
            train_log.write(row)
            print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

        if step % a.val_every == 0:
            g_mean, g_comp = val.greedy(policy)
            bo = val.best_of(policy, a.val_samples)
            eligible = math.isfinite(bo) and g_comp >= 0.99
            saved = eligible and (bo, g_mean) <= best_key
            if saved:
                best_key = (bo, g_mean)
                atomic_save({"model": policy.state_dict(), "step": step}, best_path)
            val_log.write({"step": step, "greedy_makespan": g_mean, "greedy_complete": g_comp,
                           f"best{a.val_samples}_makespan": bo, "eligible": int(eligible), "saved": int(saved)})
            print(f"[val] step {step} | greedy {g_mean:.4f} ({g_comp:.3f} complete) | best-of-{a.val_samples} "
                  f"{bo:.4f}" + (" -> best.pth" if saved else ""), flush=True)
            save_latest(step)
        if stopper.requested:
            save_latest(step)
            print(f"stopped; resume with --resume (step {step})")
            break
    pf.close()


if __name__ == "__main__":
    main()
