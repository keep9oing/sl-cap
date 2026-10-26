"""Shared parts of the two trainers: batch prefetching, the validation set, checkpoints, logging."""
import csv
import multiprocessing as mp
import os
import signal
import tempfile

import numpy as np
import torch

from . import fastpath, instances, rollout


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: training needs a GPU")
    return "cuda"


def _produce(q, B, repeat, seed):
    rs = np.random.RandomState(seed)
    while True:
        batch = instances.sample_batch(rs, B)
        q.put(rollout.collate([x for x in batch for _ in range(repeat)], "cpu"))


class Prefetcher:
    """Forked worker processes that generate and collate training batches on the CPU, each instance
    repeated `repeat` times in a row. Create it before CUDA is initialized."""

    def __init__(self, B, repeat, seed, workers=4, depth=2):
        ctx = mp.get_context("fork")
        self.q = ctx.Queue(maxsize=depth)
        self.procs = [ctx.Process(target=_produce, args=(self.q, B, repeat, seed + 1000 + 777 * w), daemon=True)
                      for w in range(max(1, workers))]
        for p in self.procs:
            p.start()

    def get(self, device):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in self.q.get().items()}

    def close(self):
        for p in self.procs:
            p.terminate()


class Validation:
    """The fixed validation set (1,024 instances of the training distribution, seed 12345), grouped by
    (K, M) with the tasks padded. Evaluated in full fp32 with the fast kernels."""

    def __init__(self, device, n=1024, seed=12345):
        self.insts = instances.validation_set(n, seed)
        self.device = device
        groups = {}
        for i, x in enumerate(self.insts):
            groups.setdefault((x["K"], x["M"]), []).append(i)
        self.groups = list(groups.values())
        with fastpath.full_fp32():
            self.states = [rollout.collate_padded([self.insts[i] for i in g], device) for g in self.groups]

    @torch.no_grad()
    def greedy(self, policy):
        """(mean makespan, fraction of complete trajectories) of greedy decoding."""
        with fastpath.full_fp32():
            out = [rollout.rollout(policy, st, "greedy")[:2] for st in self.states]
        return torch.cat([m for m, _ in out]).mean().item(), torch.cat([c for _, c in out]).float().mean().item()

    @torch.no_grad()
    def best_of(self, policy, K=128, max_rows=2048, seed=0):
        """Mean best-of-K makespan; inf unless every instance has a complete sample. Uses its own random
        stream, so validating does not change the training run."""
        best = []
        with fastpath.full_fp32(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            per = max(1, max_rows // K)
            for g in self.groups:
                order = sorted(g, key=lambda i: (self.insts[i]["N"], i))
                for s in range(0, len(order), per):
                    chunk = [self.insts[i] for i in order[s:s + per]]
                    st = rollout.collate_padded([x for x in chunk for _ in range(K)], self.device)
                    mks, comp, _, _ = rollout.rollout(policy, st, "sampling")
                    best.append(mks.masked_fill(~comp, float("inf")).view(len(chunk), K).min(dim=1).values)
        b = torch.cat(best)
        return b.mean().item() if bool(torch.isfinite(b).all()) else float("inf")


def atomic_save(obj, path):
    """torch.save through a temporary file, so an interrupted save never leaves a truncated file."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".pth", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def rng_state():
    return {"np": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


class GracefulStop:
    """SIGTERM/SIGINT set a flag; the trainer finishes the step, saves latest.pth and exits."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):  # noqa: ARG002
        self.requested = True


class CSVLog:
    """Appends dict rows to a CSV file (header from the first row)."""

    def __init__(self, path):
        self.path = path

    def write(self, row):
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
