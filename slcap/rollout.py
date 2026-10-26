"""Batched simulator on the GPU: B instances of one size (K, N, M) solved in parallel.

Each step, every instance makes one decision for its robot that becomes free first (Sec. 4): park,
recharge at a station, or travel to a task. Stations serve any number of robots at once, so the only
state is when each robot becomes free; the step is an argmin over that vector. The action mask
(Sec. 5.3) keeps every trajectory energy-feasible: a station must be reachable on the remaining
energy, a task must leave enough energy to reach its nearest station, and the last active robot may
not park while unclaimed tasks remain.

`rollout(record=True)` also returns the decoder inputs of every decision, from which `replay_loss`
recomputes the log-probabilities with gradients (training).
"""
import numpy as np
import torch

from . import fastpath

INF = 1.0e9
BETA = 10.0      # penalty per unserved task in the return of RL-CAP (Sec. 7.2)


def collate(instances, device):
    """Instances of one size -> dict of batched tensors."""
    K = instances[0]["K"]
    stack = lambda key: np.stack([x[key] for x in instances]).astype(np.float32)   # noqa: E731
    per_robot = lambda key: np.array([[x[key]] * K for x in instances], dtype=np.float32)  # noqa: E731
    t = lambda a: torch.from_numpy(a).to(device)   # noqa: E731
    spos, tpos = t(stack("stations")), t(stack("tasks"))
    return {
        "B": len(instances), "K": K, "N": instances[0]["N"], "M": instances[0]["M"],
        "rpos": t(stack("robots")), "energy": t(stack("energy0")),
        "vel": t(per_robot("speed")), "max_time": t(per_robot("energy")), "charge_time": t(per_robot("charge_time")),
        "spos": spos, "tpos": tpos,
        "task_near_dist": torch.cdist(tpos, spos).min(dim=2).values,   # for the mask
    }


def collate_padded(instances, device):
    """Instances that share K and M but not N, tasks padded to the largest N (fast kernels only):
    padded tasks start claimed and are left out of the attention, so every instance gets the same
    computation as in a batch of its own size."""
    n = [x["N"] for x in instances]
    Nmax = max(n)
    if min(n) == Nmax:
        return collate(instances, device)
    padded = []
    for x in instances:
        if x["N"] < Nmax:
            fill = np.repeat(x["tasks"][:1], Nmax - x["N"], axis=0)
            x = dict(x, tasks=np.concatenate([x["tasks"], fill]), N=Nmax)
        padded.append(x)
    st = collate(padded, device)
    n_task = torch.tensor(n, device=device)
    st["task_pad"] = torch.arange(Nmax, device=device)[None, :] >= n_task[:, None]
    st["n_task"] = n_task
    return st


@torch.no_grad()
def rollout(policy, state, decode="sampling", record=False):
    """decode: "sampling" or "greedy". Returns (makespan, complete, reward, traj), the first three
    per instance; reward = -(makespan + BETA * unserved tasks), the return of RL-CAP."""
    B, K, N, M = state["B"], state["K"], state["N"], state["M"]
    dev = state["spos"].device
    rpos, energy = state["rpos"].clone(), state["energy"].clone()
    vel, max_time, charge_time = state["vel"], state["max_time"], state["charge_time"]
    spos, tpos, task_near_dist = state["spos"], state["tpos"], state["task_near_dist"]
    task_near_exact = (tpos.unsqueeze(2) - spos.unsqueeze(1)).norm(dim=-1).min(dim=2).values   # for the inputs

    free_time = torch.zeros(B, K, device=dev)
    alive = torch.ones(B, K, dtype=torch.bool, device=dev)
    at_station = torch.full((B, K), -1, dtype=torch.long, device=dev)
    claimed = torch.zeros(B, N, dtype=torch.bool, device=dev)
    completion = torch.zeros(B, K, device=dev)
    task_pad = state.get("task_pad")
    if task_pad is not None:
        if not fastpath.ENABLED or record:
            raise ValueError("padded batches are for evaluation with the fast kernels")
        claimed = task_pad.clone()
        n_task = state["n_task"].to(torch.float32)
        cand_pad = torch.cat([torch.zeros(B, 1 + M, dtype=torch.bool, device=dev), task_pad], dim=1)
    bidx = torch.arange(B, device=dev)
    traj = [] if record else None

    max_steps = 6 * (N + K) + 40
    fast = fastpath.ENABLED
    check_every = fastpath.DONE_CHECK_EVERY if fast else 1
    B0, rows, full = B, None, None
    step = 0
    while step < max_steps:
        step += 1
        done = ~alive.any(dim=1)
        if not fast:
            if bool(done.all()):
                break
        elif (step - 1) % check_every == 0:
            n_act = int((~done).sum())
            if n_act == 0:
                break
            if n_act <= fastpath.COMPACT_FRAC * B:   # shrink the batch to the running instances
                keep = (~done).nonzero(as_tuple=True)[0]
                if full is None:
                    rows = torch.arange(B0, device=dev)
                    full = {"alive": alive.clone(), "claimed": claimed.clone(), "completion": completion.clone()}
                else:
                    for k_, v_ in (("alive", alive), ("claimed", claimed), ("completion", completion)):
                        full[k_].index_copy_(0, rows, v_)
                rows = rows[keep]
                (rpos, energy, vel, max_time, charge_time, spos, tpos, task_near_dist, task_near_exact,
                 free_time, alive, at_station, claimed, completion, done) = (
                    x[keep] for x in (rpos, energy, vel, max_time, charge_time, spos, tpos, task_near_dist,
                                      task_near_exact, free_time, alive, at_station, claimed, completion, done))
                if task_pad is not None:
                    n_task, cand_pad = n_task[keep], cand_pad[keep]
                B = keep.numel()
                bidx = torch.arange(B, device=dev)
        active = ~done

        ft = free_time.masked_fill(~alive, INF)
        dec = ft.argmin(dim=1)                                  # the deciding robot
        charged = (at_station[bidx, dec] >= 0) & active         # just finished recharging -> full
        energy[bidx, dec] = torch.where(charged, max_time[bidx, dec], energy[bidx, dec])
        now = free_time[bidx, dec]
        self_pos, self_vel = rpos[bidx, dec], vel[bidx, dec]
        self_energy, self_charge, self_at = energy[bidx, dec], charge_time[bidx, dec], at_station[bidx, dec]

        # action mask
        d_station = (spos - self_pos[:, None, :]).norm(dim=-1)
        station_invalid = d_station / self_vel[:, None] > self_energy[:, None]
        t_task = (tpos - self_pos[:, None, :]).norm(dim=-1) / self_vel[:, None]
        task_invalid = (t_task + task_near_dist / self_vel[:, None]) > self_energy[:, None]
        cur = self_at.clamp(min=0)
        station_invalid = station_invalid | (
            (self_at[:, None] >= 0) & (torch.arange(M, device=dev)[None, :] == cur[:, None]))
        task_masked = claimed | task_invalid
        has_unclaimed = (~claimed).any(dim=1)
        all_masked = station_invalid.all(dim=1) & task_masked.all(dim=1)
        must_work = active & has_unclaimed & (~all_masked)
        finish_forbidden = must_work & (alive.sum(dim=1) == 1)
        fm = torch.zeros(B, 1 + M + N, device=dev)
        fm[:, 0] = finish_forbidden.float()
        fm[:, 1:1 + M] = station_invalid.float()
        fm[:, 1 + M:] = task_masked.float()

        # inputs, relative to the deciding robot
        is_self = torch.zeros(B, K, device=dev)
        is_self[bidx, dec] = 1.0
        if task_pad is None:
            frac_unclaimed = (~claimed).float().mean(dim=1, keepdim=True).expand(B, K)
        else:
            frac_unclaimed = ((~claimed).float().sum(dim=1, keepdim=True) / n_task[:, None]).expand(B, K)
        robots_in = torch.cat([
            alive.float()[..., None], rpos - self_pos[:, None, :],
            (free_time - now[:, None]).clamp(min=0)[..., None], energy[..., None], max_time[..., None],
            is_self[..., None], frac_unclaimed[..., None],
        ], dim=-1)
        near_time = task_near_exact / self_vel[:, None]
        slack_station = self_energy[:, None] - d_station / self_vel[:, None]
        slack_task = self_energy[:, None] - (t_task + near_time)
        station_in = torch.cat([spos - self_pos[:, None, :], slack_station[..., None]], dim=-1)
        task_in = torch.cat([tpos - self_pos[:, None, :], claimed.float()[..., None],
                             near_time[..., None], slack_task[..., None]], dim=-1)
        log_prob = policy(robots_in, station_in, task_in, fm, pad=None if task_pad is None else cand_pad)
        if decode == "greedy":
            a = log_prob.argmax(dim=1)
        elif fast:
            a = fastpath.sample(log_prob)
        else:
            # The evaluation behind the paper's numbers drew one extra sample per decision and
            # discarded it; drawing it keeps the random stream, so the sampled solutions are the same.
            torch.multinomial(log_prob.exp(), 1)
            a = torch.multinomial(log_prob.exp(), 1).squeeze(1)

        if record:
            entry = (robots_in, station_in, task_in, fm, a, active)
            traj.append(entry if rows is None else tuple(_widen(x, rows, B0) for x in entry))

        # transition
        is_finish = active & (a == 0)
        is_station = active & (a >= 1) & (a <= M)
        is_task = active & (a > M)
        station_sel = a.clamp(1, M) - 1
        task_sel = (a - M - 1).clamp(0, N - 1)
        park_station = torch.where(self_at >= 0, self_at, (d_station + station_invalid.float() * INF).argmin(dim=1))
        tp = self_pos.clone()
        tp = torch.where(is_station[:, None], spos[bidx, station_sel], tp)
        tp = torch.where(is_task[:, None], tpos[bidx, task_sel], tp)
        tp = torch.where(is_finish[:, None], spos[bidx, park_station], tp)
        travel = (tp - self_pos).norm(dim=-1) / self_vel
        new_free = torch.where(is_station, now + travel + self_charge, now + travel)
        new_at = torch.where(is_station, station_sel,
                             torch.where(is_task, torch.full_like(self_at, -1),
                                         torch.where(is_finish, park_station, self_at)))
        rpos[bidx, dec] = torch.where(active[:, None], tp, rpos[bidx, dec])
        energy[bidx, dec] = torch.where(active, self_energy - travel, energy[bidx, dec])
        free_time[bidx, dec] = torch.where(active, new_free, free_time[bidx, dec])
        at_station[bidx, dec] = torch.where(active, new_at, at_station[bidx, dec])
        alive[bidx, dec] = torch.where(is_finish, torch.zeros_like(alive[bidx, dec]), alive[bidx, dec])
        completion[bidx, dec] = torch.where(is_finish, now + travel, completion[bidx, dec])
        if fast:
            claimed[bidx, task_sel] = claimed[bidx, task_sel] | is_task
        else:
            ti = is_task.nonzero(as_tuple=True)[0]
            if ti.numel() > 0:
                claimed[ti, task_sel[ti]] = True

    if fast and traj:      # drop the trailing all-finished decisions of the lazy done test
        keep = int(torch.stack([e[5].any() for e in traj]).nonzero()[-1]) + 1
        del traj[keep:]
    if full is not None:
        for k_, v_ in (("alive", alive), ("claimed", claimed), ("completion", completion)):
            full[k_].index_copy_(0, rows, v_)
        alive, claimed, completion = full["alive"], full["claimed"], full["completion"]
    complete = claimed.all(dim=1) & ~alive.any(dim=1)
    makespan = completion.max(dim=1).values
    reward = -(makespan + BETA * (~claimed).sum(dim=1).float())
    return makespan, complete, reward, traj


@torch.no_grad()
def best_of(policy, instances, K, device, seed=0, max_rows=1024):
    """Per instance, the lowest makespan among K sampled solutions (inf if none is complete); the samples
    of max_rows // K instances are drawn in one batch, as for the paper's numbers. K = 0: greedy."""
    if K == 0:
        mks, comp, _, _ = rollout(policy, collate(instances, device), "greedy")
        return mks.masked_fill(~comp, float("inf"))
    torch.manual_seed(seed)
    per = max(1, max_rows // K)
    best = []
    for s in range(0, len(instances), per):
        chunk = instances[s:s + per]
        mks, comp, _, _ = rollout(policy, collate([x for x in chunk for _ in range(K)], device), "sampling")
        best.append(mks.masked_fill(~comp, float("inf")).view(len(chunk), K).min(dim=1).values)
    return torch.cat(best)


def _widen(x, rows, width):
    """Scatter the rows of a shrunk batch back to full width (other rows zero)."""
    out = x.new_zeros((width,) + tuple(x.shape[1:]))
    return out.index_copy_(0, rows, x)


def select(traj, idx):
    """The recorded decisions of the rollouts `idx`."""
    return [tuple(x[idx] for x in entry) for entry in traj]


def replay_loss(policy, traj, weight, ent_coef):
    """Backpropagate -(1/B) sum_b weight_b sum_t log p(a_bt) - ent_coef * entropy over the recorded
    decisions, stacked into forward/backward passes of at most fastpath.REPLAY_ROWS rows (finished
    rows, whose decide-mask is 0, are left out). Returns (loss, entropy) as floats."""
    B = weight.size(0)
    total_loss = torch.zeros((), device=weight.device, dtype=torch.float64)
    total_ent = torch.zeros((), device=weight.device, dtype=torch.float64)
    counts = torch.stack([e[5].sum() for e in traj]).tolist() if traj else []
    chunks, cur, n = [], [], 0
    for i, c in enumerate(counts):
        if c == 0:
            continue
        if cur and n + c > fastpath.REPLAY_ROWS:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += c
    if cur:
        chunks.append(cur)
    for idx in chunks:
        chunk = [traj[i] for i in idx]
        sel = torch.cat([e[5] for e in chunk]).nonzero(as_tuple=True)[0]
        robots_in, station_in, task_in, fm, a = (torch.cat([e[k] for e in chunk])[sel] for k in range(5))
        w = weight.repeat(len(chunk))[sel]
        log_prob = policy(robots_in, station_in, task_in, fm)
        logp = log_prob.gather(1, a[:, None]).squeeze(1)
        ent = -(log_prob.exp() * log_prob).sum(dim=1)
        loss = (-(w * logp).sum() - ent_coef * ent.sum()) / B
        loss.backward()
        total_loss += loss.detach().double()
        total_ent += (ent.sum() / B).detach().double()
    return total_loss.item(), total_ent.item()
