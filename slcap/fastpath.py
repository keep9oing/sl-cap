"""Fast kernels for training (same mathematics as the reference kernels, different float rounding).

Off by default, so evaluation runs the reference kernels in full fp32, which reproduce the paper's
numbers. The trainers switch it on:
  * multi-head attention as one fused (B*n, D) x (D, H*d) GEMM per projection; the attention core is
    torch's scaled_dot_product_attention when both sides have >= SDPA_MIN tokens, a plain matmul for a
    single query, and a broadcast multiply-sum when one side is the handful of robots;
  * categorical sampling by the Gumbel-max form, claimed-task bookkeeping without nonzero(), the
    all-done test every DONE_CHECK_EVERY decisions, and the batch shrunk to the running rows once at
    most COMPACT_FRAC of them remain;
  * replay: recorded decisions stacked into one forward/backward of <= REPLAY_ROWS rows.
With tf32=True the training computation also runs its fp32 GEMMs on TF32 tensor cores. Validation must
not: TF32 changes 6-9 % of greedy results, so the trainers validate inside `full_fp32()`.
"""
import contextlib

import torch
import torch.nn.functional as F

ENABLED = False
DONE_CHECK_EVERY = 4
REPLAY_ROWS = 4096
SDPA_MIN = 16
COMPACT_FRAC = 0.75


def enable(flag=True, tf32=False):
    """flag: fast kernels on/off. tf32: TF32 matmuls as well (only together with the fast kernels)."""
    global ENABLED
    ENABLED = bool(flag)
    torch.backends.cuda.matmul.allow_tf32 = bool(flag and tf32)


@contextlib.contextmanager
def full_fp32():
    """Full-fp32 matmuls inside the block (validation), TF32 setting restored afterwards."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def mha(mod, q, h=None, masked=None):
    """Multi-head attention with the parameter layout of policy.MultiHeadAttention
    (w_query/w_key/w_value (H, D, d), w_out (H, d, D)).
    masked: bool, broadcastable to (B, 1, n_query, n_key), True = excluded (never a full row)."""
    self_attn = h is None
    if self_attn:
        h = q
    B, nk, D = h.shape
    nq = q.size(1)
    H, dk, dv = mod.n_heads, mod.key_dim, mod.value_dim
    wq = mod.w_query.permute(1, 0, 2).reshape(-1, H * dk)
    wk = mod.w_key.permute(1, 0, 2).reshape(D, H * dk)
    wv = mod.w_value.permute(1, 0, 2).reshape(D, H * dv)
    if self_attn:
        Q, K, V = (q.reshape(-1, D) @ torch.cat((wq, wk, wv), dim=1)).split((H * dk, H * dk, H * dv), dim=1)
    else:
        Q = q.reshape(-1, q.size(-1)) @ wq
        K, V = (h.reshape(-1, D) @ torch.cat((wk, wv), dim=1)).split((H * dk, H * dv), dim=1)
    Q = Q.view(B, nq, H, dk).transpose(1, 2)
    K = K.view(B, nk, H, dk).transpose(1, 2)
    V = V.view(B, nk, H, dv).transpose(1, 2)
    if nq >= SDPA_MIN and nk >= SDPA_MIN:
        attn_mask = None if masked is None else ~masked
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, scale=mod.norm_factor)
    else:
        if nq == 1:
            U = torch.matmul(Q, K.transpose(-1, -2))
        else:
            U = (Q.unsqueeze(-2) * K.unsqueeze(-3)).sum(-1)
        U = U * mod.norm_factor
        if masked is not None:
            U = U.masked_fill(masked, float("-inf"))
        A = torch.softmax(U, dim=-1)
        out = torch.matmul(A, V) if nq == 1 else (A.unsqueeze(-1) * V.unsqueeze(-3)).sum(-2)
    out = out.transpose(1, 2).reshape(B * nq, H * dv) @ mod.w_out.view(H * dv, -1)
    return out.view(B, nq, -1)


def sample(log_prob):
    """One categorical draw per row (Gumbel-max). Masked entries (log-prob ~ -1e8) never win."""
    e = torch.empty_like(log_prob).exponential_().clamp_min_(1e-30)
    return (log_prob - e.log()).argmax(dim=1)
