"""The charger-aware attention policy (Sec. 5 of the paper).

At every decision the deciding robot encodes the fleet and the candidate targets (park, stations,
tasks) relative to its own position and scores the candidates with a pointer decoder. The
attention blocks, the egocentric re-encoding and the decoder follow DAN (Cao, Sun & Sartoretti,
arXiv 2109.04205); the inputs carry the energy state and the charger-aware features:
    robot   [active, rel_x, rel_y, time until free, energy, capacity E, is_self, unclaimed fraction]
    station [rel_x, rel_y, energy left on arrival]
    task    [rel_x, rel_y, claimed, time to nearest station, energy left after serving and returning]
Attribute names follow the trained checkpoints (local_agent_encoder, local_target_encoder,
local_decoder).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from . import fastpath

ROBOT_DIM, STATION_DIM, TASK_DIM = 8, 3, 5


class Normalization(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.normalizer = nn.LayerNorm(dim)

    def forward(self, x):
        return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = dim
        self.value_dim = self.key_dim = dim // n_heads
        self.norm_factor = 1 / math.sqrt(self.key_dim)
        self.w_query = nn.Parameter(torch.Tensor(n_heads, dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(n_heads, dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(n_heads, dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(n_heads, self.value_dim, dim))
        for p in self.parameters():
            stdv = 1.0 / math.sqrt(p.size(-1))
            p.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """mask (B, n_key), nonzero = key excluded."""
        if fastpath.ENABLED:
            return fastpath.mha(self, q, h, None if mask is None else mask.reshape(q.size(0), 1, 1, -1).bool())
        if h is None:
            h = q
        B, nk, D = h.size()
        nq = q.size(1)
        h_flat = h.contiguous().view(-1, D)
        Q = torch.matmul(q.contiguous().view(-1, D), self.w_query).view(self.n_heads, B, nq, -1)
        K = torch.matmul(h_flat, self.w_key).view(self.n_heads, B, nk, -1)
        V = torch.matmul(h_flat, self.w_value).view(self.n_heads, B, nk, -1)
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is None:
            A = torch.softmax(U, dim=-1)
        else:
            m = mask.view(1, B, 1, nk).expand_as(U).bool()
            A = torch.softmax(U.masked_fill(m, -np.inf), dim=-1).masked_fill(m, 0)
        heads = torch.matmul(A, V)
        return torch.mm(heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
                        self.w_out.view(-1, D)).view(B, nq, D)


class SelfAttentionLayer(nn.Module):
    """Encoder layer: pre-norm self-attention and feed-forward, both residual."""

    def __init__(self, dim, ff_dim=512):
        super().__init__()
        self.layer = nn.Sequential(
            SkipConnection(nn.Sequential(Normalization(dim), MultiHeadAttention(dim))),
            SkipConnection(nn.Sequential(Normalization(dim), nn.Linear(dim, ff_dim), nn.ReLU(inplace=True),
                                         nn.Linear(ff_dim, dim))),
        )

    def forward(self, x, pad=None):
        if pad is None:
            return self.layer(x)
        norm, mha = self.layer[0].module                  # padded keys left out of the attention
        return self.layer[1](x + mha(norm(x), mask=pad))


class AgentEncoder(nn.Module):
    """One token per robot."""

    def __init__(self, dim):
        super().__init__()
        self.init_embedding = nn.Linear(ROBOT_DIM, dim)
        self.layers = nn.Sequential(SelfAttentionLayer(dim))

    def forward(self, robots):
        return self.layers(self.init_embedding(robots))


class TargetEncoder(nn.Module):
    """The candidate targets: a learned park token, the stations and the tasks."""

    def __init__(self, dim, n_layers=3):
        super().__init__()
        self.idle_embedding = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.uniform_(self.idle_embedding, 0, 1)
        self.station_init_embedding = nn.Linear(STATION_DIM, dim)
        self.task_init_embedding = nn.Linear(TASK_DIM, dim)
        self.layers = nn.Sequential(*(SelfAttentionLayer(dim) for _ in range(n_layers)))

    def forward(self, task_in, station_in, pad=None):
        idle = self.idle_embedding.expand(task_in.size(0), 1, -1)
        h = torch.cat((idle, self.station_init_embedding(station_in), self.task_init_embedding(task_in)), dim=1)
        if pad is None:
            return self.layers(h)
        for layer in self.layers:
            h = layer(h, pad=pad)
        return h


class AttentionLayer(nn.Module):
    """Decoder layer: cross-attention and feed-forward, both residual with post-norm."""

    def __init__(self, dim, ff_dim=512):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(dim)
        self.normalization1 = Normalization(dim)
        self.feedForward = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(inplace=True), nn.Linear(ff_dim, dim))
        self.normalization2 = Normalization(dim)

    def forward(self, q, h, mask=None):
        x = self.normalization1(self.multiHeadAttention(q=q, h=h, mask=mask) + q)
        return self.normalization2(self.feedForward(x) + x)


class Pointer(nn.Module):
    """Single-head compatibility with tanh clipping -> log-probabilities over the candidates."""

    def __init__(self, dim, clip):
        super().__init__()
        self.tanh_clipping = clip
        self.norm_factor = 1 / math.sqrt(dim)
        self.w_query = nn.Parameter(torch.Tensor(dim, dim))
        self.w_key = nn.Parameter(torch.Tensor(dim, dim))
        for p in self.parameters():
            stdv = 1.0 / math.sqrt(p.size(-1))
            p.data.uniform_(-stdv, stdv)

    def forward(self, q, h, mask):
        B, n, D = h.size()
        Q = torch.matmul(q.reshape(-1, D), self.w_query).view(B, q.size(1), -1)
        K = torch.matmul(h.reshape(-1, D), self.w_key).view(B, n, -1)
        U = self.tanh_clipping * torch.tanh(self.norm_factor * torch.matmul(Q, K.transpose(1, 2)))
        return torch.log_softmax(U.masked_fill(mask.view(B, 1, n).bool(), -1e8), dim=-1)


class Decoder(nn.Module):
    def __init__(self, dim, clip):
        super().__init__()
        self.target_MHA = AttentionLayer(dim)
        self.agent_MHA = AttentionLayer(dim)
        self.MHA = AttentionLayer(dim)
        self.SHA = Pointer(dim, clip)

    def forward(self, current, targets, robots, mask):
        targets = self.agent_MHA(q=targets, h=robots)           # targets in the context of the fleet
        q = self.target_MHA(q=current, h=robots)                # current state
        q = self.MHA(q=q, h=targets, mask=mask)                 # glimpse over the feasible candidates
        return self.SHA(q, targets, mask).squeeze(1)


class Policy(nn.Module):
    def __init__(self, dim=128, clip=10.0):
        super().__init__()
        self.local_agent_encoder = AgentEncoder(dim)
        self.local_target_encoder = TargetEncoder(dim)
        self.local_decoder = Decoder(dim, clip)

    def forward(self, robots, station_in, task_in, mask, pad=None):
        """-> log-probabilities (B, 1+M+N) over [park, stations, tasks]; mask and pad (B, 1+M+N),
        nonzero = infeasible / padding (padding needs the fast kernels)."""
        r = self.local_agent_encoder(robots)
        t = self.local_target_encoder(task_in, station_in, pad)
        if pad is None:
            current = t.mean(dim=1, keepdim=True)
        else:
            keep = (~pad).unsqueeze(-1).to(t.dtype)
            current = (t * keep).sum(dim=1, keepdim=True) / keep.sum(dim=1, keepdim=True)
        return self.local_decoder(current, t, r, mask)


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    policy = Policy().to(device)
    policy.load_state_dict(ck["model"])
    return policy.eval()
