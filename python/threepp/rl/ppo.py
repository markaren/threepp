"""A compact, self-contained PPO for GPU-resident vec envs (no rl_games / rsl_rl dep).

Everything stays on the GPU: the actor-critic, the running observation normalizer, and
GAE. Designed to pair with GpuSim-based envs, where the env hands back obs/reward/done as
cuda tensors — so a whole PPO iteration never touches the CPU.

Owned on purpose: ~150 lines you can read and tune, vs pulling the Isaac training stack.
"""
import math

import torch
import torch.nn as nn

_LOG2PI = math.log(2.0 * math.pi)


class RunningNorm:
    """Welford running mean/var observation normalizer, on the GPU."""

    def __init__(self, dim, device, clip=10.0, eps=1e-4):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = eps
        self.clip = clip

    @torch.no_grad()
    def update(self, x):  # x: [N, dim]
        bn = x.shape[0]
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
        delta = bmean - self.mean
        tot = self.count + bn
        self.mean += delta * bn / tot
        m_a = self.var * self.count
        m_b = bvar * bn
        m2 = m_a + m_b + delta * delta * self.count * bn / tot
        self.var = m2 / tot
        self.count = tot

    def norm(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "clip": self.clip}

    def load(self, s):
        # clone so we don't alias the (discarded) checkpoint dict; keep our device
        self.mean = s["mean"].clone().to(self.mean.device)
        self.var = s["var"].clone().to(self.var.device)
        self.count = float(s["count"]); self.clip = float(s["clip"])


def _mlp(sizes, act=nn.ELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian-policy actor-critic. Log-prob / entropy are computed by hand (no
    torch.distributions) — leaner, and it sidesteps a distribution-backward CUDA
    quirk seen on torch 2.11+cu128 with an expanded (0-stride) scale."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_init=-1.0):
        super().__init__()
        self.actor = _mlp([obs_dim, *hidden, act_dim])
        self.critic = _mlp([obs_dim, *hidden, 1])
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _logprob(self, mean, act):
        # sum over action dims of the diagonal-Gaussian log density
        var = torch.exp(2.0 * self.log_std)
        return (-0.5 * ((act - mean) ** 2) / var - self.log_std - 0.5 * _LOG2PI).sum(-1)

    def _entropy(self):
        # diagonal-Gaussian differential entropy, summed over dims (state-independent)
        return (self.log_std + 0.5 * (_LOG2PI + 1.0)).sum()

    @torch.no_grad()
    def act(self, obs):
        mean = self.actor(obs)
        a = mean + torch.exp(self.log_std) * torch.randn_like(mean)
        return a, self._logprob(mean, a), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act_mean(self, obs):
        return self.actor(obs)  # deterministic action for eval/deployment

    def evaluate(self, obs, act):
        mean = self.actor(obs)
        return self._logprob(mean, act), self._entropy(), self.critic(obs).squeeze(-1)


def compute_gae(rew, val, done, last_val, term_val, gamma=0.99, lam=0.95):
    """rew/val/done/term_val: [T, K]; last_val: [K]. `done` here is a TRUNCATION (time limit):
    bootstrap the TD target with term_val (the value of the terminal state, before reset) rather
    than zeroing it, and reset the GAE chain at the boundary. (For a true failure terminal, pass
    term_val=0 at those steps.)"""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    lastgae = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nextval = last_val if t == T - 1 else val[t + 1]
        d = done[t].float()
        bootstrap = d * term_val[t] + (1.0 - d) * nextval   # terminal value on done, else in-traj
        delta = rew[t] + gamma * bootstrap - val[t]
        lastgae = delta + gamma * lam * (1.0 - d) * lastgae  # don't propagate GAE across resets
        adv[t] = lastgae
    return adv, adv + val


def save_policy(path, ac, norm, meta):
    torch.save({"model": ac.state_dict(), "norm": norm.state(), "meta": meta}, path)


def load_policy(path, device):
    # weights_only=True: the checkpoint is only tensors + a JSON-able meta dict, so refuse to
    # run arbitrary unpickle code (don't trust a .pt to be safe just because we wrote it).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    ac = ActorCritic(meta["obs_dim"], meta["act_dim"],
                     hidden=tuple(meta.get("hidden", (256, 256)))).to(device)
    ac.load_state_dict(ckpt["model"])
    ac.eval()
    norm = RunningNorm(meta["obs_dim"], device)
    norm.load(ckpt["norm"])
    return ac, norm, meta
