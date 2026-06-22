"""A compact, self-contained PPO for GPU-resident vec envs (no rl_games / rsl_rl dep).

Everything stays on the GPU: the actor-critic, the running observation + return normalizers,
GAE, and the update. The `PPO` trainer owns the rollout+update loop so an env script is just
`PPO(env, ...).learn(N)`; the building blocks (ActorCritic, RunningNorm, compute_gae) are also
exported for hand-rolled loops.

A compact REFERENCE trainer, owned and readable on purpose — not a tuned competitor to rl_games
/ rsl_rl. It carries the robustness that actually matters for varied reward scales: running obs
AND return normalization (so the value head + its PPO clipping work whether returns are ~1 or
~1000), linear LR anneal, and target-KL early stopping.
"""
import math
import time

import numpy as np
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

    def denorm(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean

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


class PPO:
    """PPO trainer for GPU-resident vec envs. Owns the rollout + update loop so a training
    script is: build env -> PPO(env, act_dim, ...).learn(N) -> .save(path).

    Env protocol (all GPU tensors): reset() -> obs [K, obs_dim]; step(action [K, act_dim]) ->
    (obs, reward, done, terminal_obs, is_timeout). `is_timeout` flags which `done`s are
    time-limit truncations (bootstrap V(terminal)) vs true terminals (bootstrap 0); for an env
    with only time limits, return is_timeout = done. K, obs_dim and the device are inferred
    from the first reset().
    """

    def __init__(self, env, act_dim, hidden=(256, 256), *, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=5, minibatches=4, horizon=32, entropy=0.0, vfcoef=1.0,
                 log_std_init=-0.5, max_grad_norm=1.0, target_kl=0.02, anneal_lr=True,
                 normalize_returns=True, meta=None, device=None):
        self.env = env
        self.obs = env.reset()
        self.K, self.obs_dim = self.obs.shape[0], self.obs.shape[1]
        self.device = self.obs.device if device is None else torch.device(device)
        self.act_dim, self.T = act_dim, horizon
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.entropy, self.vfcoef = epochs, entropy, vfcoef
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.anneal_lr, self.lr0 = anneal_lr, lr
        self.mb = max(1, self.K * self.T // minibatches)
        self.ac = ActorCritic(self.obs_dim, act_dim, tuple(hidden), log_std_init).to(self.device)
        self.norm = RunningNorm(self.obs_dim, self.device)
        # return normalizer (clip≈off): the critic predicts NORMALIZED returns, so value
        # clipping is meaningful regardless of reward scale. denorm() back to raw for GAE.
        self.ret_norm = RunningNorm(1, self.device, clip=1e9) if normalize_returns else None
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.meta = {"obs_dim": self.obs_dim, "act_dim": act_dim, "hidden": list(hidden), **(meta or {})}

    def _value_raw(self, nobs):
        v = self.ac.critic(nobs).squeeze(-1)
        return self.ret_norm.denorm(v) if self.ret_norm else v

    def learn(self, iterations, log_every=10, on_log=None):
        dev, K, T, O, A = self.device, self.K, self.T, self.obs_dim, self.act_dim
        b_obs = torch.zeros(T, K, O, device=dev); b_term = torch.zeros(T, K, O, device=dev)
        b_act = torch.zeros(T, K, A, device=dev); b_logp = torch.zeros(T, K, device=dev)
        b_vraw = torch.zeros(T, K, device=dev); b_rew = torch.zeros(T, K, device=dev)
        b_done = torch.zeros(T, K, device=dev); b_to = torch.zeros(T, K, device=dev)
        ep_ret = torch.zeros(K, device=dev); ep_len = torch.zeros(K, device=dev)
        recent_ret, recent_len = [], []
        obs = self.obs
        t0 = time.perf_counter(); total = 0
        for it in range(1, iterations + 1):
            if self.anneal_lr:
                for g in self.opt.param_groups:
                    g["lr"] = self.lr0 * (1.0 - (it - 1) / iterations)
            for t in range(T):
                self.norm.update(obs)
                nobs = self.norm.norm(obs)
                with torch.no_grad():
                    a, logp, vnorm = self.ac.act(nobs)
                    vraw = self.ret_norm.denorm(vnorm) if self.ret_norm else vnorm
                b_obs[t] = nobs; b_act[t] = a; b_logp[t] = logp; b_vraw[t] = vraw
                obs, rew, done, term_obs, is_to = self.env.step(a)
                b_rew[t] = rew; b_done[t] = done.float(); b_term[t] = term_obs; b_to[t] = is_to.float()
                ep_ret += rew; ep_len += 1
                d = done.nonzero(as_tuple=False).squeeze(-1)
                if d.numel() > 0:
                    recent_ret.extend(ep_ret[d].tolist()); recent_len.extend(ep_len[d].tolist())
                    ep_ret[d] = 0.0; ep_len[d] = 0.0
            total += T * K

            with torch.no_grad():
                last_v = self._value_raw(self.norm.norm(obs))
                term_v = self._value_raw(self.norm.norm(b_term.reshape(-1, O))).reshape(T, K) * b_to
            adv, ret = compute_gae(b_rew, b_vraw, b_done, last_v, term_v, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            if self.ret_norm:
                self.ret_norm.update(ret.reshape(-1, 1))
                vtarg = self.ret_norm.norm(ret.reshape(-1))     # normalized return target
                vold = self.ret_norm.norm(b_vraw.reshape(-1))   # normalized old value (clip anchor)
            else:
                vtarg, vold = ret.reshape(-1), b_vraw.reshape(-1)

            f_obs = b_obs.reshape(-1, O); f_act = b_act.reshape(-1, A)
            f_logp = b_logp.reshape(-1); f_adv = adv.reshape(-1)
            n = f_obs.shape[0]
            for _ in range(self.epochs):
                idx = torch.randperm(n, device=dev)
                kls = []
                for s in range(0, n, self.mb):
                    j = idx[s:s + self.mb]
                    newlogp, ent, vnorm = self.ac.evaluate(f_obs[j], f_act[j])
                    logratio = newlogp - f_logp[j]
                    ratio = logratio.clamp(-20, 20).exp()
                    pg = -torch.min(ratio * f_adv[j],
                                    ratio.clamp(1 - self.clip, 1 + self.clip) * f_adv[j]).mean()
                    v_clip = vold[j] + (vnorm - vold[j]).clamp(-self.clip, self.clip)
                    vf = torch.max((vnorm - vtarg[j]).pow(2), (v_clip - vtarg[j]).pow(2)).mean()
                    loss = pg + self.vfcoef * vf - self.entropy * ent.mean()
                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.opt.step()
                    kls.append(((ratio - 1) - logratio).detach().mean())   # Schulman low-var KL
                # one host sync per epoch (not per minibatch) for the early-stop check
                if self.target_kl is not None and kls and torch.stack(kls).mean().item() > self.target_kl:
                    break

            if it % log_every == 0 or it == 1:
                el = time.perf_counter() - t0
                rr = float(np.mean(recent_ret[-400:])) if recent_ret else float("nan")
                rl = float(np.mean(recent_len[-400:])) if recent_len else float("nan")
                msg = (f"it {it:4d} | ep_ret {rr:8.1f} | ep_len {rl:5.0f} | "
                       f"{total / el / 1e3:6.1f}k steps/s | {el:5.1f}s")
                (on_log or print)(msg)
                recent_ret = recent_ret[-400:]; recent_len = recent_len[-400:]
        self.obs = obs
        return self.ac, self.norm, self.meta

    def save(self, path):
        save_policy(path, self.ac, self.norm, self.meta)
