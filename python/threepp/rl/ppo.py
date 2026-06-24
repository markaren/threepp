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


def _make_cnn(image_shape, feat=256):
    """Small conv trunk for [N,C,H,W] obs -> [N, feat]. C may be channels*frame_stack."""
    c, h, w = image_shape
    body = nn.Sequential(
        nn.Conv2d(c, 16, 5, stride=2, padding=2), nn.ELU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        flat = body(torch.zeros(1, c, h, w)).shape[1]
    return nn.Sequential(body, nn.Linear(flat, feat), nn.ELU()), feat


class ActorCritic(nn.Module):
    """Gaussian-policy actor-critic. Log-prob / entropy are computed by hand (no
    torch.distributions) — leaner, and it sidesteps a distribution-backward CUDA
    quirk seen on torch 2.11+cu128 with an expanded (0-stride) scale.

    State obs ([N, obs_dim]) feed the MLP heads directly. Image obs ([N, C, H, W]) pass
    `image_shape=(C,H,W)` to grow a shared CNN trunk feeding the same heads; uint8 pixels
    are cast and scaled to [0,1] in-net (so no RunningNorm for images)."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_init=-1.0, image_shape=None):
        super().__init__()
        if image_shape is not None:
            self.cnn, feat = _make_cnn(tuple(image_shape))
        else:
            self.cnn, feat = None, obs_dim
        self.actor = _mlp([feat, *hidden, act_dim])
        self.critic = _mlp([feat, *hidden, 1])
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _feat(self, obs):
        if self.cnn is None:
            return obs                      # MLP path: byte-identical to the no-CNN model
        return self.cnn(obs.float() / 255.0)  # [N,C,H,W] uint8/float pixels -> [N, feat]

    def _logprob(self, mean, act):
        # sum over action dims of the diagonal-Gaussian log density
        var = torch.exp(2.0 * self.log_std)
        return (-0.5 * ((act - mean) ** 2) / var - self.log_std - 0.5 * _LOG2PI).sum(-1)

    def _entropy(self):
        # diagonal-Gaussian differential entropy, summed over dims (state-independent)
        return (self.log_std + 0.5 * (_LOG2PI + 1.0)).sum()

    @torch.no_grad()
    def act(self, obs):
        f = self._feat(obs)
        mean = self.actor(f)
        a = mean + torch.exp(self.log_std) * torch.randn_like(mean)
        return a, self._logprob(mean, a), self.critic(f).squeeze(-1)

    @torch.no_grad()
    def act_mean(self, obs):
        return self.actor(self._feat(obs))  # deterministic action for eval/deployment

    @torch.no_grad()
    def value(self, obs):
        return self.critic(self._feat(obs)).squeeze(-1)

    def evaluate(self, obs, act):
        f = self._feat(obs)
        mean = self.actor(f)
        return self._logprob(mean, act), self._entropy(), self.critic(f).squeeze(-1)


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
    torch.save({"model": ac.state_dict(), "norm": norm.state() if norm is not None else None,
                "meta": meta}, path)


def load_policy(path, device):
    # weights_only=True: the checkpoint is only tensors + a JSON-able meta dict, so refuse to
    # run arbitrary unpickle code (don't trust a .pt to be safe just because we wrote it).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    ac = ActorCritic(meta.get("obs_dim", 0), meta["act_dim"],
                     hidden=tuple(meta.get("hidden", (256, 256))),
                     image_shape=meta.get("image_shape")).to(device)
    ac.load_state_dict(ckpt["model"])
    ac.eval()
    norm = None   # image policies normalize in-net (no obs RunningNorm)
    if ckpt.get("norm") is not None:
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
                 normalize_returns=True, normalize_obs=True, meta=None, device=None, aux_loss=None):
        self.env = env
        # aux_loss(ac, obs_minibatch) -> scalar, added to the PPO loss each minibatch (None = off).
        # General hook for extra objectives: symmetry augmentation, a BC/KL anchor to a reference policy, etc.
        self.aux_loss = aux_loss
        self.obs = env.reset()
        self.is_image = self.obs.ndim == 4                          # [K,C,H,W] image vs [K,obs_dim]
        self.K = self.obs.shape[0]
        self.image_shape = tuple(self.obs.shape[1:]) if self.is_image else None
        self.obs_dim = 0 if self.is_image else self.obs.shape[1]
        self.device = self.obs.device if device is None else torch.device(device)
        self.act_dim, self.T = act_dim, horizon
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.entropy, self.vfcoef = epochs, entropy, vfcoef
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.anneal_lr, self.lr0 = anneal_lr, lr
        self.mb = max(1, self.K * self.T // minibatches)
        self.ac = ActorCritic(self.obs_dim, act_dim, tuple(hidden), log_std_init,
                              image_shape=self.image_shape).to(self.device)
        # image obs are normalized in-net (uint8/255) -> no obs RunningNorm. normalize_obs=False also
        # disables it for state obs — needed when WARM-STARTING a policy trained on raw (un-normalized)
        # obs (e.g. an Isaac/rsl_rl actor with an Identity normalizer): a RunningNorm would shift the obs
        # out from under the loaded first-layer weights.
        self.norm = RunningNorm(self.obs_dim, self.device) if (normalize_obs and not self.is_image) else None
        # return normalizer (clip≈off): the critic predicts NORMALIZED returns, so value
        # clipping is meaningful regardless of reward scale. denorm() back to raw for GAE.
        self.ret_norm = RunningNorm(1, self.device, clip=1e9) if normalize_returns else None
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.meta = {"obs_dim": self.obs_dim, "act_dim": act_dim, "hidden": list(hidden),
                     "image_shape": list(self.image_shape) if self.image_shape else None, **(meta or {})}

    def _normobs(self, obs):
        return obs if self.norm is None else self.norm.norm(obs)

    def _value_raw(self, nobs):
        v = self.ac.value(nobs)
        return self.ret_norm.denorm(v) if self.ret_norm else v

    def learn(self, iterations, log_every=10, on_log=None):
        dev, K, T, A = self.device, self.K, self.T, self.act_dim
        oshape = self.image_shape if self.is_image else (self.obs_dim,)
        odtype = torch.uint8 if self.is_image else torch.float32      # uint8 pixels save VRAM
        b_obs = torch.zeros(T, K, *oshape, dtype=odtype, device=dev)
        b_term = torch.zeros(T, K, *oshape, dtype=odtype, device=dev)
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
                if self.norm is not None:
                    self.norm.update(obs)
                nobs = self._normobs(obs)
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
                last_v = self._value_raw(self._normobs(obs))
                term_v = self._value_raw(self._normobs(b_term.reshape(-1, *oshape))).reshape(T, K) * b_to
            adv, ret = compute_gae(b_rew, b_vraw, b_done, last_v, term_v, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            if self.ret_norm:
                self.ret_norm.update(ret.reshape(-1, 1))
                vtarg = self.ret_norm.norm(ret.reshape(-1))     # normalized return target
                vold = self.ret_norm.norm(b_vraw.reshape(-1))   # normalized old value (clip anchor)
            else:
                vtarg, vold = ret.reshape(-1), b_vraw.reshape(-1)

            f_obs = b_obs.reshape(-1, *oshape); f_act = b_act.reshape(-1, A)
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
                    if self.aux_loss is not None:                       # symmetry / BC anchor / etc.
                        loss = loss + self.aux_loss(self.ac, f_obs[j])
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
