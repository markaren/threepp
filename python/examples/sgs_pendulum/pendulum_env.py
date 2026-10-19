"""Torque-limited pendulum swing-up over a POOL of start configurations: the toy for
success-guided sampling (threepp.rl.sgs).

Sim-less VecTask (build_robot=None): the dynamics are two lines of torch, so thousands of
pendulums train in a minute on a laptop GPU. What makes it a test of the sampler and not of the
pendulum is the configuration pool. A cell is (start angle bin, start velocity bin, torque
limit); the pool spans starts from "already upright" to "hanging still under a torque limit a
tenth of gravity's", so at every stage of training some cells are solved, some are learnable
and some are out of reach. The reward is sparse: one point per step held upright, a small effort
penalty. Success is holding upright through the final `HOLD_S` seconds of the episode.

Angle convention: theta = 0 upright, wrapped to (-pi, pi]. Gym's pendulum constants (g = 10,
m = l = 1, |theta_dot| <= 8, torque 2) sit in the middle of the torque ladder.
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from threepp.rl import VecTask                   # noqa: E402
from threepp.rl.sgs import SuccessGuidedSampler  # noqa: E402

CONTROL_HZ = 20
SUBSTEPS = 2
EPISODE_S = 8.0
HOLD_S = 1.5                                   # upright through the last HOLD_S seconds = success
G, M, L = 10.0, 1.0, 1.0
W_MAX = 8.0
UP_COS = 0.95                                  # |theta| < ~18 deg counts as upright
UP_W = 2.0

# the configuration pool
N_TH, N_W = 24, 5
W0_MAX = 4.0
TAUS = (1.0, 1.5, 2.0, 3.0, 5.0)               # torque limits, hardest first
N_CELLS = N_TH * N_W * len(TAUS)
OBS_DIM, ACT_DIM = 4, 1


def cell_centres(device):
    """[N_CELLS, 3] (theta0, w0, tau) centre of every cell, cell = (ti * N_W + wi) * len(TAUS) + ki."""
    th = (torch.arange(N_TH, device=device) + 0.5) / N_TH * 2 * math.pi - math.pi
    w = (torch.arange(N_W, device=device) + 0.5) / N_W * 2 * W0_MAX - W0_MAX
    tau = torch.tensor(TAUS, device=device)
    grid = torch.cartesian_prod(th, w, tau)
    return grid


def cell_tau_index(cells):
    return cells % len(TAUS)


class PendulumPoolEnv(VecTask):
    control_hz = CONTROL_HZ
    episode_s = EPISODE_S
    act_dim = ACT_DIM

    def __init__(self, num_envs, device="cuda", seed=0, *, sampler=None, stratified=False, cells=None, dense=0.0, graph=False):
        """
        sampler      a SuccessGuidedSampler over N_CELLS (any mode); None = uniform draws
        stratified   evaluation layout: env i always starts in cell i % N_CELLS (ignores sampler)
        cells        a fixed [M] list of cells: env i always starts in cells[i % M] (viewer layout)
        dense        weight of gym's pendulum cost -(theta^2 + 0.1 w^2), normalised to [-1, 0], per
                     step; 0 = the sparse reward alone
        """
        super().__init__(num_envs, None, device=device, seed=seed, graph=graph)
        self.sampler = sampler
        self.stratified = stratified
        self.fixed_cells = None if cells is None else torch.as_tensor(cells, dtype=torch.long, device=self.device)
        self.dense = float(dense)
        self.centres = cell_centres(self.device)
        self.d_th = 2 * math.pi / N_TH
        self.d_w = 2 * W0_MAX / N_W
        self.theta = self.env_state(())
        self.w = self.env_state(())
        self.tau = self.env_state((), init=TAUS[0])
        self.cell = self.env_state((), init=0, dtype=torch.long)
        self.hold_up = self.env_state(())          # upright steps inside the hold window
        self.hold_steps = int(HOLD_S * CONTROL_HZ)
        self.last_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.ep_success = torch.zeros((), device=self.device)   # since the last stats read
        self.ep_count = torch.zeros((), device=self.device)

    # ---- configuration -> start state ----------------------------------------------
    def on_reset(self, idx):
        n = idx.numel()
        if self.fixed_cells is not None:
            cells = self.fixed_cells[idx % self.fixed_cells.numel()]
        elif self.stratified:
            cells = idx % N_CELLS
        elif self.sampler is not None:
            cells = self.sampler.sample(n)
        else:
            cells = torch.randint(0, N_CELLS, (n,), device=self.device, generator=self.g)
        c = self.centres[cells]
        u = torch.rand(n, 2, device=self.device, generator=self.g) - 0.5
        self.cell[idx] = cells
        self.theta[idx] = c[:, 0] + u[:, 0] * self.d_th
        self.w[idx] = c[:, 1] + u[:, 1] * self.d_w
        self.tau[idx] = c[:, 2]

    # ---- dynamics ----------------------------------------------------------------------
    def simulate(self, a):
        u = a[:, 0] * self.tau
        h = self.dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            acc = (G / L) * torch.sin(self.theta) + u / (M * L * L)
            self.w.copy_((self.w + h * acc).clamp(-W_MAX, W_MAX))   # in place: registered buffers
            self.theta.add_(h * self.w)
        self.theta.copy_(torch.remainder(self.theta + math.pi, 2 * math.pi) - math.pi)

    def _upright(self):
        return (torch.cos(self.theta) > UP_COS) & (self.w.abs() < UP_W)

    def on_step(self, s):
        in_hold = self.steps >= (self.max_steps - self.hold_steps)
        self.hold_up += (self._upright() & in_hold).float()

    def observe(self, s):
        return torch.stack([torch.cos(self.theta), torch.sin(self.theta), self.w / W_MAX,
                            self.tau / TAUS[-1]], dim=-1)

    def reward_terms(self, s, a):
        terms = {"up": self._upright().float(), "effort": -0.001 * (a[:, 0] * self.tau) ** 2}
        if self.dense:
            terms["shape"] = -self.dense * (self.theta ** 2 + 0.1 * self.w ** 2) / (math.pi ** 2 + 0.1 * W_MAX ** 2)
        return terms

    def on_done(self, idx):
        succ = self.hold_up[idx] >= 0.9 * self.hold_steps
        self.last_success[idx] = succ
        self.ep_success += succ.float().sum()
        self.ep_count += idx.numel()
        if self.sampler is not None and not self.stratified and self.fixed_cells is None:
            self.sampler.record(self.cell[idx], succ)

    def success_since_last(self):
        n = self.ep_count.item()
        r = self.ep_success.item() / n if n else float("nan")
        self.ep_success.zero_(); self.ep_count.zero_()
        return r

    def config(self):
        return {**super().config(), "n_cells": N_CELLS, "taus": list(TAUS), "obs_dim": OBS_DIM}


# ---- the fixed evaluation: every cell, deterministic policy -----------------------------
@torch.no_grad()
def evaluate(ac, norm, device, reps=8, seed=12345):
    """Success per cell under the mean action, [N_CELLS], from `reps` episodes each; the
    paper's metric is the mean over the uniform configuration distribution."""
    env = PendulumPoolEnv(N_CELLS * reps, device=device, seed=seed, stratified=True)
    obs = env.reset()
    hits = torch.zeros(N_CELLS, device=device)
    for _ in range(env.max_steps):
        nobs = obs if norm is None else norm.norm(obs)
        obs, _, done, _, _ = env.step(ac.actor(nobs))
    hits.index_add_(0, env.cell, env.last_success.float())   # cells are fixed, so cell[] still maps
    return hits / reps


def summarize(per_cell):
    """Overall success and success per torque limit (hardest first)."""
    by_tau = [per_cell[cell_tau_index(torch.arange(N_CELLS, device=per_cell.device)) == k].mean().item()
              for k in range(len(TAUS))]
    return per_cell.mean().item(), by_tau


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = SuccessGuidedSampler(N_CELLS, dev, generator=torch.Generator(device=dev).manual_seed(0))
    env = PendulumPoolEnv(512, device=dev, sampler=s)
    obs = env.reset()
    for _ in range(env.max_steps + 5):
        obs, rew, done, term, to = env.step(torch.rand(env.K, 1, device=dev) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    assert bool((to == done).all())
    print("cells recorded", int((s.count > 0).sum()), "/", N_CELLS,
          "random-policy success", f"{env.success_since_last():.3f}")
    print("PENDULUM POOL ENV SELFTEST: PASS")
