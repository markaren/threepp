"""spotv2 — UNEVEN (rough) terrain instead of discrete steps.

Same velocity-tracking + steering-preservation machinery as spot_terrain_env (warm-started Isaac
walker, 58-d obs with zero-init terrain cols, exp-kernel command tracking, scan-gated imitation
anchor), but the terrain is GENTLE ROLLING BUMPS from smooth low-amplitude noise rather than tents
/ stairs. Discrete steps are the hard case (the flat walker stalls on a tall riser); continuous
uneven ground is what the warm-started gait can actually negotiate, so it should learn far faster.

Terrain: per-lane strips of full-width boxes whose tops follow smooth multi-octave sine noise
(amplitude graded across lanes = a built-in difficulty sweep; FLAT_FRAC lanes are flat for
full-steering replay). Humps rise above the flat ground (troughs sit at ground level). The obs scan
+ reward read the EXACT box-top heights (a precomputed [K,N] table), so terrain ground-truth is
exact and needs no raycast. Everything else (commands, reward, warm-start, obs layout) is identical
to spot_terrain_env, so train_spot_rough.py / play_spot_rough.py reuse the same trainer + viewer.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))

import threepp as tp
from threepp.rl import GpuSim
from spot_deploy import default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, fetch_assets
# reuse the shared math helpers, robot factory, ground, obs/policy + tuning constants
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, SpotGpu, _flat_ground,
                              CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z, PROBE_DX, OBS_DIM, ACT_DIM,
                              HIDDEN, HALF_W, FLAT_FRAC, VX_LO, VX_HI, VY_HI, WZ_HI, STAND_PROB,
                              FWD_DRIVE_FRAC, CMD_MIN, CMD_MAX, SIG, W_IMIT)

# --------------------------------------------------------------------------- #
#  Rough terrain — smooth multi-octave noise humps (box-built, exact ground-truth)
# --------------------------------------------------------------------------- #
ROUGH_X0 = -1.5             # bump field starts behind the spawn
ROUGH_LEN = 11.0           # bump field length (m)
ROUGH_RUN = 0.30           # box run along x (m)
N_BOXES = int(round(ROUGH_LEN / ROUGH_RUN))      # ~37 boxes/lane
AMP_MIN, AMP_MAX = 0.0, 0.08                      # per-lane bump amplitude (graded curriculum)
FREQS = (0.30, 0.65, 1.05)                        # cycles/m — LOW so the terrain is gentle/smooth
WEIGHTS = (0.55, 0.30, 0.15)
ROUGH_EPISODE_S = 14.0

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "terrain": "rough", "rough_len": ROUGH_LEN, "rough_run": ROUGH_RUN, "n_boxes": N_BOXES,
          "amp_min": AMP_MIN, "amp_max": AMP_MAX, "episode_s": ROUGH_EPISODE_S,
          "probe_dx": list(PROBE_DX), "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": list(HIDDEN),
          "vx": [VX_LO, VX_HI], "vy_hi": VY_HI, "wz_hi": WZ_HI, "stand_prob": STAND_PROB,
          "sig": SIG, "w_imit": W_IMIT}


def _box_centers():
    return ROUGH_X0 + (np.arange(N_BOXES) + 0.5) * ROUGH_RUN     # [N] world-x of each box center


def make_rough_heights(k, seed=0, amp_max=AMP_MAX, flat_frac=FLAT_FRAC):
    """Per-lane box-top heights [K,N] from smooth multi-octave sine noise. Amplitude graded
    AMP_MIN..amp_max across lanes (the difficulty sweep); flat_frac of lanes set flat (amp=0) for
    full-steering replay. Heights are in [0, 2*amp] (humps rising above the flat ground)."""
    rng = np.random.default_rng(seed)
    amp = np.linspace(AMP_MIN, amp_max, k).astype(np.float32)
    amp[rng.random(k) < flat_frac] = 0.0
    xs = _box_centers()
    ph = rng.uniform(0.0, 2.0 * np.pi, size=(k, len(FREQS))).astype(np.float32)
    smooth = np.zeros((k, N_BOXES), np.float32)
    for fi, (f, w) in enumerate(zip(FREQS, WEIGHTS)):
        smooth += w * np.sin(2.0 * np.pi * f * xs[None, :] + ph[:, fi:fi + 1])
    heights = (amp[:, None] * (1.0 + np.clip(smooth, -1.0, 1.0))).astype(np.float32)
    return heights, amp


def rough_profile(amp=0.06, seed=0):
    """A single lane's box-top heights [N] (for the viewer). Same noise as make_rough_heights."""
    rng = np.random.default_rng(seed)
    xs = _box_centers()
    ph = rng.uniform(0.0, 2.0 * np.pi, size=len(FREQS))
    smooth = sum(w * np.sin(2.0 * np.pi * f * xs + ph[i]) for i, (f, w) in enumerate(zip(FREQS, WEIGHTS)))
    return (amp * (1.0 + np.clip(smooth, -1.0, 1.0))).astype(np.float32)


def _add_rough(world, k, spacing, heights, run=ROUGH_RUN, x0=ROUGH_X0, w_frac=0.92):
    """Per-lane full-width box strip; box j has height heights[i,j] (skip ~0 troughs -> flat ground)."""
    w = spacing * w_frac
    n = heights.shape[1]
    for i in range(k):
        for j in range(n):
            h = float(heights[i, j])
            if h < 0.006:                                   # trough -> no box (robot rests on the ground)
                continue
            b = tp.Mesh(tp.BoxGeometry(run, w, h), tp.MeshStandardMaterial())
            b.position.set(x0 + j * run + run * 0.5, i * spacing, h * 0.5)
            world.add_static(b)


class SpotRoughEnv:
    def __init__(self, num_envs=2048, device="cuda", seed=0, amp_max=AMP_MAX, flat_only=False):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(ROUGH_EPISODE_S * CONTROL_HZ)
        heights_np, amp_np = make_rough_heights(num_envs, seed=seed,
                                                amp_max=(0.0 if flat_only else amp_max),
                                                flat_frac=(1.0 if flat_only else FLAT_FRAC))
        self.sim = GpuSim(num_envs, lambda world, i: SpotGpu(world, i, SPACING),
                          gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, read_root=True,
                          build_world=lambda world: (_flat_ground(world, num_envs, SPACING),
                                                      _add_rough(world, num_envs, SPACING, heights_np)))
        dev = self.sim.device
        self.default_q = torch.from_numpy(default_q).to(dev)                  # [12] isaac order
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)
        self.stand_q_add = self.default_q[self.a2i].expand(num_envs, -1).contiguous()
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        self.heights = torch.from_numpy(heights_np).to(dev)                  # [K,N] box-top heights
        self.amp = torch.from_numpy(amp_np).to(dev)                          # [K] per-lane amplitude
        self.is_rough = self.amp > 0.005                                     # rough lane vs flat lane
        self.probe = torch.tensor(PROBE_DX, device=dev)
        self.imit_policy = torch.jit.load(os.path.join(fetch_assets(), "spot_policy.pt"),
                                          map_location=dev).eval()
        self.lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev); pos[:, 1] = self.lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.last_act = z(num_envs, ACT_DIM)
        self._last_obs = z(num_envs, OBS_DIM)
        self.up = z(num_envs)
        self.cmd = z(num_envs, 3)
        self.cmd_timer = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.ep_start_x = z(num_envs); self.ep_max_climb = z(num_envs)
        self._resample_cmd(torch.arange(num_envs, device=dev))
        self.last_track = 0.0; self.last_flat_track = 0.0; self.last_climb = 0.0; self.last_fell = 0.0

    def _terrain_h(self, x, y):
        """Exact box-top height at (x,y), gathered from the heights table. x,y [K] or [K,P]."""
        idx = torch.clamp(torch.floor((x - ROUGH_X0) / ROUGH_RUN).long(), 0, N_BOXES - 1)
        if x.dim() == 2:
            h = self.heights.gather(1, idx)
            lane = self.lane_y[:, None]
        else:
            h = self.heights.gather(1, idx.unsqueeze(1)).squeeze(1)
            lane = self.lane_y
        on = (torch.abs(y - lane) < HALF_W).float()
        return h * on

    def _resample_cmd(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        vx = torch.empty(n, device=dev).uniform_(VX_LO, VX_HI)
        vy = torch.empty(n, device=dev).uniform_(-VY_HI, VY_HI)
        wz = torch.empty(n, device=dev).uniform_(-WZ_HI, WZ_HI)
        drive = (torch.rand(n, device=dev) < FWD_DRIVE_FRAC) & self.is_rough[idx]   # rough lanes drive fwd over bumps
        vx = torch.where(drive, torch.empty(n, device=dev).uniform_(0.4, VX_HI), vx)
        vy = torch.where(drive, vy * 0.2, vy)
        wz = torch.where(drive, wz * 0.2, wz)
        cmd = torch.stack([vx, vy, wz], dim=1)
        cmd[torch.rand(n, device=dev) < STAND_PROB] = 0.0
        self.cmd[idx] = cmd
        self.cmd_timer[idx] = torch.randint(CMD_MIN, CMD_MAX + 1, (n,), device=dev)

    def _obs(self):
        s = self.sim
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, s.root_linvel)
        ang_b = quat_rotate_inverse(q, s.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos = s.joint_pos[:, self.i2a] - self.default_q
        jv_isaac = s.joint_vel[:, self.i2a]
        x, y, zz = s.root_position[:, 0], s.root_position[:, 1], s.root_position[:, 2]
        cyaw, syaw = heading_cossin(q)
        h_here = self._terrain_h(x, y)
        px = x[:, None] + self.probe[None, :] * cyaw[:, None]
        py = y[:, None] + self.probe[None, :] * syaw[:, None]
        ahead = (self._terrain_h(px, py) - h_here[:, None]).clamp(-1.0, 1.0)
        base_above = (zz - h_here).unsqueeze(-1)
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act,
                         base_above, ahead], dim=1)
        self._last_obs = obs
        return obs

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        pose = self.base_pose[idx].clone()                                   # [n,7] [quat(4), pos(3)]
        sx = torch.rand(n, device=dev) * 3.0                                 # spawn ON the bump field, x in [0,3]
        # spawn above the HIGHEST box under the footprint (front/back feet) -> drop-settle, no spawn jolt
        fdx = torch.tensor([0.30, -0.30, 0.0], device=dev)
        sz = self._spawn_terrain(idx, sx[:, None] + fdx[None, :]).max(dim=1).values + SPAWN_Z + 0.03
        pose[:, 4] = sx                                                      # x = index 4
        pose[:, 6] = sz                                                      # z = index 6
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, self.stand_q_add[idx], torch.zeros(n, self.sim.dof, device=dev))
        self.steps[idx] = 0; self.last_act[idx] = 0.0
        self.ep_start_x[idx] = sx; self.ep_max_climb[idx] = 0.0
        self._resample_cmd(idx)

    def _spawn_terrain(self, idx, x):
        """Box-top height under the subset `idx` at world-x (full-width boxes -> no y dependence). x [n] or [n,P]."""
        j = torch.clamp(torch.floor((x - ROUGH_X0) / ROUGH_RUN).long(), 0, N_BOXES - 1)
        if x.dim() == 2:
            return self.heights[idx].gather(1, j)
        return self.heights[idx].gather(1, j.unsqueeze(1)).squeeze(1)

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        self.sim.read()
        for _ in range(20):
            self.sim.apply_drive_target(self.stand_q_add)
            self.sim.substep(DT / SUBSTEPS, SUBSTEPS)        # advance n substeps, read once
        self.last_act.zero_()
        self.up = up_z(self.sim.root_quat)
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        a = action                                                          # NO clamp (isaac action ~+-8)
        prev_a = self.last_act
        targets_isaac = self.default_q + ACTION_SCALE * a
        self.sim.apply_drive_target(targets_isaac[:, self.a2i])
        self.sim.substep(DT / SUBSTEPS, SUBSTEPS)                            # advance n substeps, read once
        self.steps += 1
        self.last_act = a
        self.cmd_timer -= 1
        self._resample_cmd(torch.nonzero(self.cmd_timer <= 0, as_tuple=False).squeeze(-1))

        q = self.sim.root_quat
        self.up = up_z(q)
        x, y, zz = self.sim.root_position[:, 0], self.sim.root_position[:, 1], self.sim.root_position[:, 2]
        roll = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        h_here = self._terrain_h(x, y)
        base_above = zz - h_here
        cyaw, syaw = heading_cossin(q)
        px = x[:, None] + self.probe[None, :] * cyaw[:, None]
        py = y[:, None] + self.probe[None, :] * syaw[:, None]
        ahead = self._terrain_h(px, py) - h_here[:, None]
        change = ahead.abs().max(dim=1).values
        w_imit = (1.0 - change / 0.10).clamp(0.0, 1.0)
        isaac_a = self.imit_policy(self._last_obs[:, :48])
        imit = w_imit * (a - isaac_a).pow(2).mean(dim=1)
        arate = a - prev_a
        fell = (self.up < 0.35) | (base_above < 0.18)

        e_lin = (self.cmd[:, 0] - lin_b[:, 0]).pow(2) + (self.cmd[:, 1] - lin_b[:, 1]).pow(2)
        e_ang = (self.cmd[:, 2] - ang_b[:, 2]).pow(2)
        track_lin = torch.exp(-e_lin / SIG)
        track_ang = torch.exp(-e_ang / SIG)
        rew = (3.0 * track_lin
               + 1.5 * track_ang
               + 0.05
               - 1.0 * roll.pow(2)
               - 0.1 * lin_b[:, 2].pow(2)
               - 0.05 * (ang_b[:, 0].pow(2) + ang_b[:, 2].pow(2))
               - 3.0 * torch.relu(0.30 - base_above)
               - 0.001 * arate.pow(2).mean(dim=1)
               - W_IMIT * imit
               - 5.0 * fell.float())

        self.ep_max_climb = torch.maximum(self.ep_max_climb, (x - self.ep_start_x).clamp_min(0.0))  # forward progress
        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self.last_climb = self.ep_max_climb[d].mean().item()            # mean forward distance covered
            self._reset_idx(d)
            self.sim.read()
            self.up = up_z(self.sim.root_quat)
            obs = self._obs()
        else:
            obs = term_obs
        self.last_fell = fell.float().mean().item()
        self.last_track = (track_lin + track_ang).mean().item()
        flat = ~self.is_rough
        self.last_flat_track = ((track_lin + track_ang)[flat]).mean().item() if bool(flat.any()) else float("nan")
        return obs, rew, done, term_obs, timeout

    @torch.no_grad()
    def measure_tracking(self, act_fn, cmd, steps=160, warm=60):
        """Mean tracking error ||lin_b_xy - cmd_xy|| + |ang_b_z - wz| under a FIXED command (for the
        held-out flat-steering regression eval; run on a flat_only env)."""
        dev = self.sim.device
        c = torch.tensor(cmd, device=dev, dtype=torch.float32).expand(self.K, 3).contiguous()
        obs = self.reset()
        errs = []
        for t in range(steps):
            self.cmd = c
            self.cmd_timer.fill_(10 ** 9)
            obs, _, _, _, _ = self.step(act_fn(obs))
            if t >= warm:
                q = self.sim.root_quat
                lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
                ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
                e = (lin_b[:, :2] - c[:, :2]).norm(dim=1).mean() + (ang_b[:, 2] - c[:, 2]).abs().mean()
                errs.append(e.item())
        return sum(errs) / max(1, len(errs))


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    K = int(os.environ.get("K", "64"))
    env = SpotRoughEnv(num_envs=K)
    obs = env.reset()
    print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}) finite={bool(torch.isfinite(obs).all())}  "
          f"N_BOXES={N_BOXES} amp_max={AMP_MAX}")
    for _ in range(200):
        obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.sim.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action (stand): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"dist={env.last_climb:.3f}  fell/step={env.last_fell:.3f}  rew={rew.mean().item():+.3f}")
    print("SPOTV2-ROUGH ENV SELFTEST: PASS")
