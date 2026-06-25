"""spotv2 — SMOOTH heightfield terrain (true 2-D rough, no box steps).

The box rough env (`spot_rough_env`) approximates bumps with full-width boxes; past ~0.20 m amplitude
those box-to-box jumps turn back into steps (the hard case). This env builds a real CONTINUOUS
triangle-mesh heightfield instead — smooth in BOTH x and y — so it stays smooth at higher amplitude.

NO C++ changes: the heightfield is a Python triangle-soup (`BufferGeometry.set_from_points`) handed to
`world.add_static_trimesh` (which auto-indexes a non-indexed soup). For scale, a small set of distinct
smooth-noise TILES is built once (shape diversity) and shared across lanes; each lane references one
tile with `Mesh.scale.z = amp[lane]`, so amplitude is graded across lanes (the difficulty curriculum)
from one geometry per shape. Tiles are edge-tapered to z~0 so they blend into the surrounding flat
ground. The obs scan + reward read the EXACT same height grids by bilinear interpolation -> ground-truth
matches the collision mesh. Same warm-start + velocity-tracking + imitation-anchor machinery as the rest
of spotv2 (identical 58-d obs), so train_spot_heightfield.py / play_spot_heightfield.py reuse it.
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
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, SpotGpu, _flat_ground,
                              scan_offsets, scan_xy,
                              CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z, PROBE_DX, OBS_DIM, ACT_DIM,
                              HIDDEN, HALF_W, FLAT_FRAC, VX_LO, VX_HI, VY_HI, WZ_HI, STAND_PROB,
                              FWD_DRIVE_FRAC, CMD_MIN, CMD_MAX, SIG, W_IMIT)

# Heightfield-specific anti-forgetting overrides (stronger than the shared box defaults): the smooth
# 0.18 m terrain + longer runs drifted STRAFE tracking (held-out eval caught a ~2x regression the
# averaged-flat gate missed), so anchor HARDER to the Isaac teacher on flat and replay MORE flat lanes.
W_IMIT = 0.25
FLAT_FRAC = 0.35

# --------------------------------------------------------------------------- #
#  Heightfield tile geometry
# --------------------------------------------------------------------------- #
HF_X0, HF_X1 = -2.0, 10.0          # tile x-span (the robot's forward roam)
HF_NX, HF_NY = 49, 15              # grid resolution (~0.25 m x, ~0.20 m y across the lane width)
NUM_SHAPES = 16                    # distinct smooth-noise tiles (shape diversity) shared across lanes
HF_AMP_MIN, HF_AMP_MAX = 0.0, 0.18 # per-lane amplitude (graded curriculum); heightfield stays SMOOTH high
HF_OCTAVES = [(1.5, 1.0, 0.5), (3.0, 2.0, 0.3), (5.0, 3.5, 0.2)]   # (x-cycles, y-cycles, weight) across the tile
HF_EPISODE_S = 14.0
FOOT_DX = (0.30, 0.30, -0.30, -0.30)   # stance foot offsets from base (front/back) for spawn-clearance sampling
FOOT_DY = (0.17, -0.17, 0.17, -0.17)   # (left/right)

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "terrain": "heightfield", "hf_x": [HF_X0, HF_X1], "hf_grid": [HF_NX, HF_NY],
          "num_shapes": NUM_SHAPES, "amp_min": HF_AMP_MIN, "amp_max": HF_AMP_MAX,
          "episode_s": HF_EPISODE_S, "probe_dx": list(PROBE_DX), "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
          "hidden": list(HIDDEN), "vx": [VX_LO, VX_HI], "vy_hi": VY_HI, "wz_hi": WZ_HI,
          "stand_prob": STAND_PROB, "sig": SIG, "w_imit": W_IMIT}


def _taper(n, frac=0.15):
    """Raised-cosine edge window (1 interior, 0 at the edges) so tiles meet the flat ground at z~0."""
    w = np.ones(n, np.float32)
    k = max(1, int(n * frac))
    ramp = (0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, k)))).astype(np.float32)
    w[:k] = ramp; w[-k:] = ramp[::-1]
    return w


def make_hf_grids(num_shapes=NUM_SHAPES, seed=0):
    """`num_shapes` smooth 2-D noise height grids H[s, nx, ny] in [0,1], edge-tapered. xs/ys are the
    world-local grid coordinates (y is lane-local: -HALF_W..HALF_W)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(HF_X0, HF_X1, HF_NX).astype(np.float32)
    ys = np.linspace(-HALF_W, HALF_W, HF_NY).astype(np.float32)
    u = (xs - HF_X0) / (HF_X1 - HF_X0)
    v = (ys + HALF_W) / (2.0 * HALF_W)
    win = _taper(HF_NX)[:, None] * _taper(HF_NY)[None, :]
    H = np.zeros((num_shapes, HF_NX, HF_NY), np.float32)
    for s in range(num_shapes):
        h = np.zeros((HF_NX, HF_NY), np.float32)
        for fx, fy, w in HF_OCTAVES:
            px, py = rng.uniform(0.0, 2.0 * np.pi, 2)
            h += w * np.sin(2.0 * np.pi * fx * u[:, None] + px) * np.sin(2.0 * np.pi * fy * v[None, :] + py)
        h = (h - h.min()) / (h.max() - h.min() + 1e-9)               # [0,1]
        H[s] = h * win                                                # taper edges to 0
    return H, xs, ys


def build_hf_geom(H2d, xs, ys):
    """One tile's triangle-soup BufferGeometry (non-indexed; add_static_trimesh auto-indexes it)."""
    nx, ny = H2d.shape
    pts = []
    V = lambda i, j: tp.Vector3(float(xs[i]), float(ys[j]), float(H2d[i, j]))
    for i in range(nx - 1):
        for j in range(ny - 1):
            pts += [V(i, j), V(i + 1, j), V(i + 1, j + 1), V(i, j), V(i + 1, j + 1), V(i, j + 1)]
    g = tp.BufferGeometry(); g.set_from_points(pts); g.compute_vertex_normals()
    return g


def _add_heightfield(world, k, spacing, shape_idx, amps, geoms):
    """Per-lane: a tile (shared geometry geoms[shape_idx[i]]) at the lane, z-scaled by amps[i] so the
    SAME shape grades to the lane's amplitude. Flat lanes (amp~0) skip the tile (rest on the ground)."""
    for i in range(k):
        a = float(amps[i])
        if a < 0.006:
            continue
        m = tp.Mesh(geoms[int(shape_idx[i])], tp.MeshStandardMaterial())
        m.position.set(0.0, i * spacing, 0.0)
        m.scale.set(1.0, 1.0, a)                                      # scale the [0,1] tile to amplitude a
        world.add_static_trimesh(m)


class SpotHeightfieldEnv:
    def __init__(self, num_envs=1024, device="cuda", seed=0, amp_max=HF_AMP_MAX, flat_only=False):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(HF_EPISODE_S * CONTROL_HZ)
        rng = np.random.default_rng(seed)
        H_np, xs_np, ys_np = make_hf_grids(seed=seed)
        amps = np.linspace(HF_AMP_MIN, (0.0 if flat_only else amp_max), num_envs).astype(np.float32)
        if not flat_only:
            amps[rng.random(num_envs) < FLAT_FRAC] = 0.0
        shape_idx = (np.arange(num_envs) % NUM_SHAPES).astype(np.int64)   # round-robin tile assignment
        geoms = [build_hf_geom(H_np[s], xs_np, ys_np) for s in range(NUM_SHAPES)]
        self.sim = GpuSim(num_envs, lambda world, i: SpotGpu(world, i, SPACING),
                          gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, read_root=True,
                          build_world=lambda world: (_flat_ground(world, num_envs, SPACING),
                                                      _add_heightfield(world, num_envs, SPACING,
                                                                       shape_idx, amps, geoms)))
        dev = self.sim.device
        self.default_q = torch.from_numpy(default_q).to(dev)
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)
        self.stand_q_add = self.default_q[self.a2i].expand(num_envs, -1).contiguous()
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        # terrain ground-truth: each lane's tapered height grid (flattened) + its amplitude
        self.amp = torch.from_numpy(amps).to(dev)
        self.is_rough = self.amp > 0.005
        H_t = torch.from_numpy(H_np).to(dev)                         # [S, nx, ny]
        self.Hsel = H_t[torch.from_numpy(shape_idx).to(dev)].reshape(num_envs, HF_NX * HF_NY)  # [K, nx*ny]
        self.xs = torch.from_numpy(xs_np).to(dev); self.ys = torch.from_numpy(ys_np).to(dev)
        self.gx, self.gy = scan_offsets(dev)                              # [N_SCAN] heading-relative grid offsets
        self.imit_policy = torch.jit.load(os.path.join(fetch_assets(), "spot_policy.pt"),
                                          map_location=dev).eval()
        self.lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev); pos[:, 1] = self.lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.last_act = z(num_envs, ACT_DIM); self._last_obs = z(num_envs, OBS_DIM)
        self.up = z(num_envs); self.cmd = z(num_envs, 3)
        self.cmd_timer = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.ep_start_x = z(num_envs); self.ep_max_climb = z(num_envs)
        self._resample_cmd(torch.arange(num_envs, device=dev))
        self.last_track = 0.0; self.last_flat_track = 0.0; self.last_climb = 0.0; self.last_fell = 0.0

    def _bilinear(self, x, y_local):
        """Per-lane bilinear height in [0,1] from each lane's grid. x,y_local [K] or [K,P]."""
        fx = ((x - HF_X0) / (HF_X1 - HF_X0) * (HF_NX - 1)).clamp(0.0, HF_NX - 1.0001)
        fy = ((y_local + HALF_W) / (2.0 * HALF_W) * (HF_NY - 1)).clamp(0.0, HF_NY - 1.0001)
        ix = fx.long(); iy = fy.long(); tx = fx - ix; ty = fy - iy
        if x.dim() == 1:
            gat = lambda ii, jj: self.Hsel.gather(1, (ii * HF_NY + jj).unsqueeze(1)).squeeze(1)
        else:
            gat = lambda ii, jj: self.Hsel.gather(1, ii * HF_NY + jj)
        h00 = gat(ix, iy); h10 = gat(ix + 1, iy); h01 = gat(ix, iy + 1); h11 = gat(ix + 1, iy + 1)
        return (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11

    def _terrain_h(self, x, y):
        """Exact heightfield height at (x,y): amp * bilinear(grid), gated to the tile footprint."""
        lane = self.lane_y if x.dim() == 1 else self.lane_y[:, None]
        amp = self.amp if x.dim() == 1 else self.amp[:, None]
        y_local = y - lane
        base = self._bilinear(x, y_local)
        on = ((x >= HF_X0) & (x <= HF_X1) & (y_local.abs() < HALF_W)).float()
        return amp * base * on

    def _resample_cmd(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        vx = torch.empty(n, device=dev).uniform_(VX_LO, VX_HI)
        vy = torch.empty(n, device=dev).uniform_(-VY_HI, VY_HI)
        wz = torch.empty(n, device=dev).uniform_(-WZ_HI, WZ_HI)
        drive = (torch.rand(n, device=dev) < FWD_DRIVE_FRAC) & self.is_rough[idx]
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
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
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
        pose = self.base_pose[idx].clone()
        sx = torch.rand(n, device=dev) * 3.0                         # spawn ON the tile, x in [0,3]
        # Reference spawn height to the HIGHEST terrain under the stance footprint (the 4 feet) + a
        # margin, so no foot spawns INSIDE the surface -> the robot drop-settles instead of taking a
        # depenetration jolt ("spawn into the amplitude"). Critical as amplitude ramps up.
        fdx = torch.tensor(FOOT_DX, device=dev); fdy = torch.tensor(FOOT_DY, device=dev)
        fx = sx[:, None] + fdx[None, :]                              # [n,4] foot world-x
        fyl = fdy[None, :].expand(n, 4)                              # [n,4] foot y (lane-local)
        sz = self._sample(idx, fx, fyl).max(dim=1).values + SPAWN_Z + 0.03
        pose[:, 4] = sx                                             # x = index 4
        pose[:, 6] = sz                                             # z = index 6
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, self.stand_q_add[idx], torch.zeros(n, self.sim.dof, device=dev))
        self.steps[idx] = 0; self.last_act[idx] = 0.0
        self.ep_start_x[idx] = sx; self.ep_max_climb[idx] = 0.0
        self._resample_cmd(idx)

    def _sample(self, idx, x, y_local):
        """Bilinear terrain height at (x, y_local) for the subset `idx`. x, y_local are [n,P]."""
        fx = ((x - HF_X0) / (HF_X1 - HF_X0) * (HF_NX - 1)).clamp(0.0, HF_NX - 1.0001)
        fy = ((y_local + HALF_W) / (2.0 * HALF_W) * (HF_NY - 1)).clamp(0.0, HF_NY - 1.0001)
        ix = fx.long(); iy = fy.long(); tx = fx - ix; ty = fy - iy
        H = self.Hsel[idx]                                          # [n, nx*ny]
        g = lambda ii, jj: H.gather(1, ii * HF_NY + jj)            # [n,P]
        h00 = g(ix, iy); h10 = g(ix + 1, iy); h01 = g(ix, iy + 1); h11 = g(ix + 1, iy + 1)
        base = (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11
        on = ((x >= HF_X0) & (x <= HF_X1) & (y_local.abs() < HALF_W)).float()
        return self.amp[idx][:, None] * base * on

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
        a = action
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
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
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

        self.ep_max_climb = torch.maximum(self.ep_max_climb, (x - self.ep_start_x).clamp_min(0.0))
        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self.last_climb = self.ep_max_climb[d].mean().item()
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
    env = SpotHeightfieldEnv(num_envs=K)
    obs = env.reset()
    print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}) finite={bool(torch.isfinite(obs).all())}  "
          f"shapes={NUM_SHAPES} amp_max={HF_AMP_MAX}")
    for _ in range(200):
        obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.sim.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action (stand): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"dist={env.last_climb:.3f}  fell/step={env.last_fell:.3f}  rew={rew.mean().item():+.3f}")
    print("SPOTV2-HEIGHTFIELD ENV SELFTEST: PASS")
