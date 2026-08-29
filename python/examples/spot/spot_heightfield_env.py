"""spotv2 — SMOOTH heightfield terrain (true 2-D rough, no box steps).

A box-stepped bump field (full-width boxes) approximates rough ground; past ~0.20 m amplitude
those box-to-box jumps turn back into steps (the hard case). This env builds a real CONTINUOUS
triangle-mesh heightfield instead — smooth in BOTH x and y — so it stays smooth at higher amplitude.

NO C++ changes: the heightfield is a Python triangle-soup (`BufferGeometry.set_from_points`) handed to
`world.add_static_trimesh` (which auto-indexes a non-indexed soup). For scale, a small set of distinct
smooth-noise TILES is built once (shape diversity) and shared across lanes; each lane references one
tile with `Mesh.scale.z = amp[lane]`, so amplitude is graded across lanes (the difficulty curriculum)
from one geometry per shape. Tiles are edge-tapered to z~0 so they blend into the surrounding flat
ground. The obs scan + reward read the EXACT same height grids by bilinear interpolation -> ground-truth
matches the collision mesh. Same warm-start + velocity-tracking + imitation-anchor machinery as the rest
of spotv2 (identical 96-d clock obs), so train_spot_heightfield.py / play_spot_heightfield.py reuse it.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))   # scratch_clock / scratch_env

import threepp as tp
from threepp.rl import GpuSim, VecTask, load_policy
from spot_deploy import build_spot, default_q, add_to_isaac, isaac_to_add, ACTION_SCALE
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, _flat_ground,
                              scan_offsets, scan_xy, N_SCAN,
                              CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z, PROBE_DX, ACT_DIM,
                              HIDDEN, HALF_W, FLAT_FRAC, VX_LO, VX_HI, VY_HI, WZ_HI, STAND_PROB,
                              FWD_DRIVE_FRAC, CMD_MIN, CMD_MAX, SIG, W_IMIT)
from scratch_clock import CLOCK0, CLOCK_DIM, GAIT_PERIOD, advance, clock_obs, reset_phi
from scratch_env import STIFF_GAINS

OBS_DIM = 48 + CLOCK_DIM + 1 + N_SCAN   # = 96: [proprio(48)|clock(2)|base_above(1)|scan(45)]

# Heightfield-specific anti-forgetting overrides (stronger than the shared box defaults): the smooth
# 0.18 m terrain + longer runs drifted STRAFE tracking (held-out eval caught a ~2x regression the
# averaged-flat gate missed), so anchor HARDER to the base gait on flat and replay MORE flat lanes.
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
          "stand_prob": STAND_PROB, "sig": SIG, "w_imit": W_IMIT,
          "stiff_gains": {k: list(v) for k, v in STIFF_GAINS.items()},
          "gait_period": GAIT_PERIOD}


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


class SpotHeightfieldEnv(VecTask):
    control_hz = CONTROL_HZ
    episode_s = HF_EPISODE_S
    act_dim = ACT_DIM
    control = "drive"                # stiff position PD drives -> targets, not forces
    substeps = SUBSTEPS
    settle_steps = 20                # settle to a clean stand (default targets) after a full reset
    clip_actions = None              # the policy emits ~[-8,8]; do NOT clamp

    def __init__(self, num_envs=1024, device="cuda", seed=0, amp_max=HF_AMP_MAX, flat_only=False,
                 height_source="analytic"):
        if height_source not in ("analytic", "raycast"):
            raise ValueError(f"height_source must be 'analytic' or 'raycast', got {height_source!r}")
        self.height_source = height_source
        self.rays = None                       # set below when height_source == "raycast"
        rng = np.random.default_rng(seed)
        H_np, xs_np, ys_np = make_hf_grids(seed=seed)
        amps = np.linspace(HF_AMP_MIN, (0.0 if flat_only else amp_max), num_envs).astype(np.float32)
        if not flat_only:
            amps[rng.random(num_envs) < FLAT_FRAC] = 0.0
        shape_idx = (np.arange(num_envs) % NUM_SHAPES).astype(np.int64)   # round-robin tile assignment
        geoms = [build_hf_geom(H_np[s], xs_np, ys_np) for s in range(NUM_SHAPES)]
        # Stiff gains (90) = same plant the base gait scratch_flat_best.pt was trained on.
        class _StiffSpot:
            def __init__(self_, world, i):
                self_.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * SPACING), gains=STIFF_GAINS)
        # Under "raycast" the builders write into a CollectedWorld, which forwards every call to
        # the real world and keeps the Mesh it was handed — so the BVH is built from the very
        # triangle soup PhysX cooked, and the scan reads the surface the feet collide with rather
        # than the bilinear field the grid only approximates.
        collector = []

        def _build(world):
            if height_source == "raycast":
                from threepp.rl.raycast import CollectedWorld
                world = CollectedWorld(world)
                collector.append(world)
            _flat_ground(world, num_envs, SPACING)
            _add_heightfield(world, num_envs, SPACING, shape_idx, amps, geoms)

        super().__init__(num_envs, lambda world, i: _StiffSpot(world, i),
                         gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, seed=seed,
                         read_root=True, build_world=_build)
        if height_source == "raycast":
            from threepp.rl.raycast import TerrainRays
            self.rays = TerrainRays.from_objects(collector[0].meshes, device=self.device, up="z")
            collector[0].meshes.clear()      # the BVH owns the triangles now; drop the threepp objects
        dev = self.device
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
        # Anchor = the clock-aware base gait (50-d, normalize_obs=True); frozen throughout.
        _scratch = os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt")
        self.anchor_ac, self.anchor_norm, _ = load_policy(_scratch, device=dev)
        self.anchor_ac.eval()
        self.lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev); pos[:, 1] = self.lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        # per-episode state: registered so the base re-inits it on every reset, full or partial
        self.last_act = self.env_state((ACT_DIM,))
        self.prev_act = self.env_state((ACT_DIM,))
        self.phi = self.env_state(())                                     # phase clock in [0,1)
        self.cmd = self.env_state((3,))
        self.cmd_timer = self.env_state((), init=0, dtype=torch.long)
        self.ep_start_x = self.env_state(())
        self.ep_max_climb = self.env_state(())                            # forward distance this episode
        self._last_obs = torch.zeros(num_envs, OBS_DIM, device=dev)
        self.up = torch.zeros(num_envs, device=dev)
        self._resample_cmd(torch.arange(num_envs, device=dev))            # valid cmd before the first reset()
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
        """Heightfield height at (x,y), gated to the tile footprint.

        Under height_source="raycast" this is a ray into the BVH over the cooked triangle soup.
        The analytic branch is amp * bilinear(grid) — a different surface from the triangulation
        the physics actually uses, so the two agree only up to the bilinear-vs-triangle gap
        (about a centimetre at full amplitude), and it is the raycast that matches the feet."""
        if self.rays is not None:
            return self.rays.heights(x, y)
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

    # ---- the task ---------------------------------------------------------------
    def on_reset(self, idx):
        n = idx.numel()
        dev = self.device
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
        self.phi[idx] = reset_phi(n, dev)       # randomise phase (decorrelate batch)
        self.ep_start_x[idx] = sx
        self._resample_cmd(idx)

    def _sample(self, idx, x, y_local):
        """Terrain height at (x, y_local) for the subset `idx`. x, y_local are [n,P]."""
        if self.rays is not None:
            return self.rays.heights(x, y_local + self.lane_y[idx][:, None])
        fx = ((x - HF_X0) / (HF_X1 - HF_X0) * (HF_NX - 1)).clamp(0.0, HF_NX - 1.0001)
        fy = ((y_local + HALF_W) / (2.0 * HALF_W) * (HF_NY - 1)).clamp(0.0, HF_NY - 1.0001)
        ix = fx.long(); iy = fy.long(); tx = fx - ix; ty = fy - iy
        H = self.Hsel[idx]                                          # [n, nx*ny]
        g = lambda ii, jj: H.gather(1, ii * HF_NY + jj)            # [n,P]
        h00 = g(ix, iy); h10 = g(ix + 1, iy); h01 = g(ix, iy + 1); h11 = g(ix + 1, iy + 1)
        base = (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11
        on = ((x >= HF_X0) & (x <= HF_X1) & (y_local.abs() < HALF_W)).float()
        return self.amp[idx][:, None] * base * on

    def act(self, a):
        # FULL policy action (not a residual): isaac -> add-order drive targets. The settle loop
        # feeds a=0, which lands exactly on the default stand targets.
        self.prev_act.copy_(self.last_act)
        self.last_act.copy_(a)
        return (self.default_q + ACTION_SCALE * a)[:, self.a2i]

    def on_settled(self):
        self.up = up_z(self.sim.root_quat)

    def on_step(self, s):
        self.phi.copy_(advance(self.phi))                                    # clock after physics, before next obs
        self.cmd_timer -= 1
        self._resample_cmd(torch.nonzero(self.cmd_timer <= 0, as_tuple=False).squeeze(-1))

        q = self.sim.root_quat
        self.up = up_z(q)
        x, y, zz = self.sim.root_position[:, 0], self.sim.root_position[:, 1], self.sim.root_position[:, 2]
        self._roll = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        self._ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        self._lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        h_here = self._terrain_h(x, y)
        self._base_above = zz - h_here
        cyaw, syaw = heading_cossin(q)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        ahead = self._terrain_h(px, py) - h_here[:, None]
        change = ahead.abs().max(dim=1).values
        self._w_imit = (1.0 - change / 0.10).clamp(0.0, 1.0)
        self.ep_max_climb.copy_(torch.maximum(self.ep_max_climb, (x - self.ep_start_x).clamp_min(0.0)))

    def terminated(self, s):
        return (self.up < 0.35) | (self._base_above < 0.18)

    def reward_terms(self, s, a):
        # Anchor: the 50-d clock base gait with its frozen RunningNorm (obs[:,:50] = proprio+clock).
        anchor_a = self.anchor_ac.act_mean(self.anchor_norm.norm(self._last_obs[:, :50]))
        imit = self._w_imit * (a - anchor_a).pow(2).mean(dim=1)
        arate = a - self.prev_act

        e_lin = (self.cmd[:, 0] - self._lin_b[:, 0]).pow(2) + (self.cmd[:, 1] - self._lin_b[:, 1]).pow(2)
        e_ang = (self.cmd[:, 2] - self._ang_b[:, 2]).pow(2)
        track_lin = torch.exp(-e_lin / SIG)
        track_ang = torch.exp(-e_ang / SIG)
        terms = {
            "track_lin": 3.0 * track_lin,
            "track_ang": 1.5 * track_ang,
            "alive": torch.full((self.K,), 0.05, device=self.device),
            "roll": -1.0 * self._roll.pow(2),
            "vz": -0.1 * self._lin_b[:, 2].pow(2),
            "angrate": -0.05 * (self._ang_b[:, 0].pow(2) + self._ang_b[:, 2].pow(2)),
            "scrape": -3.0 * torch.relu(0.30 - self._base_above),
            "arate": -0.001 * arate.pow(2).mean(dim=1),
            "imit": -W_IMIT * imit,
            "fell": -5.0 * s.terminated.float(),
        }
        self.last_fell = s.terminated.float().mean().item()
        self.last_track = (track_lin + track_ang).mean().item()
        flat = ~self.is_rough
        self.last_flat_track = ((track_lin + track_ang)[flat]).mean().item() if bool(flat.any()) else float("nan")
        return terms

    def on_done(self, idx):
        self.last_climb = self.ep_max_climb[idx].mean().item()

    def observe(self, s):
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos = s.joint_pos[:, self.i2a] - self.default_q
        jv_isaac = s.joint_vel[:, self.i2a]
        x, y, zz = s.root_pos[:, 0], s.root_pos[:, 1], s.root_pos[:, 2]
        cyaw, syaw = heading_cossin(q)
        h_here = self._terrain_h(x, y)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        ahead = (self._terrain_h(px, py) - h_here[:, None]).clamp(-1.0, 1.0)
        base_above = (zz - h_here).unsqueeze(-1)
        clk = clock_obs(self.phi)                                          # [K,2] clock after last substep
        # Layout: [proprio(48)|clock(2)|base_above(1)|scan(45)] = 96-d
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act,
                         clk, base_above, ahead], dim=1)
        self._last_obs = obs
        return obs

    def config(self):
        return {**super().config(), **CONFIG}

    @torch.no_grad()
    def measure_tracking(self, act_fn, cmd, steps=160, warm=60):
        dev = self.device
        c = torch.tensor(cmd, device=dev, dtype=torch.float32).expand(self.K, 3).contiguous()
        obs = self.reset()
        errs = []
        for t in range(steps):
            self.cmd.copy_(c)
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
    src = os.environ.get("HEIGHT_SOURCE", "analytic")     # HEIGHT_SOURCE=raycast -> Warp BVH scan
    env = SpotHeightfieldEnv(num_envs=K, height_source=src)
    obs = env.reset()
    assert obs.shape == (K, 96), f"expected obs (K,96), got {tuple(obs.shape)}"
    print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}=96) finite={bool(torch.isfinite(obs).all())}  "
          f"shapes={NUM_SHAPES} amp_max={HF_AMP_MAX}  height_source={src}"
          + (f" {env.rays}" if env.rays is not None else ""))
    for _ in range(200):
        obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action (stand): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"dist={env.last_climb:.3f}  fell/step={env.last_fell:.3f}  rew={rew.mean().item():+.3f}")
    print("per-term:", env.stats_line() or "(no episode finished yet)")
    print("SPOTV2-HEIGHTFIELD ENV SELFTEST: PASS")
