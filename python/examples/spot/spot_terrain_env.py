"""spotv2 — Phase 1: velocity-command TRACKING transfer of the Isaac Spot walker onto terrain.

This is the smallest-diff step of the uneven-terrain plan, kept in its OWN folder so the working
`examples/spot/` stays untouched. It depends ONLY on the stable asset/robot layer
(`spot_deploy.build_spot` + the Isaac joint orderings); the small GpuSim helpers are reimplemented
here so this env is decoupled from any in-progress edits to `examples/spot/spot_stairs_env.py`.

WHAT CHANGES vs. the working stairs fine-tune (`SpotStairFTEnv`):
  - KEEP   : warm-start of the Isaac flat walker into a 58-d obs whose terrain columns zero-init
             (so the policy BEGINS bit-identical to the flat walker — see train_spot_terrain.py),
             the scan-gated imitation anchor, raw obs (normalize_obs=False), the tent terrain.
  - REPLACE: the fixed FWD_CMD=0.6 + deterministic lane-hold `_cmd()` law, the `20*new_x` /
             `15*new_high` ratchets, and the `-2*y_err^2` / `-1*yaw^2` lane terms — ALL of which
             forbid turning and overfit to climbing one tent straight-on.
  - WITH   : RANDOMIZED body-frame velocity commands [vx,vy,wz] (+ a stand fraction) and an
             exp-kernel velocity-TRACKING reward. Steering becomes a directly-rewarded skill on
             every lane, climbing emerges from tracking forward velocity over the tent, and the
             flat lanes (FLAT_FRAC) now carry the FULL steering envelope so that distribution is
             never out-of-sample. The forward height-scan is rotated by heading (correct under turns).

GOAL OF PHASE 1: prove that (a) tracking-driven climbing works on the existing tents and (b) flat
steering survives (held-out flat-tracking within ~1.1x the teacher — see train_spot_terrain.py
--eval) BEFORE adding terrain variety / a 2-D scan / a difficulty curriculum (Phase 2).

  obs (94): base lin vel (body 3), base ang vel (body 3), projected gravity (3), command [vx,vy,wz] (3),
            joint pos rel default (12 isaac order), joint vel (12 isaac), last action (12),
            base height over local terrain (1), 2-D height-scan grid (45 = 9 fwd x 5 lat, heading-relative)
  action (12): the FULL Isaac-order policy action -> joint targets default_q + ACTION_SCALE*a (NOT clamped)
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
_PYROOT = os.path.dirname(_EXAMPLES)
sys.path.insert(0, _PYROOT)                              # `import threepp` / `threepp.rl`
sys.path.insert(0, os.path.join(_EXAMPLES, "spot"))      # stable asset layer: spot_deploy

import threepp as tp
from threepp.rl import GpuSim
from spot_deploy import (build_spot, fetch_assets, default_q, add_to_isaac, isaac_to_add,
                         ACTION_SCALE)

# --------------------------------------------------------------------------- #
#  Sim constants (Isaac contract: 50 Hz policy, Z-up, gravity -9.81)
# --------------------------------------------------------------------------- #
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
SUBSTEPS = 4                 # GPU physics substeps per control tick
SPACING = 3.0               # metres between per-env lanes (along +Y)
SPAWN_Z = 0.42              # natural default-pose stand height (Isaac default soft gains; feet on ground)

# --------------------------------------------------------------------------- #
#  Tent terrain (reused from the working stairs env so Phase 1 validates on KNOWN geometry)
# --------------------------------------------------------------------------- #
STAIR_X0 = 1.6              # tents start 1.6 m ahead of each spawn
LAND_LEN = 1.0             # flat landing at a tent's peak (between ascent and descent)
RISE_MIN, RISE_MAX = 0.02, 0.20   # per-lane step height, graded across lanes = a built-in difficulty sweep
RUN_MIN, RUN_MAX = 0.28, 0.36     # per-lane tread depth (randomized so it doesn't overfit one geometry)
N_UP_MIN, N_UP_MAX = 3, 8         # steps up (= steps down) per tent lane
FLAT_FRAC = 0.25                  # fraction of FLAT lanes (rise=0) -> these carry the FULL steering envelope
HALF_W = SPACING * 0.46           # half-width of a tent (= box width / 2); beyond it is flat ground (2-D terrain)

# --------------------------------------------------------------------------- #
#  Obs / policy
# --------------------------------------------------------------------------- #
# 2-D heading-relative height-scan GRID: forward (x_local) x lateral (y_local), rotated by heading.
# The forward axis keeps the old 1-D PROBE_DX offsets and the lateral axis is symmetric about 0, so
# the grid's centerline row (dy=0) IS the old 1-D scan -> a trained 1-D policy transfers cleanly
# (centerline copied, lateral columns zero-init). See warmstart_expand_scan in train_spot_terrain.py.
PROBE_DX = (-0.35, -0.15, 0.05, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1)   # forward offsets (m ahead, heading-relative)
PROBE_DY = (-0.30, -0.15, 0.0, 0.15, 0.30)                       # lateral offsets (m, +y = LEFT of heading)
N_DX, N_DY = len(PROBE_DX), len(PROBE_DY)
# flattened FORWARD-MAJOR grid: index = fi*N_DY + dj  (for each forward offset, the N_DY lateral cols)
SCAN_GX = tuple(float(dx) for dx in PROBE_DX for _dy in PROBE_DY)   # [N_SCAN] local forward offset per cell
SCAN_GY = tuple(float(dy) for _dx in PROBE_DX for dy in PROBE_DY)   # [N_SCAN] local lateral offset per cell
N_SCAN = len(SCAN_GX)                                               # = 45
SCAN_CENTER = N_DY // 2                                             # lateral index of dy=0 (the old 1-D line)
# left<->right mirror (y -> -y): reverse the N_DY lateral cols within each forward group
SCAN_MIRROR_PERM = tuple(fi * N_DY + (N_DY - 1 - dj) for fi in range(N_DX) for dj in range(N_DY))
OBS_DIM = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 1 + N_SCAN                 # = 94 (48 proprio + base_above + 45 scan)
ACT_DIM = 12
HIDDEN = (512, 256, 128)         # matches the Isaac actor MLP (48->512->256->128->12) for the warm-start

# --------------------------------------------------------------------------- #
#  Task: randomized velocity commands (the steering envelope) + episode
# --------------------------------------------------------------------------- #
EPISODE_S = 20.0           # time to drive around + climb up/across/down at the commanded velocity
VX_LO, VX_HI = -1.0, 1.5   # forward command range (within the warm-started walker's demonstrated envelope)
VY_HI = 0.8                # strafe command range (+-)
WZ_HI = 1.2                # yaw-rate command range (+-)
STAND_PROB = 0.12          # fraction of commands forced to ZERO -> learn to stand still on command
FWD_DRIVE_FRAC = 0.5       # fraction of TENT-lane commands biased ~straight forward -> actually drive into + climb tents
TENT_SPAWN_FRAC = 0.5      # fraction of TENT-lane resets that spawn ON the tent (forced forward) -> climbing IS the experience
CMD_MIN, CMD_MAX = 120, 320   # per-env steps between in-episode command resamples
SIG = 0.25                 # exp-kernel width for velocity tracking (== IsaacLab std 0.5)
W_IMIT = 0.1               # scan-gated imitation-of-the-Isaac-walker weight (anti-forgetting; was 0.05 in stairs FT)

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "rise_min": RISE_MIN, "rise_max": RISE_MAX, "run_min": RUN_MIN, "run_max": RUN_MAX,
          "episode_s": EPISODE_S, "stair_x0": STAIR_X0, "probe_dx": list(PROBE_DX),
          "probe_dy": list(PROBE_DY), "n_scan": N_SCAN,
          "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": list(HIDDEN),
          "vx": [VX_LO, VX_HI], "vy_hi": VY_HI, "wz_hi": WZ_HI, "stand_prob": STAND_PROB,
          "sig": SIG, "w_imit": W_IMIT}


# --------------------------------------------------------------------------- #
#  Small GpuSim helpers (reimplemented locally so spotv2 is self-contained)
# --------------------------------------------------------------------------- #
def quat_rotate_inverse(q, v):
    """Rotate world-frame vectors v [N,3] into the body frame given body->world quat q [N,4]
    (qx,qy,qz,qw). Isaac Lab's exact formula (the policy was trained with it)."""
    qw = q[:, 3]
    qvec = q[:, :3]
    a = v * (2.0 * qw ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(qvec, v, dim=-1) * qw.unsqueeze(-1) * 2.0
    c = qvec * (qvec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def up_z(q):
    """World-z component of the body up-axis (cos of base tilt). q=[N,4] (qx,qy,qz,qw)."""
    return 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)


def heading_cossin(q):
    """cos(yaw), sin(yaw) of the body +x axis projected into the world XY plane. q=[N,4]."""
    hx = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    hy = 2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    nrm = (hx * hx + hy * hy).sqrt().clamp(min=1e-6)
    return hx / nrm, hy / nrm


# --------------------------------------------------------------------------- #
#  Heading-relative 2-D scan grid (shared by every env + the deploy players)
# --------------------------------------------------------------------------- #
def scan_offsets(device):
    """Local-frame grid offsets (gx forward, gy lateral) as torch tensors [N_SCAN]. Store once per env."""
    return (torch.tensor(SCAN_GX, device=device), torch.tensor(SCAN_GY, device=device))


def scan_xy(x, y, cyaw, syaw, gx, gy):
    """World (px, py) of the heading-relative grid -> each [K, N_SCAN]. x,y,cyaw,syaw: [K]; gx,gy: [N_SCAN].
    Rotate the local (gx, gy) offsets by yaw: world = R(yaw) @ local (so the grid turns with the robot)."""
    dx = gx[None, :] * cyaw[:, None] - gy[None, :] * syaw[:, None]
    dy = gx[None, :] * syaw[:, None] + gy[None, :] * cyaw[:, None]
    return x[:, None] + dx, y[:, None] + dy


def scan_xy_np(x, y, cyaw, syaw):
    """numpy single-robot version of scan_xy: scalars x,y,cyaw,syaw -> world (px, py) arrays [N_SCAN]."""
    gx = np.asarray(SCAN_GX); gy = np.asarray(SCAN_GY)
    return x + gx * cyaw - gy * syaw, y + gx * syaw + gy * cyaw


class SpotGpu:
    """build_robot factory for GpuSim: one Spot at a per-env lateral offset (Isaac default PD gains —
    the warm-started walker was trained with them). Exposes `.art`."""
    def __init__(self, world, i, spacing=SPACING):
        self.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * spacing))


def _flat_ground(world, k, spacing):
    g = tp.Mesh(tp.BoxGeometry(60, spacing * k + 20, 1.0), tp.MeshStandardMaterial())
    g.position.set(20.0, spacing * (k - 1) * 0.5, -0.5)     # top at z=0, spans the lane grid + forward travel
    world.add_static(g)


def _add_tent(world, k, spacing, rises, runs, n_ups, x0=STAIR_X0, land=LAND_LEN):
    """Per-lane up-then-down staircase (a 'tent'): ascend n -> flat landing -> descend n. rise<0.005 ->
    flat lane (no boxes). Solid boxes from the ground up to each tread top; box width = spacing*0.92
    (= 2*HALF_W) so off-lane is flat ground (the 2-D terrain the heading-relative scan reads)."""
    for i in range(k):
        r = float(rises[i]); run = float(runs[i]); n = int(n_ups[i])
        if r < 0.005:
            continue
        w = spacing * 0.92
        for s in range(n):                                  # ascend: tread s top at (s+1)*r
            h = (s + 1) * r
            b = tp.Mesh(tp.BoxGeometry(run, w, h), tp.MeshStandardMaterial())
            b.position.set(x0 + s * run + run * 0.5, i * spacing, h * 0.5)
            world.add_static(b)
        up_end = x0 + n * run
        top = n * r
        lb = tp.Mesh(tp.BoxGeometry(land, w, top), tp.MeshStandardMaterial())   # flat landing at the peak
        lb.position.set(up_end + land * 0.5, i * spacing, top * 0.5)
        world.add_static(lb)
        land_end = up_end + land
        for s in range(n - 1):                              # descend: tread s top at (n-1-s)*r (last=0 -> ground)
            h = (n - 1 - s) * r
            b = tp.Mesh(tp.BoxGeometry(run, w, h), tp.MeshStandardMaterial())
            b.position.set(land_end + s * run + run * 0.5, i * spacing, h * 0.5)
            world.add_static(b)


def _tent_profile(x, rise, run, n_up, x0=STAIR_X0, land=LAND_LEN):
    """Up-then-down stair height at world-x (1-D in x): flat approach -> ascend n_up -> flat landing ->
    descend n_up -> flat run-out. Vectorized; rise/run/n_up broadcast with x ([K] or [K,P])."""
    up_end = x0 + n_up * run
    land_end = up_end + land
    asc = torch.minimum(torch.clamp(torch.floor((x - x0) / run) + 1.0, min=0.0), n_up)       # steps climbed
    desc = torch.minimum(torch.clamp(torch.floor((x - land_end) / run) + 1.0, min=0.0), n_up)  # steps descended
    steps = torch.where(x < up_end, asc, torch.where(x < land_end, n_up, n_up - desc))
    return steps * rise


# =============================================================================
#  SpotTerrainEnv — velocity-command tracking on tent terrain
# =============================================================================
class SpotTerrainEnv:
    def __init__(self, num_envs=2048, device="cuda", seed=0, rise_max=RISE_MAX, flat_only=False):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(EPISODE_S * CONTROL_HZ)
        rng = np.random.default_rng(seed)
        if flat_only:                                                # held-out flat-steering eval env
            rises = np.zeros(num_envs, np.float32)
        else:
            rises = np.linspace(RISE_MIN, rise_max, num_envs).astype(np.float32)   # graded = difficulty sweep
            rises[rng.random(num_envs) < FLAT_FRAC] = 0.0                          # FLAT lanes -> full-steering replay
        runs = rng.uniform(RUN_MIN, RUN_MAX, num_envs).astype(np.float32)
        n_ups = rng.integers(N_UP_MIN, N_UP_MAX + 1, num_envs).astype(np.float32)
        self.sim = GpuSim(num_envs, lambda world, i: SpotGpu(world, i, SPACING),
                          gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, read_root=True,
                          build_world=lambda world: (_flat_ground(world, num_envs, SPACING),
                                                      _add_tent(world, num_envs, SPACING, rises, runs, n_ups)))
        dev = self.sim.device
        self.default_q = torch.from_numpy(default_q).to(dev)                  # [12] isaac order
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)    # add -> isaac index
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)    # isaac -> add index
        self.stand_q_add = self.default_q[self.a2i].expand(num_envs, -1).contiguous()
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        self.rise = torch.from_numpy(rises).to(dev)
        self.run = torch.from_numpy(runs).to(dev)
        self.n_up = torch.from_numpy(n_ups).to(dev)
        self.is_tent = self.rise > 0.005                                     # [K] tent lane vs flat lane
        self.gx, self.gy = scan_offsets(dev)                                 # [N_SCAN] heading-relative grid offsets
        self.imit_policy = torch.jit.load(os.path.join(fetch_assets(), "spot_policy.pt"),
                                          map_location=dev).eval()           # Isaac flat walker = the flat-gait anchor
        self.lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev); pos[:, 1] = self.lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)   # [K,7] facing +x
        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.last_act = z(num_envs, ACT_DIM)                                  # isaac-order action (for the obs)
        self._last_obs = z(num_envs, OBS_DIM)                                 # obs the policy acted on (imitation input)
        self.up = z(num_envs)
        self.cmd = z(num_envs, 3)                                             # [vx, vy, wz] body-frame velocity command
        self.cmd_timer = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.ep_max_climb = z(num_envs)                                       # peak terrain height under base this episode
        self._resample_cmd(torch.arange(num_envs, device=dev))
        self.last_track = 0.0; self.last_flat_track = 0.0; self.last_climb = 0.0; self.last_fell = 0.0

    # ----- terrain (2-D: tent in-lane, flat outside the box width) -----
    def _terrain_h(self, x, y):
        """Real ground height at world (x,y). x,y broadcastable [K] or [K,P]. Tent height in-lane,
        0 (flat) beyond HALF_W -> correct when the robot turns/strafes off its lane."""
        lane, rise, run, nup = self.lane_y, self.rise, self.run, self.n_up
        if x.dim() == 2:
            lane = lane[:, None]; rise = rise[:, None]; run = run[:, None]; nup = nup[:, None]
        h = _tent_profile(x, rise, run, nup)
        on = (torch.abs(y - lane) < HALF_W).float()
        return h * on

    # ----- randomized velocity commands (the steering envelope) -----
    def _resample_cmd(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        vx = torch.empty(n, device=dev).uniform_(VX_LO, VX_HI)
        vy = torch.empty(n, device=dev).uniform_(-VY_HI, VY_HI)
        wz = torch.empty(n, device=dev).uniform_(-WZ_HI, WZ_HI)
        # forward-drive bias only on TENT lanes -> the robot actually drives into + climbs tents.
        # FLAT lanes never get the bias -> they always sample the FULL turn/strafe/reverse envelope
        # (the steering-replay that keeps the original walker's command distribution in-sample).
        drive = (torch.rand(n, device=dev) < FWD_DRIVE_FRAC) & self.is_tent[idx]
        vx = torch.where(drive, torch.empty(n, device=dev).uniform_(0.4, VX_HI), vx)
        vy = torch.where(drive, vy * 0.2, vy)
        wz = torch.where(drive, wz * 0.2, wz)
        cmd = torch.stack([vx, vy, wz], dim=1)
        cmd[torch.rand(n, device=dev) < STAND_PROB] = 0.0                     # some envs get a STAND (zero) command
        self.cmd[idx] = cmd
        self.cmd_timer[idx] = torch.randint(CMD_MIN, CMD_MAX + 1, (n,), device=dev)

    def _obs(self):
        s = self.sim
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, s.root_linvel)
        ang_b = quat_rotate_inverse(q, s.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos = s.joint_pos[:, self.i2a] - self.default_q                      # isaac-order joint deviation
        jv_isaac = s.joint_vel[:, self.i2a]
        x, y, zz = s.root_position[:, 0], s.root_position[:, 1], s.root_position[:, 2]
        cyaw, syaw = heading_cossin(q)
        h_here = self._terrain_h(x, y)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)                  # HEADING-relative 2-D scan grid
        ahead = (self._terrain_h(px, py) - h_here[:, None]).clamp(-1.0, 1.0)  # terrain rise/drop per cell (clipped)
        base_above = (zz - h_here).unsqueeze(-1)
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act,
                         base_above, ahead], dim=1)                           # [K, OBS_DIM]
        self._last_obs = obs                                                  # imitation reads obs[:, :48]
        return obs

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        pose = self.base_pose[idx].clone()                                   # [n,7] PhysX layout [quat(4), pos(3)]
        on_tent = (torch.rand(n, device=dev) < TENT_SPAWN_FRAC) & self.is_tent[idx]
        span = 2.0 * self.n_up[idx] * self.run[idx] + LAND_LEN               # full tent length (up + landing + down)
        sx = STAIR_X0 + torch.rand(n, device=dev) * span                     # random point along the tent
        sz = _tent_profile(sx, self.rise[idx], self.run[idx], self.n_up[idx]) + SPAWN_Z   # base = terrain + stand clearance
        pose[:, 4] = torch.where(on_tent, sx, pose[:, 4])                    # x is index 4 (quat-first root layout)
        pose[:, 6] = torch.where(on_tent, sz, pose[:, 6])                    # z is index 6
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, self.stand_q_add[idx], torch.zeros(n, self.sim.dof, device=dev))
        self.steps[idx] = 0; self.last_act[idx] = 0.0
        self.ep_max_climb[idx] = _tent_profile(pose[:, 4], self.rise[idx], self.run[idx], self.n_up[idx])
        self._resample_cmd(idx)
        fwd = idx[on_tent]                                                   # spawned ON the tent -> force a forward cmd
        if fwd.numel() > 0:
            self.cmd[fwd, 0] = torch.empty(fwd.numel(), device=dev).uniform_(0.5, VX_HI)
            self.cmd[fwd, 1:] = 0.0

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        self.sim.read()
        for _ in range(20):                                                  # settle to a clean stand (default targets)
            self.sim.apply_drive_target(self.stand_q_add)
            self.sim.substep(DT / SUBSTEPS, SUBSTEPS)        # advance n substeps, read once
        self.last_act.zero_()
        self.up = up_z(self.sim.root_quat)
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        # NO clamp: the Isaac walker emits actions in ~[-8, 8] (default_q + ACTION_SCALE*a is the joint
        # target); clamping to [-1,1] would destroy the warm-started gait.
        a = action
        prev_a = self.last_act
        targets_isaac = self.default_q + ACTION_SCALE * a                    # FULL policy action (not a residual)
        self.sim.apply_drive_target(targets_isaac[:, self.a2i])              # isaac -> add-order drive targets
        self.sim.substep(DT / SUBSTEPS, SUBSTEPS)                            # advance n substeps, read once
        self.steps += 1
        self.last_act = a
        self.cmd_timer -= 1                                                  # in-episode velocity-command changes
        self._resample_cmd(torch.nonzero(self.cmd_timer <= 0, as_tuple=False).squeeze(-1))

        q = self.sim.root_quat
        self.up = up_z(q)
        x, y, zz = self.sim.root_position[:, 0], self.sim.root_position[:, 1], self.sim.root_position[:, 2]
        roll = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])                  # lateral-tilt proxy (keep level)
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        h_here = self._terrain_h(x, y)
        base_above = zz - h_here
        # scan (unclipped) for the imitation gate: anchor to the Isaac walker only where terrain is flat
        cyaw, syaw = heading_cossin(q)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        ahead = self._terrain_h(px, py) - h_here[:, None]
        change = ahead.abs().max(dim=1).values                               # nearest terrain change (up OR down)
        w_imit = (1.0 - change / 0.10).clamp(0.0, 1.0)                        # 1 on TRUE flat -> 0 once a step nears
        isaac_a = self.imit_policy(self._last_obs[:, :48])                    # the clean Isaac walker's action here
        imit = w_imit * (a - isaac_a).pow(2).mean(dim=1)
        arate = a - prev_a
        fell = (self.up < 0.35) | (base_above < 0.18)

        # ** PRIMARY OBJECTIVE: track the commanded body-frame velocity (this IS the steering). **
        e_lin = (self.cmd[:, 0] - lin_b[:, 0]).pow(2) + (self.cmd[:, 1] - lin_b[:, 1]).pow(2)
        e_ang = (self.cmd[:, 2] - ang_b[:, 2]).pow(2)
        track_lin = torch.exp(-e_lin / SIG)
        track_ang = torch.exp(-e_ang / SIG)
        rew = (3.0 * track_lin                          # track commanded vx,vy (forward + strafe)
               + 1.5 * track_ang                        # track commanded yaw rate (turn)
               + 0.05                                   # alive
               - 1.0 * roll.pow(2)                       # stay level — ROLL only (climbing legitimately PITCHES)
               - 0.1 * lin_b[:, 2].pow(2)               # LIGHT vertical-vel damp (climbing needs vz)
               - 0.05 * (ang_b[:, 0].pow(2) + ang_b[:, 2].pow(2))   # light roll/yaw-rate damp (pitch-rate left free)
               - 3.0 * torch.relu(0.30 - base_above)    # anti-scrape: don't drag the body over the terrain
               - 0.001 * arate.pow(2).mean(dim=1)       # action rate (smooth; scaled for the ~+-7 action range)
               - W_IMIT * imit                          # BE the Isaac walker on flat -> preserve the steering gait
               - 5.0 * fell.float())
        # NOTE (Phase 2): foot terms (feet_air_time / clearance / slip) plug in here. They are feasible
        # without a new binding -- set GpuSim(..., read_links=True) and port spot_walk_env._foot_world()
        # (link_pose/link_linvel give foot-tip kinematics; contact = tip_z < threshold). Omitted in Phase 1
        # to isolate the command+reward-structure change as the single variable.

        self.ep_max_climb = torch.maximum(self.ep_max_climb, h_here)         # peak terrain height under base (climb metric)
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
        flat = ~self.is_tent                                                 # flat-lane tracking = the steering-regression proxy
        self.last_flat_track = ((track_lin + track_ang)[flat]).mean().item() if bool(flat.any()) else float("nan")
        return obs, rew, done, term_obs, timeout

    @torch.no_grad()
    def measure_tracking(self, act_fn, cmd, steps=160, warm=60):
        """Hold a FIXED command and report mean tracking error ||lin_b_xy - cmd_xy|| + |ang_b_z - wz|
        over the last (steps-warm) ticks. `act_fn(obs) -> [K,12]` isaac-order action. For the held-out
        flat-steering regression eval (run it on a flat_only env for both the policy and the teacher)."""
        dev = self.sim.device
        c = torch.tensor(cmd, device=dev, dtype=torch.float32).expand(self.K, 3).contiguous()
        obs = self.reset()
        errs = []
        for t in range(steps):
            self.cmd = c                                                     # freeze the command (override resamples)
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
    env = SpotTerrainEnv(num_envs=K)
    obs = env.reset()
    print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}) finite={bool(torch.isfinite(obs).all())}")
    for _ in range(200):
        obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.sim.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action (stand): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"climb={env.last_climb:.3f}  fell/step={env.last_fell:.3f}  rew={rew.mean().item():+.3f}")
    print("SPOTV2-TERRAIN ENV SELFTEST: PASS")
