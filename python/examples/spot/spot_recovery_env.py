"""Fall recovery for Spot: get from lying on the ground (on its back, on a side, anyhow) back onto its feet and stand.

A VecTask on GpuSim with Spot's walking plant (stiff gains, real torque limits 45/45/115 N·m) plus self-collision, so a
rolling robot cannot swing a leg through its own body. The observation is the walking policy's 96-d layout with the
command at zero and the clock at the (0,0) stand sentinel, the scan against flat ground (all zeros) and base_above =
base height, so the course policy's warm start and deploy code run it unchanged (spot_slam.py --recovery-model).

Feasibility, MEASURED 2026-09-13 (one-off probes, direct-GPU TGS 5 ms, real torque limits):
  - self-collision: the stance is bit-identical with it on (no self-contact), nothing explodes over 512 random extreme
    poses, and a spawn with a leg already inside the body is kicked out at up to 27 m/s: spawns are drawn clear of
    self-penetration here (_clear_of_self).
  - drops from 0.3-0.6 m in random orientations with random joints and limp drives: no NaN, no link below the ground, joint
    limits overshot by at most 0.056 rad; 63-69% of drops come to rest on the back.
  - roll-over: a cross-entropy search over open-loop joint-target keyframes (4 x 0.5 s, K=1024, 8 rounds) from lying on
    the right side found rollouts that end standing (up_z 0.999, base 0.466 m) with torque at the caps (hx 45, hy 45,
    kn 115 N·m peak; saturated <= 10% of substeps per joint). About 1 in 1000 random keyframe sets gets there, so the
    plant can do it and the search is the hard part.
  - hop (for the jump later): open-loop crouch then extension reaches a 0.83 m base rise and 0.75 s of flight upright,
    peak joint speeds 8 / 23 / 31 rad/s (hx / hy / kn). PhysX's per-joint max velocity is not bound (C++ or Python), so
    nothing caps those speeds and there is no torque-speed curve: that number is optimistic.

Resets (per env, every episode): a share STAND_FRAC starts upright (tilt <= 15 deg, near the stance). The rest are fallen
at the env's curriculum level: L0 tilted <= 25 deg, L1 25-60 deg, L2 on a side or tilted 60-120 deg, L3 40% on the back,
40% on a side, 20% any orientation; random yaw; L0-L1 joints near the stance, L2-L3 uniform within the limits (drawn
clear of self-penetration). The lowest point starts DROP above the ground and every episode opens with LIMP_TICKS of limp
drives (the target follows the joints), so it lands the way a fall lands. Rewards are masked over the limp ticks.
Promote on a success within the episode, demote on none. No fall terminations: only a numerical blow-up ends an episode.

Success: up_z > SUCCESS_UP, base height > SUCCESS_Z, mean |q - stance| < SUCCESS_QERR, held SUCCESS_HOLD ticks (1 s).

Wave 2 knobs (2026-09-13), each off by default so the defaults are the pilot bit for bit:
  - action_scale S + target_clip: targets = default_q + S * action, clipped to the joint limits. The pilot's 0.2 left the
    eval p95 torque at 10-25 N·m while the roll-over search needed the caps.
  - w_prog W: potential-based progress shaping W * (gamma * phi(s') - phi(s)), phi = (1 - g_z) / 2 with g_z the z of
    projected gravity in the body frame (-1 upright, +1 on the back): phi is 0 on the back, 0.5 on a side, 1 upright.
    up_slope C: the upright term becomes W_UP * ((1 - C) phi^2 + C phi), whose slope on the back is W_UP * C instead
    of the pilot's zero (0% back success in all four pilot runs).
  - hard_frac F: a share F of the fallen resets draws an L3 start whatever the env's level, and those episodes neither
    promote nor demote (only 6-58 of 2048 envs reached L3 in 1000 pilot iterations).
  - fallen_effort_free: the torque / action-rate / joint-limit penalties are scaled by clamp((up_z - 0.3) / 0.6, 0, 1),
    and w_down C pays -C per live tick while up_z < 0.5. Watching the pilot in spot_slam, "it kinda just gives up": while
    down every attempt paid effort penalties and nothing paid partial progress, so lying still was the cheapest policy.
  - gave_up (a metric, always on, never in obs or reward): the episode had >= 1 s in a row down (up_z < 0.5) with mean
    |joint velocity| < 0.25 rad/s after the limp ticks. Calibrated 2026-09-13 at K=256 from L3 starts: zero actions (a
    policy that stopped trying) count 98.4% gave up at a median 0.064 rad/s, while per-tick target jitter of 0.06 rad
    already moves the joints at a median 1.34 rad/s and counts 0%.

Wave 3 knobs (2026-09-14), off by default so the defaults are wave 2 bit for bit. Watched in spot_slam, wave 2's
recovery "does not seem physically possible": after the limp settle its joints reach |qd| p99 14-27 and max 34-72 rad/s
(256 L3 starts), and only PhysX's default max joint velocity (100 rad/s, also the Isaac export's velocity_limit) caps them.
  - qd_max V: the drive target moves at most V * DT per control tick. The limit acts on the action (V * DT / S), so
    last_act, which the policy observes, is the action the drive actually holds; the joint-limit clip comes after it and
    only shortens a step. Limp ticks are not limited (their target follows the joints), so each episode's limiter starts
    from where the joints landed.
  - w_qd_excess W: -W * mean((|qd| - V)_+^2) on the measured joint speeds, for the speed a target limit cannot stop
    (gravity whips, bounces off the ground).
  - joint speed metrics, always on: |qd| per joint group after the limp ticks (p95, p99, max, share over 10 / 15 / 20
    rad/s) and the fastest commanded target speed between two policy-driven ticks, for the logs and the eval JSON.
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from threepp.rl import GpuSim, VecTask
from spot_deploy import (build_spot, default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, LIM, STIFF_GAINS, HIP_X,
                         HIP_Y, HY_Y, KN)
from spot_terrain_env import quat_rotate_inverse, up_z, _flat_ground, CONTROL_HZ, DT, SUBSTEPS, SPACING, ACT_DIM, N_SCAN

OBS_DIM = 48 + 2 + 1 + N_SCAN           # = 96, the walking layout
EPISODE_S = 6.0
LIMP_TICKS = 25                         # 0.5 s: a 0.6 m drop lands in 0.35 s
STAND_FRAC = 0.18
N_LEVELS = 4
STAND_Z = 0.46                          # settled stance base height (CPU and GPU probes: 0.4603 / 0.4604 m)
SUCCESS_UP, SUCCESS_Z, SUCCESS_QERR, SUCCESS_HOLD = 0.9, 0.40, 0.25, 50
LEGS = ("fl", "fr", "hl", "hr")
KIND_NAMES = ("stand", "tilt", "side", "back", "random")
STAND, TILT, SIDE, BACK, RANDOM = range(len(KIND_NAMES))
# reward weights: standing still in the stance earns W_UP + W_HEIGHT + W_POSE + W_STILL = 3.0 per step, lying on the back
# 0, on a side 0.25 (the uprightness term alone); the penalties together stay under ~0.4 per step at full effort
W_UP, W_HEIGHT, W_POSE, W_STILL = 1.0, 1.0, 0.5, 0.5
W_TORQUE, W_ARATE, W_LIMIT, W_IMPACT = 0.1, 0.005, 0.2, 0.05
CONTACT_EVERY = 5                       # self-contact is measured every this many ticks (and counted x this)
SPAWN_BLEND = (1.0, 0.8, 0.6, 0.45, 0.3, 0.15)   # redraw i pulls the joint draw this far from the stance
TAU_BINS, TAU_W = 48, 2.5               # |torque| histogram 0..120 N·m per joint group, for p95
OBS_LAST_ACT = slice(36, 48)            # last_act in the 96-d layout: lin_b, ang_b, proj_g, cmd (3 each), qpos, jv (12)
DOWN_UP = 0.5                           # "not upright" for w_down and the gave-up metric (= spot_slam's RECOVER_UP)
EFFORT_UP_LO, EFFORT_UP_SPAN = 0.3, 0.6  # fallen_effort_free: effort penalties x clamp((up_z - 0.3) / 0.6, 0, 1)
GIVEUP_JV, GIVEUP_TICKS = 0.25, 50      # gave up: not upright and mean |joint velocity| < 0.25 rad/s for 1 s in a row
QD_BINS, QD_W = 400, 0.25               # |joint velocity| histogram 0..100 rad/s per joint group, for p95 / p99
QD_OVER = (10.0, 15.0, 20.0)            # shares of live joint-speed samples above these (rad/s)
JOINT_GROUPS = ("hx", "hy", "kn")
CONFIG = {"task": "recovery", "control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "obs_dim": OBS_DIM,
          "act_dim": ACT_DIM, "episode_s": EPISODE_S, "limp_ticks": LIMP_TICKS, "stand_frac": STAND_FRAC,
          "n_levels": N_LEVELS, "stand_z": STAND_Z,
          "success": {"up": SUCCESS_UP, "z": SUCCESS_Z, "qerr": SUCCESS_QERR, "hold_ticks": SUCCESS_HOLD},
          "weights": {"up": W_UP, "height": W_HEIGHT, "pose": W_POSE, "still": W_STILL, "torque": W_TORQUE,
                      "arate": W_ARATE, "limit": W_LIMIT, "impact": W_IMPACT},
          "stiff_gains": {k: list(v) for k, v in STIFF_GAINS.items()}, "stand_mode": True}


# ---- quaternions and leg kinematics (torch, batched) --------------------------------------------------------------
def quat_axis_angle(axis, ang):
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.cat([axis * torch.sin(ang / 2)[:, None], torch.cos(ang / 2)[:, None]], dim=1)


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(1)
    bx, by, bz, bw = b.unbind(1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], dim=1)


def quat_rot(q, v):
    """rotate [n, m, 3] vectors by [n, 4] (x, y, z, w) quaternions"""
    qv, w = q[:, None, :3].expand_as(v), q[:, None, 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def leg_segments(q_add, samples=7):
    """Body-frame collider samples of every leg from add-order joints [n, 12] -> points [n, 12, S, 3] (hips 0-3, upper legs
    4-7, lower legs 8-11, legs fl/fr/hl/hr: the GPU link order minus the base) and radii [12]. The chain is build_spot's:
    hip about x at (sx HIP_X, sy HIP_Y, 0), thigh about y at +sy HY_Y, knee at +KN, capsules 0.06 / 0.30 / 0.30 long."""
    n, dev = q_add.shape[0], q_add.device
    q = q_add.view(n, 4, 3)
    sx = q.new_tensor([1.0, 1.0, -1.0, -1.0]); sy = q.new_tensor([1.0, -1.0, 1.0, -1.0])
    hx, hy, kn = q[..., 0], q[..., 1], q[..., 2]
    ch, sh = torch.cos(hx), torch.sin(hx)

    def rx(x, y, z):                      # the hip rotation about body x, about the hip axis at (., sy HIP_Y, 0)
        return torch.stack([sx * HIP_X + x, sy * HIP_Y + y * ch - z * sh, y * sh + z * ch], dim=-1)

    zero = torch.zeros_like(hy)
    j_hx = rx(zero, zero, zero)
    j_hy = rx(zero, sy * HY_Y, zero)
    kx = KN[0] * torch.cos(hy) + KN[2] * torch.sin(hy); kz = -KN[0] * torch.sin(hy) + KN[2] * torch.cos(hy)
    j_kn = rx(kx, sy * HY_Y, kz)
    fx = kx - 0.34 * torch.sin(hy + kn); fz = kz - 0.34 * torch.cos(hy + kn)
    j_ft = rx(fx, sy * HY_Y, fz)
    u = torch.linspace(-1.0, 1.0, samples, device=dev)[:, None]

    def caps(a, b, half):                 # [n,4,3] joint ends -> [n,4,S,3] samples along the capsule's cylinder
        c, d = 0.5 * (a + b), b - a
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return c[:, :, None, :] + half * u[None, None] * d[:, :, None, :]

    pts = torch.cat([caps(j_hx, j_hy, 0.03), caps(j_hy, j_kn, 0.15), caps(j_kn, j_ft, 0.15)], dim=1)
    rad = torch.tensor([0.045] * 8 + [0.028] * 4, device=dev)
    return pts, rad


_BASE_HALF = (0.35, 0.09, 0.095)
_PAIRS = None


def _pairs(dev):
    """Non-adjacent collider pairs among the 12 leg segments: every segment of one leg against every segment of another,
    plus a leg's hip against its own lower leg. (Base against upper and lower legs is checked separately.)"""
    global _PAIRS
    if _PAIRS is None or _PAIRS[0].device != torch.device(dev):
        a, b = [], []
        for i in range(12):
            for j in range(i + 1, 12):
                li, lj = i % 4, j % 4
                if li != lj or (i // 4 == 0 and j // 4 == 2):
                    a.append(i); b.append(j)
        _PAIRS = (torch.tensor(a, device=dev), torch.tensor(b, device=dev))
    return _PAIRS


def self_clearance(pts, rad, base_rot=None, base_pos=None):
    """Smallest surface distance between non-adjacent colliders -> [n]. pts [n, 12, S, 3] in the base frame, or in the world
    frame with the base pose (base_rot [n,4], base_pos [n,3]) to measure against the base box."""
    n = pts.shape[0]
    pa, pb = _pairs(pts.device)
    A, B = pts[:, pa], pts[:, pb]                                           # [n, P, S, 3]
    d = (A[:, :, :, None, :] - B[:, :, None, :, :]).norm(dim=-1).amin(dim=(2, 3)) - rad[pa] - rad[pb]
    legs = pts[:, 4:].reshape(n, -1, 3)                                     # upper and lower legs against the base box
    if base_rot is not None:
        inv = base_rot * base_rot.new_tensor([-1.0, -1.0, -1.0, 1.0])
        legs = quat_rot(inv, legs - base_pos[:, None, :])
    half = legs.new_tensor(_BASE_HALF)
    box = (legs.abs() - half).clamp_min(0.0).norm(dim=-1).view(n, 8, -1).amin(dim=2) - rad[4:]
    return torch.minimum(d.amin(dim=1), box.amin(dim=1))


def lowest_point(quat, q_add):
    """World-z of the robot's lowest surface point relative to the base origin -> [n]."""
    pts, rad = leg_segments(q_add, samples=3)
    n = q_add.shape[0]
    c = torch.tensor([[x, y, z] for x in (-0.35, 0.35) for y in (-0.09, 0.09) for z in (-0.095, 0.095)],
                     device=q_add.device)
    legs = quat_rot(quat, pts.reshape(n, -1, 3))[..., 2] - rad.repeat_interleave(pts.shape[2])[None]
    return torch.minimum(quat_rot(quat, c.expand(n, 8, 3))[..., 2].amin(dim=1), legs.amin(dim=1))


class SpotRecoveryEnv(VecTask):
    control_hz = CONTROL_HZ
    episode_s = EPISODE_S
    act_dim = ACT_DIM
    control = "drive"
    substeps = SUBSTEPS
    settle_steps = 0
    clip_actions = None

    def __init__(self, num_envs=2048, device="cuda", seed=0, self_collision=True, stand_frac=STAND_FRAC,
                 init_level=0, freeze_level=False, count_episodes=None, limp_ticks=LIMP_TICKS, action_scale=ACTION_SCALE,
                 target_clip=False, w_prog=0.0, prog_gamma=0.99, up_slope=0.0, hard_frac=0.0, fallen_effort_free=False,
                 w_down=0.0, qd_max=None, w_qd_excess=0.0):
        """self_collision: legs collide with the body and each other (the point of this env; False is the walking plant).
        init_level int or [K]; freeze_level holds it (evaluation). count_episodes E: the counters take each env's first
        E episodes only (None = all). The wave 2 knobs (module docstring) default to the pilot: action_scale 0.2 without
        a target clip, no progress shaping (prog_gamma must be the trainer's PPO gamma), the pilot's upright term, no hard
        draws. fallen_effort_free: the torque / action-rate / joint-limit penalties are scaled by
        clamp((up_z - 0.3) / 0.6, 0, 1), so trying costs nothing while down (the impact penalty stays whole). w_down C:
        -C per live step while up_z < DOWN_UP, so lying still is strictly worse than trying. qd_max V (rad/s, None or 0 =
        off): the drive targets are rate-limited to V; w_qd_excess W: -W mean((|qd| - V)_+^2), needs qd_max."""
        self.self_collision = bool(self_collision)
        self.action_scale, self.target_clip = float(action_scale), bool(target_clip)
        self.w_prog, self.prog_gamma, self.up_slope = float(w_prog), float(prog_gamma), float(up_slope)
        self.hard_frac, self.fallen_effort_free, self.w_down = float(hard_frac), bool(fallen_effort_free), float(w_down)
        self.qd_max = float(qd_max) if qd_max else None
        self.w_qd_excess = float(w_qd_excess)
        if self.w_qd_excess > 0 and self.qd_max is None:
            raise ValueError("w_qd_excess penalizes joint speed above qd_max: set qd_max too")

        class _Robot:
            def __init__(self_, world, i):
                self_.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * SPACING), gains=STIFF_GAINS,
                                          drive_limits_are_forces=True, self_collision=self.self_collision)

        super().__init__(num_envs, lambda world, i: _Robot(world, i), gravity=(0.0, 0.0, -9.81), spacing=SPACING,
                         device=device, seed=seed, read_root=True, read_links=True,
                         build_world=lambda world: _flat_ground(world, num_envs, SPACING, length=40.0, x_center=0.0))
        dev = self.device
        K = num_envs
        self.default_q = torch.from_numpy(default_q).to(dev)
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)
        self.stance = self.default_q[self.a2i]                               # add order
        self.q_lo = torch.tensor([LIM[j][0] for _ in LEGS for j in ("hx", "hy", "kn")], device=dev)
        self.q_hi = torch.tensor([LIM[j][1] for _ in LEGS for j in ("hx", "hy", "kn")], device=dev)
        self.kp = torch.tensor([STIFF_GAINS[j][0] for _ in LEGS for j in ("hx", "hy", "kn")], device=dev)
        self.kd = torch.tensor([STIFF_GAINS[j][1] for _ in LEGS for j in ("hx", "hy", "kn")], device=dev)
        self.cap = torch.tensor([STIFF_GAINS[j][2] for _ in LEGS for j in ("hx", "hy", "kn")], device=dev)
        self.lane_y = torch.arange(K, device=dev, dtype=torch.float32) * SPACING
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        self.stand_frac, self.freeze_level = float(stand_frac), bool(freeze_level)
        self.limp_ticks = int(limp_ticks)
        lv = np.asarray(init_level, np.int64)
        self.level = (torch.full((K,), int(lv), dtype=torch.long, device=dev) if lv.ndim == 0 else
                      torch.from_numpy(lv.reshape(-1)).to(dev))
        self._g = torch.Generator(device=dev).manual_seed(int(seed) + 2718281)
        # per-episode state
        self.last_act = self.env_state((ACT_DIM,))
        self.prev_act = self.env_state((ACT_DIM,))
        self.kind = self.env_state((), init=0, dtype=torch.long)
        self.succ_run = self.env_state((), init=0, dtype=torch.long)
        self.succ_tick = self.env_state((), init=-1, dtype=torch.long)       # tick the first 1 s hold began (-1: none)
        self.contact_ticks = self.env_state((), init=0, dtype=torch.long)
        self.tau_peak = self.env_state((3,))
        self.still_run = self.env_state((), init=0, dtype=torch.long)       # consecutive down-and-motionless live ticks
        self.gave_up = self.env_state((), init=False, dtype=torch.bool)
        self.hard = torch.zeros(K, dtype=torch.bool, device=dev)            # this episode is a hard_frac L3 draw
        self._phi = torch.zeros(K, device=dev)                              # uprightness potential now / a tick ago
        self._phi_prev = torch.zeros(K, device=dev)
        self.up = torch.ones(K, device=dev)
        self._tgt = self.stance.expand(K, 12).clone()
        self._tau = torch.zeros(K, 12, device=dev)
        self._last_obs = torch.zeros(K, OBS_DIM, device=dev)
        self._blowup = torch.zeros(K, dtype=torch.bool, device=dev)
        self._tick = 0                                                      # control ticks stepped (host counter)
        self.count_episodes = None if count_episodes is None else int(count_episodes)
        self.ep_index = torch.zeros(K, dtype=torch.long, device=dev)
        # counters per (level, start kind): episodes, success by 3 s, by 6 s, recovery-time sum, contact ticks, blow-ups,
        # gave up. A hard_frac draw counts under L3, the level it started at.
        self._stats = torch.zeros(N_LEVELS, len(KIND_NAMES), 7, dtype=torch.float64, device=dev)
        self._hard = torch.zeros(4, dtype=torch.float64, device=dev)     # hard draws, of them above the env's level, eps, successes
        self._prog = torch.zeros(4, dtype=torch.float64, device=dev)     # live ticks, sum and sum |.| of the progress term, sum w_down
        self._qd_hist = torch.zeros(3, QD_BINS, dtype=torch.float64, device=dev)   # live |qd| samples per joint group
        self._qd_peak = torch.zeros(3, device=dev)
        self._cmd_qd = torch.zeros(3, device=dev)                        # fastest commanded target speed per group, rad/s
        self._qdx = torch.zeros(2, dtype=torch.float64, device=dev)      # live ticks, sum of the qd_excess term
        self._tau_hist = torch.zeros(3, TAU_BINS, dtype=torch.float64, device=dev)
        self._moves = torch.zeros(N_LEVELS, 3, dtype=torch.float64, device=dev)   # episodes, promoted, demoted
        self._spawn = torch.zeros(3, dtype=torch.float64, device=dev)            # resets, redraws, fell back to stance

    # ---- resets -----------------------------------------------------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self._g)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self._g)

    def _sample_joints(self, n, uniform):
        """uniform [n] bool: uniform within the limits, else the stance + N(0, 0.25) clipped. With self_collision, a draw
        whose colliders interpenetrate is redrawn, each redraw pulled further toward the stance (SPAWN_BLEND), and only
        the last resort is the stance itself. Measured 2026-09-13 (K=2048, level 3): with plain redraws at a 1 cm margin
        43% of fallen spawns fell back to the stance, because most uniform joint vectors fold a leg into the body or
        another leg (500 of 512 random extreme poses touch, probe a)."""
        def draw(s):
            u = self.q_lo + self._rand(n, 12) * (self.q_hi - self.q_lo)
            near = (self.stance + 0.25 * self._randn(n, 12)).clamp(self.q_lo, self.q_hi)
            return self.stance + s * (torch.where(uniform[:, None], u, near) - self.stance)

        q = draw(1.0)
        if not self.self_collision:
            return q
        redraws = 0
        for s in SPAWN_BLEND:
            bad = self_clearance(*leg_segments(q, samples=11)) < 0.0
            if not bool(bad.any()):
                break
            redraws += int(bad.sum())
            q = torch.where(bad[:, None], draw(s), q)
        bad = self_clearance(*leg_segments(q, samples=11)) < 0.0
        q = torch.where(bad[:, None], self.stance.expand(n, 12), q)
        self._spawn += torch.tensor([0.0, float(redraws), float(bad.sum())], device=self.device, dtype=torch.float64)
        return q

    def on_reset(self, idx):
        n, dev = idx.numel(), self.device
        lvl = self.level[idx]
        u = self._rand(n)
        stand = self._rand(n) < self.stand_frac
        if self.hard_frac > 0:                  # drawn only when on, so the pilot's random stream is untouched
            hard = ~stand & (self._rand(n) < self.hard_frac)
            self._hard[:2] += torch.stack([hard.sum(), (hard & (lvl < N_LEVELS - 1)).sum()]).double()
            lvl = torch.where(hard, N_LEVELS - 1, lvl)
            self.hard[idx] = hard
        # kind by level: L0 / L1 tilt, L2 side or tilt 60-120, L3 back / side / random
        kind = torch.where(lvl <= 1, TILT, torch.where(lvl == 2, torch.where(u < 0.5, SIDE, TILT),
                           torch.where(u < 0.4, BACK, torch.where(u < 0.8, SIDE, RANDOM))))
        kind = torch.where(stand, STAND, kind)
        # tilt magnitude (TILT / STAND), about a random horizontal axis
        lo = torch.where(lvl == 0, 0.0, torch.where(lvl == 1, math.radians(25), math.radians(60)))
        hi = torch.where(lvl == 0, math.radians(25), torch.where(lvl == 1, math.radians(60), math.radians(120)))
        ang = torch.where(stand, self._rand(n) * math.radians(15), lo + self._rand(n) * (hi - lo))
        a = self._rand(n) * 2 * math.pi
        haxis = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], dim=1)
        tilt = quat_axis_angle(haxis, ang)
        xaxis = torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(n, 3)
        jitter = math.radians(20) * (2 * self._rand(n) - 1)
        side = quat_axis_angle(xaxis, torch.where(self._rand(n) < 0.5, 1.0, -1.0) * math.pi / 2 + jitter)
        back = quat_axis_angle(xaxis, math.pi + jitter)
        rnd = self._randn(n, 4); rnd = rnd / rnd.norm(dim=1, keepdim=True)
        k = kind[:, None]
        q_or = torch.where(k == SIDE, side, torch.where(k == BACK, back, torch.where(k == RANDOM, rnd, tilt)))
        yaw = quat_axis_angle(torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3), self._rand(n) * 2 * math.pi)
        quat = quat_mul(yaw, q_or)
        joints = self._sample_joints(n, uniform=(kind >= SIDE))
        joints = torch.where((kind == STAND)[:, None], (self.stance + 0.05 * self._randn(n, 12)).clamp(self.q_lo, self.q_hi),
                             joints)
        drop = torch.where(kind == STAND, 0.02 * self._rand(n),
                           torch.where(kind == TILT, 0.1 + 0.2 * self._rand(n), 0.3 + 0.3 * self._rand(n)))
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :4] = quat
        pose[:, 5] = self.lane_y[idx]
        pose[:, 6] = drop - lowest_point(quat, joints)
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, joints, torch.zeros(n, self.sim.dof, device=dev))
        self.kind[idx] = kind
        self._blowup[idx] = False
        self._spawn[0] += n
        if self.qd_max and self.limp_ticks == 0:
            # no limp ticks to hand the limiter the landed joints: start it at the spawn joints
            self.last_act[idx] = (joints[:, self.i2a] - self.default_q) / self.action_scale
        if self.w_prog > 0 or self.up_slope > 0:
            # the new episode's potential, so its first progress term does not difference against the last episode's end
            self._phi[idx] = self._potential(quat)

    def _potential(self, q):
        """phi = (1 - g_z) / 2, g_z = projected gravity's body-frame z: 0 on the back, 0.5 on a side, 1 upright."""
        return 0.5 * (1.0 - quat_rotate_inverse(q, self.grav.expand(q.shape[0], 3))[:, 2])

    # ---- step -------------------------------------------------------------------------------------------------
    def act(self, a):
        limp = (self.steps < self.limp_ticks)[:, None]
        # limp: the drive target follows the joints, so the PD spring does nothing and the robot lands as it falls
        a_limp = (self.sim.joint_pos[:, self.i2a] - self.default_q) / self.action_scale
        if self.qd_max:
            # the rate limit, in action units: the target may move qd_max * dt per tick from the last tick's target, whose
            # action last_act holds (the limp ones included, so the limiter starts from where the joints landed)
            da = self.qd_max * self.dt / self.action_scale
            a = self.last_act + (a - self.last_act).clamp(-da, da)
        a = torch.where(limp, a_limp, a)
        self.prev_act.copy_(self.last_act)
        self.last_act.copy_(a)
        tgt = (self.default_q + self.action_scale * a)[:, self.a2i]
        if self.target_clip:
            # the policy's targets stay inside the joint limits (at S = 1 an action of 3 would ask hy for 2.7 rad past the
            # stance); the limp ticks keep following the joints, which a landing can push up to 0.056 rad past a limit
            tgt = torch.where(limp, tgt, torch.maximum(torch.minimum(tgt, self.q_hi), self.q_lo))
        # the fastest commanded target speed per joint group, between two policy-driven ticks (a limp target follows the
        # falling joints and a reset jumps it): a metric only, read from the targets themselves
        cmd = ((tgt - self._tgt).abs() / self.dt).view(-1, 4, 3).amax(dim=1) * (self.steps > self.limp_ticks)[:, None]
        self._cmd_qd.copy_(torch.maximum(self._cmd_qd, cmd.amax(dim=0)))
        self._tgt.copy_(tgt)
        return tgt

    def on_step(self, s):
        q = self.sim.root_quat
        self.up = up_z(q)
        self._lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        self._ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        jp, jv = self.sim.joint_pos, self.sim.joint_vel
        tau = self.kp * (self._tgt - jp) - self.kd * jv
        self._tau.copy_(torch.maximum(torch.minimum(tau, self.cap), -self.cap))
        z = self.sim.root_position[:, 2]
        qerr = (jp - self.stance).abs().mean(dim=1)
        ok = (self.up > SUCCESS_UP) & (z > SUCCESS_Z) & (qerr < SUCCESS_QERR) & (self.steps > self.limp_ticks)
        self.succ_run.copy_(torch.where(ok, self.succ_run + 1, 0))
        first = (self.succ_tick < 0) & (self.succ_run >= SUCCESS_HOLD)
        self.succ_tick.copy_(torch.where(first, self.steps - (SUCCESS_HOLD - 1), self.succ_tick))
        if self.w_prog > 0 or self.up_slope > 0:
            self._phi_prev.copy_(self._phi)
            self._phi.copy_(self._potential(q))
        # gave up: down and nearly motionless for GIVEUP_TICKS in a row after the limp settle ("it kinda just gives up",
        # Lars watching the pilot policy in spot_slam, 2026-09-13). A metric only: nothing here feeds obs or reward.
        still = (self.up < DOWN_UP) & (jv.abs().mean(dim=1) < GIVEUP_JV) & (self.steps > self.limp_ticks)
        self.still_run.copy_(torch.where(still, self.still_run + 1, 0))
        self.gave_up.copy_(self.gave_up | (self.still_run >= GIVEUP_TICKS))
        # joint speed after the limp ticks, per joint group (the add order is per leg: hx, hy, kn); a metric only
        live_q = self.steps > self.limp_ticks
        if self.count_episodes:
            live_q = live_q & (self.ep_index < self.count_episodes)
        jva = torch.nan_to_num(jv.abs(), nan=0.0, posinf=1e3).view(-1, 4, 3)
        qbins = torch.clamp((jva / QD_W).long(), 0, QD_BINS - 1).permute(2, 0, 1).reshape(3, -1)
        self._qd_hist.scatter_add_(1, qbins, live_q.double()[:, None].expand(-1, 4).reshape(1, -1).expand(3, -1).contiguous())
        self._qd_peak.copy_(torch.maximum(self._qd_peak, (jva.amax(dim=1) * live_q[:, None]).amax(dim=0)))
        grp = self._tau.abs().view(-1, 4, 3).amax(dim=1)                    # [K, 3] hx hy kn
        self.tau_peak.copy_(torch.maximum(self.tau_peak, grp))
        # [joint group, env x leg] bins: the add order is per leg (hx, hy, kn), so the group is the last axis
        bins = torch.clamp((self._tau.abs() / TAU_W).long(), 0, TAU_BINS - 1).view(-1, 4, 3).permute(2, 0, 1).reshape(3, -1)
        self._tau_hist.scatter_add_(1, bins, torch.ones_like(bins, dtype=torch.float64))
        self._tick += 1
        if self._tick % CONTACT_EVERY == 0:
            lp = self.sim.link_pose
            pts = self._link_segments(lp)
            clr = self_clearance(pts, self._seg_rad, base_rot=lp[:, 0, 0:4], base_pos=lp[:, 0, 4:7])
            self.contact_ticks += (clr < 0.01).long() * CONTACT_EVERY
        rp, rv = self.sim.root_pose, self.sim.root_linvel
        self._blowup.copy_(~torch.isfinite(rp).all(dim=1) | ~torch.isfinite(jp).all(dim=1) | (rv.norm(dim=1) > 30.0)
                           | (jv.abs().amax(dim=1) > 150.0) | (rp[:, 6] > 5.0))

    def _link_segments(self, lp):
        """World-frame collider samples from the GPU link poses (links 1-12 = hips, upper, lower legs; capsule axis +Y)."""
        if not hasattr(self, "_seg_half"):
            self._seg_half = torch.tensor([0.03] * 4 + [0.15] * 8, device=self.device)
            self._seg_rad = torch.tensor([0.045] * 8 + [0.028] * 4, device=self.device)
            self._seg_u = torch.linspace(-1.0, 1.0, 7, device=self.device)
        ll = lp[:, 1:13]
        x, y, z, w = ll[..., 0], ll[..., 1], ll[..., 2], ll[..., 3]
        ax = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
        return ll[..., None, 4:7] + (self._seg_half[None, :, None] * self._seg_u[None, None, :])[..., None] * ax[:, :, None, :]

    def terminated(self, s):
        return self._blowup

    def reward_terms(self, s, a):
        live = (self.steps > self.limp_ticks).float()
        up = self.up
        z = self.sim.root_position[:, 2]
        gate = ((up - 0.5) / 0.4).clamp(0.0, 1.0)
        jp = self.sim.joint_pos
        stood = ((up > SUCCESS_UP) & (z > SUCCESS_Z)).float()
        near = torch.minimum(jp - self.q_lo, self.q_hi - jp)
        lv = self.sim.link_linvel
        lz = self.sim.link_pose[:, :9, 6]                                   # base, hips, upper legs
        hit = ((lv[:, :9].norm(dim=2) - 1.5).clamp_min(0.0) ** 2 * (lz < 0.2).float()).sum(dim=1)
        if self.up_slope > 0:
            c, phi = self.up_slope, self._phi
            upright = W_UP * ((1.0 - c) * phi * phi + c * phi)
        else:
            upright = W_UP * ((up + 1.0) * 0.5) ** 2       # the pilot's: zero slope on the back
        # effort gate: 1 upright, 0 below up_z 0.3 (fallen_effort_free), else no gate (the pilot)
        eff = ((up - EFFORT_UP_LO) / EFFORT_UP_SPAN).clamp(0.0, 1.0) if self.fallen_effort_free else 1.0
        terms = {
            "upright": upright,
            "height": W_HEIGHT * gate * (1.0 - (z - STAND_Z).abs() / 0.25).clamp(0.0, 1.0),
            "pose": W_POSE * gate * torch.exp(-(jp - self.stance).pow(2).mean(dim=1) / 0.1),
            "still": W_STILL * stood * torch.exp(-(self._lin_b.pow(2).sum(dim=1) + 0.25 * self._ang_b.pow(2).sum(dim=1)) / 0.1),
            "torque": -W_TORQUE * eff * (self._tau / self.cap).pow(2).mean(dim=1),
            "arate": -W_ARATE * eff * (a - self.prev_act).pow(2).mean(dim=1),
            "limit": -W_LIMIT * eff * ((0.1 - near).clamp_min(0.0) / 0.1).mean(dim=1),
            "impact": -W_IMPACT * hit,
        }
        # extra terms only when on: a zero term would still change the stacked sum's kernel and the stats line
        if self.w_prog > 0:
            # potential-based (Ng 1999) with the PPO gamma: a roll back -> side -> upright telescopes to +W, falling
            # over to -W, and standing pays a (1 - gamma) W drain per step (0.05 at W 5 against 3.0 for standing)
            terms["progress"] = self.w_prog * (self.prog_gamma * self._phi - self._phi_prev)
        if self.w_down > 0:
            terms["down"] = -self.w_down * (up < DOWN_UP).float()
        if self.w_qd_excess > 0:
            # on the MEASURED joints, so a whip or a bounce that no target limit stops also costs
            terms["qd_excess"] = -self.w_qd_excess * (self.sim.joint_vel.abs() - self.qd_max).clamp_min(0.0).pow(2).mean(dim=1)
        out = {k: v * live for k, v in terms.items()}
        if self.w_prog > 0 or self.w_down > 0:
            pr = out.get("progress", torch.zeros_like(live))
            self._prog += torch.stack([live.sum(), pr.sum(), pr.abs().sum(),
                                       out.get("down", torch.zeros_like(live)).sum()]).double()
        if self.w_qd_excess > 0:
            self._qdx += torch.stack([live.sum(), out["qd_excess"].sum()]).double()
        return out

    def observe(self, s):
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos = s.joint_pos[:, self.i2a] - self.default_q
        jv = s.joint_vel[:, self.i2a]
        zeros3 = torch.zeros(self.K, 3, device=self.device)
        clk = torch.zeros(self.K, 2, device=self.device)                    # the stand sentinel: command exactly zero
        base_above = s.root_pos[:, 2:3]                                     # flat ground at z = 0
        ahead = torch.zeros(self.K, N_SCAN, device=self.device)             # the scan of flat ground
        obs = torch.cat([lin_b, ang_b, proj_g, zeros3, qpos, jv, self.last_act, clk, base_above, ahead], dim=1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_obs.copy_(obs)
        return obs

    # ---- episode ends -----------------------------------------------------------------------------------------
    def on_done(self, idx):
        lvl, kind = self.level[idx], self.kind[idx]
        st = self.succ_tick[idx]
        s3 = (st >= 0) & (st + SUCCESS_HOLD - 1 <= int(3.0 * CONTROL_HZ))
        s6 = st >= 0
        rt = torch.where(s6, st.double() * DT, 0.0)
        blow = self._blowup[idx]
        m = torch.stack([torch.ones_like(s6), s3, s6, rt, self.contact_ticks[idx], blow, self.gave_up[idx]], dim=1).double()
        if self.count_episodes:
            m = m * (self.ep_index[idx] < self.count_episodes).double()[:, None]
        hard = self.hard[idx]
        lvl_s = torch.where(hard, N_LEVELS - 1, lvl) if self.hard_frac > 0 else lvl
        self._stats.view(-1, m.shape[1]).index_add_(0, lvl_s * len(KIND_NAMES) + kind, m)
        if self.hard_frac > 0:
            self._hard[2:] += torch.stack([hard.sum(), (hard & s6).sum()]).double()
        if not self.freeze_level:
            fallen = kind != STAND
            if self.hard_frac > 0:
                fallen = fallen & ~hard         # a forced L3 draw says nothing about the env's own level
            up_ = fallen & s6
            dn = fallen & ~s6
            self._moves.index_add_(0, lvl, torch.stack([fallen, up_ & (lvl < N_LEVELS - 1), dn & (lvl > 0)], 1).double())
            self.level[idx] = (lvl + up_.long() - dn.long()).clamp(0, N_LEVELS - 1)
        self.ep_index[idx] += 1

    def recovery_stats(self, reset=True):
        """Counters since the last reset, one host read: per level and start kind episodes / success by 3 s and 6 s / mean
        recovery time / self-contact ticks per episode / blow-ups; p95 commanded torque per joint group; level histogram;
        promotions; spawn redraws."""
        s = self._stats.cpu().numpy()
        h = self._tau_hist.cpu().numpy()
        out = {"by_level_kind": {}, "total": {}}
        div = lambda a, b: (a / b) if b else None
        for L in range(N_LEVELS):
            for k, name in enumerate(KIND_NAMES):
                e = s[L, k, 0]
                if e:
                    out["by_level_kind"][f"L{L}_{name}"] = {"episodes": int(e), "success_3s": s[L, k, 1] / e,
                                                          "success_6s": s[L, k, 2] / e,
                                                          "recovery_s": div(s[L, k, 3], s[L, k, 2]),
                                                          "contact_ticks_per_ep": s[L, k, 4] / e, "blowups": int(s[L, k, 5]),
                                                          "gave_up": s[L, k, 6] / e}
        fallen = s[:, 1:].sum(axis=(0, 1)); standing = s[:, 0].sum(axis=0)
        for tag, v in (("fallen", fallen), ("stand", standing)):
            out["total"][tag] = {"episodes": int(v[0]), "success_3s": div(v[1], v[0]), "success_6s": div(v[2], v[0]),
                                 "recovery_s": div(v[3], v[2]), "contact_ticks_per_ep": div(v[4], v[0]), "blowups": int(v[5]),
                                 "gave_up": div(v[6], v[0])}
        # gave up per L3 start kind, whatever level the episode was counted under in the training mix (hard draws are L3)
        out["gave_up_L3"] = {name: div(s[N_LEVELS - 1, k, 6], s[N_LEVELS - 1, k, 0]) for k, name in enumerate(KIND_NAMES)
                             if k in (BACK, SIDE, RANDOM)}
        if self.hard_frac > 0:
            hd = self._hard.cpu().numpy()
            out["hard"] = {"draws": int(hd[0]), "above_level": int(hd[1]), "episodes": int(hd[2]),
                           "success_6s": div(hd[3], hd[2])}
        if self.w_prog > 0 or self.w_down > 0:
            pg = self._prog.cpu().numpy()
            out["shaping"] = {"live_ticks": int(pg[0]), "progress_mean": div(pg[1], pg[0]),
                              "progress_mean_abs": div(pg[2], pg[0]), "down_mean": div(pg[3], pg[0])}
        p95 = []
        for g in range(3):
            c = np.cumsum(h[g]); tot = c[-1]
            p95.append(float((np.searchsorted(c, 0.95 * tot) + 1) * TAU_W) if tot else None)
        out["tau_p95_hx_hy_kn"] = p95
        out["levels"] = np.bincount(self.level.cpu().numpy(), minlength=N_LEVELS).tolist()
        mv = self._moves.cpu().numpy()
        out["moves_by_level"] = {f"L{L}": {"fallen_eps": int(mv[L, 0]), "promoted": int(mv[L, 1]), "demoted": int(mv[L, 2])}
                                 for L in range(N_LEVELS) if mv[L, 0]}
        sp = self._spawn.cpu().numpy()
        out["spawn"] = {"resets": int(sp[0]), "self_penetration_redraws": int(sp[1]), "fell_back_to_stance": int(sp[2])}
        hq, pk = self._qd_hist.cpu().numpy(), self._qd_peak.cpu().numpy()
        qd = {}
        for g, name in enumerate(JOINT_GROUPS):
            c = np.cumsum(hq[g]); tot = c[-1]
            pct = [float((np.searchsorted(c, p * tot) + 1) * QD_W) if tot else None for p in (0.95, 0.99)]
            qd[name] = {"samples": int(tot), "p95": pct[0], "p99": pct[1], "max": float(pk[g]) if tot else None,
                        **{f"over{int(v)}": (float(hq[g, int(round(v / QD_W)):].sum() / tot) if tot else None)
                           for v in QD_OVER}}
        out["qd"] = qd
        out["cmd_qd_max_hx_hy_kn"] = [float(v) for v in self._cmd_qd.cpu().numpy()]
        if self.w_qd_excess > 0:
            qx = self._qdx.cpu().numpy()
            out["qd_excess"] = {"live_ticks": int(qx[0]), "mean": div(qx[1], qx[0])}
        if reset:
            self._stats.zero_(); self._tau_hist.zero_(); self._moves.zero_(); self._hard.zero_(); self._prog.zero_()
            self._qd_hist.zero_(); self._qd_peak.zero_(); self._cmd_qd.zero_(); self._qdx.zero_()
        return out

    def knob_config(self):
        """The wave 2 and 3 knobs as the checkpoint meta records them (a meta without them is the pilot; without qd_max,
        wave 2)."""
        return {"action_scale": self.action_scale, "target_clip": self.target_clip, "w_prog": self.w_prog,
                "prog_gamma": self.prog_gamma, "up_slope": self.up_slope, "hard_frac": self.hard_frac,
                "fallen_effort_free": self.fallen_effort_free, "w_down": self.w_down, "down_up": DOWN_UP,
                "effort_up": [EFFORT_UP_LO, EFFORT_UP_SPAN], "giveup": {"jv": GIVEUP_JV, "ticks": GIVEUP_TICKS},
                "qd_max": self.qd_max, "w_qd_excess": self.w_qd_excess}

    def config(self):
        return {**super().config(), **CONFIG, "self_collision": self.self_collision, **self.knob_config()}


if __name__ == "__main__":
    K = int(os.environ.get("K", "64"))
    env = SpotRecoveryEnv(num_envs=K)
    obs = env.reset()
    assert obs.shape == (K, OBS_DIM)
    for _ in range(320):
        obs, rew, done, _, _ = env.step(torch.zeros(K, ACT_DIM, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(env.stats_line())
    print(env.recovery_stats())
    print("SPOT RECOVERY ENV SELFTEST: PASS")
