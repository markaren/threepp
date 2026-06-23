"""Spot locomotion FROM SCRATCH on GpuSim — no frozen Isaac walker, no CPG.

The policy outputs 12 joint-position targets (offset on the default stance, PD-driven). It learns a
diagonal TROT via a gait-phase clock: a periodic phase advances each step, the obs carries (sin,cos)
of it, and the reward pays each foot for matching a desired trot CONTACT SCHEDULE (pair FL+HR down for
the first half-cycle then lifted, pair FR+HL opposite). This PRESCRIBES the trot, where penalty-only
shaping just produced a shuffle. Velocity tracking + a heading-hold command keep it walking straight;
grippy restitution-0 feet + per-env friction randomization (the material API) make it transfer to the
CPU deploy. K Spots train in one direct-GPU PhysX scene.

  obs (50): base lin vel (body), base ang vel (body), projected gravity, command [vx,vy,wz],
            joint pos (rel default, add-order), joint vel, last action, gait phase (sin, cos)
  action (12): joint-target offset on the default stance, scaled by ACTION_SCALE
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from threepp.rl import GpuSim
from spot_deploy import default_q, add_to_isaac
from spot_gpu import SpotGpu, quat_rotate_inverse, flat_ground, SPACING, DT, CONTROL_HZ

SUBSTEPS = 6                # finer GPU physics than the stairs env's 4 — closes the GPU<->CPU transfer gap
ACTION_SCALE = 0.4          # joint authority (0.5 scooted; 0.3 suppressed the forward drive; 0.4 is the balance)
EPISODE_S = 12.0
COMMAND_HOLD_S = 3.0
MAX_SPEED = 1.2             # forward m/s ceiling — FAST (Spot's real top walk is ~1.6); the speed-scaled gait clock handles it
MAX_LAT = 0.4
MAX_YAW = 1.0              # rad/s
SETTLE_RESET = 12          # ticks holding the default stance after a reset
TARGET_H = 0.45            # nominal stand height (informational; the reward uses the H_LOW/H_HIGH band below)
OBS_DIM = 50               # +2 for the gait-phase clock (sin, cos) appended to the obs
ACT_DIM = 12
# GAIT-PHASE CLOCK (the technique that PRESCRIBES a trot instead of hoping penalties produce it).
# A periodic phase advances each step; the obs carries (sin,cos) of it; the reward pays each foot for
# matching a desired diagonal-trot CONTACT SCHEDULE (pair A = FL+HR down in the first half-cycle, up in
# the second; pair B = FR+HL opposite). Only enforced when commanded to MOVE (a stand command -> no schedule).
GAIT_FREQ = 2.4            # MAX trot cadence (Hz) at full command; the clock advances at freq*(command/MAX_SPEED)
W_GAIT = 1.5               # weight on matching the trot contact schedule (strong -> the trot dominates)
W_SPEED = 1.5              # NON-SATURATING pull toward the commanded forward speed: reward = min(vx, cmd_vx).
                          # The exp velocity-track basin goes flat far from a HIGH target (no gradient -> short
                          # strides at a fast cadence); this linear term keeps pulling -> learns to stride out FAST.
W_HEADING = 1.0            # penalty on heading error (yaw^2) -> walks a STRAIGHT line, not just zero yaw-rate
OVERLIFT_H = 0.17          # feet lifting above ~this are over-lifting toward a fly -> penalize (close that basin)
W_OVERLIFT = 1.0
GAIT_STANCE_H = 0.03       # a stance foot must be firmly PLANTED (tip_z below ~this) to score the schedule
GAIT_SWING_H = 0.12        # target swing clearance (m); reward caps here (bigger lift than 0.09 -> bigger steps)
TWO_PI = 6.283185307179586
# domain randomization — for a ROBUST gait that survives a different (CPU/real) solver, not just the training sim
OBS_NOISE = 0.02            # gaussian observation noise
INIT_JITTER = 0.06          # random joint offset at reset (rad) — small, so the stiff drive isn't kicked into a tip
PUSH_PROB = 0.005           # per-step chance of a random horizontal velocity shove (moderate -> learn balance, don't tip a learner)
PUSH_VEL = 0.6              # shove magnitude (m/s)
# contact friction (the new material API): grippy restitution-0 feet + ground so push-off doesn't slip and
# footfalls don't bounce (the 0.5/0.5/0.2 default did both). Per-env STATIC friction randomization (each env a
# fixed random foot mu for its lifetime) makes the gait robust to a different solver's effective friction.
FOOT_FRIC_MIN, FOOT_FRIC_MAX = 0.7, 1.2   # per-env static foot friction (real rubber ~1.0)
GROUND_FRIC = 1.4                          # ground friction; combine eMIN -> the foot's per-env mu governs the contact
# "learn to stand" via a STATIONARY command mix (not a ramping curriculum — the ramp made the reward
# non-stationary and PPO diverged). A fixed P_STAND fraction of commands is a pure stand (vx=0): the policy
# must hold upright + recover from shoves, so the dead-still deploy start (which made the old policy go
# passive and tip) is always in-distribution, AND walking is learned concurrently on a balancing base.
P_STAND = 0.30             # fraction of commands that are a pure stand (vx=0)
# base-height band — ASYMMETRIC with a FREE middle so a dynamic trot has vertical room (the v6 symmetric
# -12*(z-0.45)^2 pinned the base rigidly flat -> a stiff shuffle, per the gait review). Free in [H_LOW,H_HIGH];
# penalize crouch/fold below H_LOW AND a hop/launch above H_HIGH (keeps the anti-hop, allows the trot bounce).
H_LOW, H_HIGH = 0.42, 0.50
W_CROUCH, W_HOP = 15.0, 15.0
# foot sensing — the foot tip (for the gait contact schedule) is offset from the lleg link frame.
FOOT_TIP_OFFSET = (0.0, 0.178, 0.0)   # foot tip in the lleg link frame: +Y = knee->foot, len/2 + radius
CONTACT_H = 0.04            # height scale (m): a foot below ~this is "planted" (soft gate via exp)
W_DRAG = 0.4               # planted-foot slide penalty (raised: force a CLEAN plant+push, not a slide, when chasing speed)
TRACK_SIG = 0.15           # velocity-tracking sharpness (tighter -> the speed command is actually tracked; throttle responds)
# PD gains for the from-scratch gait. The Isaac default (stiffness 60) is too SOFT for our ~28 kg
# model: the legs sag ~12 deg (worst ~34 deg, the knees) under body weight and the body crouches at
# ~0.38 — so a lifted foot folds the sagging stance and only a scoot stays upright. Stiffness 140
# stands rigid (~6 deg sag, base ~0.47), which is what lets a real stepping gait form. (_probe_actuation.py)
STIFFNESS, DAMPING = 120.0, 4.0   # rigid enough (~7 deg sag) but gentler than 140; damping near-critical to not oscillate
WALK_GAINS = {"hx": (STIFFNESS, DAMPING, 45.0), "hy": (STIFFNESS, DAMPING, 45.0), "kn": (STIFFNESS, DAMPING, 115.0)}
# Spawn at the STIFF stance's natural height (feet on ground at the default pose). The stairs env's
# SPAWN_Z=0.42 was tuned for the soft-gain sag (~0.38); with stiff gains the rigid drive would launch
# the robot off the ground from 0.42 and tip it. ~0.47 = the measured rigid stand (_probe_actuation.py).
WALK_SPAWN_Z = 0.47

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "action_scale": ACTION_SCALE, "episode_s": EPISODE_S, "command_hold_s": COMMAND_HOLD_S,
          "max_speed": MAX_SPEED, "max_lat": MAX_LAT, "max_yaw": MAX_YAW, "target_h": TARGET_H,
          "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "spawn_z": WALK_SPAWN_Z,
          "stiffness": STIFFNESS, "damping": DAMPING}


class SpotWalkEnv:
    def __init__(self, num_envs=4096, device="cuda", seed=0, forward_only=True):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(EPISODE_S * CONTROL_HZ)
        self.command_hold = int(COMMAND_HOLD_S * CONTROL_HZ)
        self.forward_only = forward_only
        # per-env static foot friction (fixed random mu per env) + a grippy restitution-0 ground material
        self.foot_frics = np.random.default_rng(seed).uniform(FOOT_FRIC_MIN, FOOT_FRIC_MAX, num_envs).astype(np.float32)

        def _build_world(world):
            gmat = world.create_material(static_friction=GROUND_FRIC, dynamic_friction=GROUND_FRIC * 0.9,
                                         restitution=0.0, friction_combine="min", restitution_combine="min")
            flat_ground(world, num_envs, SPACING, material=gmat)

        self.sim = GpuSim(num_envs,
                          lambda world, i: SpotGpu(world, i, SPACING, gains=WALK_GAINS,
                                                   foot_friction=self.foot_frics[i]),
                          gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, read_root=True,
                          read_links=True,   # per-link kinematics -> foot clearance/slip (anti-shuffle)
                          build_world=_build_world)
        dev = self.sim.device
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.cmd_speed = MAX_SPEED    # walk-command ceiling (stationary; stand mix handled in _sample_cmd)
        self.stance = torch.from_numpy(default_q[add_to_isaac]).to(dev)        # [12] add-order default stand
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)

        lane_y = torch.arange(self.K, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(self.K, 3, device=dev); pos[:, 1] = lane_y; pos[:, 2] = WALK_SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        self.stand_q = self.stance.expand(self.K, -1).contiguous()

        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.cmd = z(self.K, 3)
        self.last_act = z(self.K, ACT_DIM)
        self.last_speed = 0.0; self.last_up = 0.0; self.last_climb = 0.0; self.last_fell = 0.0
        self.r_local = torch.tensor(FOOT_TIP_OFFSET, device=dev)
        self.foot_idx = None        # 4 foot links, auto-detected as the lowest links at the settled stand
        self.last_drag = 0.0; self.last_clear = 0.0
        self.phase = z(self.K)                                  # gait-phase clock [0,1) per env
        self.foot_pair = None                                   # [4] +1=pairA(FL,HR), -1=pairB(FR,HL); set at reset
        self.cur_yaw = z(self.K)                                # base heading (yaw) for the straight-line penalty
        self.last_gait = 0.0

    def _sample_cmd(self, n):
        dev = self.sim.device
        rnd = lambda: torch.rand(n, generator=self.g, device=dev)
        z = torch.zeros(n, device=dev)
        r = rnd()                                                # mutually-exclusive mode selector
        vx = rnd() * (MAX_SPEED - 0.2) + 0.2
        stand = r < P_STAND                                      # P_STAND: pure stand (vx=0) -> active balance
        vx = torch.where(stand, z, vx)
        if self.forward_only:
            return torch.stack([vx, z, z], dim=-1)
        # drive mix: 25% turning, 15% strafe, the rest straight forward (+ the stand fraction). Keeping most
        # commands straight (vy=wz=0) means the deploy heading-hold's small yaw command is well-trained.
        turn = (r >= P_STAND) & (r < P_STAND + 0.25)
        strafe = (r >= P_STAND + 0.25) & (r < P_STAND + 0.40)
        vy = torch.where(strafe, (rnd() - 0.5) * 2 * MAX_LAT, z)
        wz = torch.where(turn, (rnd() - 0.5) * 2 * MAX_YAW, z)
        return torch.stack([vx, vy, wz], dim=-1)

    def _frame(self):
        s = self.sim; q = s.root_quat
        return (quat_rotate_inverse(q, s.root_linvel),
                quat_rotate_inverse(q, s.root_angvel),
                quat_rotate_inverse(q, self.grav.expand(self.K, 3)))

    @staticmethod
    def _quat_rot(q, v):                       # rotate v [*,3] by quat q [*,4] (xyzw)
        qv, qw = q[..., :3], q[..., 3:4]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    def _foot_world(self):
        """Foot-tip world position [K,4,3] + velocity [K,4,3]. A planted foot is the pivot, so its tip
        velocity is the link velocity carried to the tip: v_tip = v_link + omega x (R * r_local)."""
        q = self.sim.link_pose[:, self.foot_idx, 0:4]
        p = self.sim.link_pose[:, self.foot_idx, 4:7]
        rw = self._quat_rot(q, self.r_local.expand_as(p))         # tip offset in world frame
        tip = p + rw
        tip_vel = (self.sim.link_linvel[:, self.foot_idx, :]
                   + torch.cross(self.sim.link_angvel[:, self.foot_idx, :], rw, dim=-1))
        return tip, tip_vel

    def _obs(self):
        lin_b, ang_b, proj_g = self._frame()
        qpos = self.sim.joint_pos - self.stand_q
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, self.sim.joint_vel, self.last_act], dim=1)
        obs = obs + torch.randn_like(obs) * OBS_NOISE           # observation noise (robustness)
        ph = self.phase[:, None] * TWO_PI                       # gait-phase clock (clean, no noise)
        return torch.cat([obs, torch.sin(ph), torch.cos(ph)], dim=1)   # [K, 50]

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        self.sim.set_root_state(idx, self.base_pose[idx])
        jitter = (torch.rand(n, self.sim.dof, generator=self.g, device=self.sim.device) - 0.5) * 2 * INIT_JITTER
        self.sim.set_joint_state(idx, self.stand_q[idx] + jitter,             # random initial pose (robustness)
                                 torch.zeros(n, self.sim.dof, device=self.sim.device))
        self.steps[idx] = 0
        self.last_act[idx] = 0.0
        self.cmd[idx] = self._sample_cmd(n)
        self.phase[idx] = torch.rand(n, generator=self.g, device=self.sim.device)   # random gait phase (decorrelate)

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        self.sim.read()
        for _ in range(SETTLE_RESET):                          # fold into the stance (default targets)
            self.sim.apply_drive_target(self.stand_q)
            for _ in range(SUBSTEPS):
                self.sim.step(DT / SUBSTEPS)
        if self.foot_idx is None:                              # 4 lowest links at the stand == the feet (lleg)
            self.foot_idx = torch.argsort(self.sim.link_pose[:, :, 6].mean(0))[:4]
            tip, _ = self._foot_world()                        # [K,4,3] foot tips (world)
            rel = (tip - self.sim.root_position[:, None, :]).mean(0)   # [4,3] body-relative (upright stand)
            self.foot_pair = torch.sign(rel[:, 0] * rel[:, 1])  # [4] +1=pairA(FL,HR), -1=pairB(FR,HL) diagonal trot
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        self.sim.apply_drive_target(self.stand_q + ACTION_SCALE * a)
        for _ in range(SUBSTEPS):
            self.sim.step(DT / SUBSTEPS)
        self.steps += 1
        lin_b, ang_b, proj_g = self._frame()
        zz = self.sim.root_position[:, 2]

        roll_c = (self.steps % self.command_hold == 0)
        ridx = torch.nonzero(roll_c, as_tuple=False).squeeze(-1)
        if ridx.numel() > 0:
            self.cmd[ridx] = self._sample_cmd(ridx.numel())
        if self.forward_only:                                  # HEADING-HOLD: command a yaw rate that steers
            q0 = self.sim.root_quat                            # back toward straight (+x) so it walks a straight
            yaw = torch.atan2(2.0 * (q0[:, 0] * q0[:, 1] + q0[:, 2] * q0[:, 3]),   # line, not just zero yaw-RATE
                              1.0 - 2.0 * (q0[:, 1] ** 2 + q0[:, 2] ** 2))         # (which drifts). cmd is in the obs.
            self.cmd[:, 2] = torch.clamp(-2.0 * yaw, -1.5, 1.5)
            self.cur_yaw = yaw                                  # keep for the heading-error penalty
        else:
            self.cur_yaw = torch.zeros(self.K, device=self.sim.device)
        # gait-phase clock advances PROPORTIONAL to the commanded speed: a 0 command freezes the phase ->
        # the policy STANDS; a bigger command -> faster cadence -> faster trot. The command is in the obs,
        # so the policy is speed-conditioned (and the deploy advances the phase by the same rule).
        move = self.cmd[:, 0].abs() + self.cmd[:, 1].abs() + 0.3 * self.cmd[:, 2].abs()
        freq = torch.clamp(move / MAX_SPEED, 0.0, 1.5) * GAIT_FREQ
        self.phase = (self.phase + freq * DT) % 1.0

        arate = a - self.last_act
        self.last_act = a
        up = -proj_g[:, 2]                                      # ~1 when upright
        tip, tip_vel = self._foot_world()
        tip_z = tip[:, :, 2]                                    # foot-tip heights [K,4]
        foot_spd = tip_vel[:, :, :2].norm(dim=-1)              # horizontal foot speed [K,4]
        contact = tip_z < CONTACT_H                            # planted foot [K,4]
        num_contact = contact.float().sum(1)                   # feet on the ground [K] (metric)
        drag = (foot_spd * torch.exp(-tip_z.clamp(min=0.0) / CONTACT_H)).sum(1)  # planted feet sliding [K]
        # GAIT-PHASE reward — PRESCRIBE the diagonal trot: pay each foot for matching the desired contact
        # schedule. pairA (FL,HR) DOWN in the first half-cycle (phase<0.5), lifted in the second; pairB
        # (FR,HL) opposite. STAND (~0 command, frozen clock) instead targets ALL FOUR feet DOWN: a trot has
        # no all-down phase, so freezing mid-cycle would strand two feet in the air -> the robot topples.
        in_first = (self.phase < 0.5).float()[:, None]                    # [K,1]
        pairA = (self.foot_pair > 0).float()[None, :]                     # [1,4] FL,HR
        trot_down = pairA * in_first + (1.0 - pairA) * (1.0 - in_first)   # [K,4] trot schedule (feet DOWN now)
        standing = (move < 0.12).float()[:, None]                         # [K,1] ~0 command -> plant all feet
        desired_down = standing + (1.0 - standing) * trot_down            # [K,4] stand: all DOWN; move: trot
        stance_score = torch.exp(-tip_z.clamp(min=0.0) / GAIT_STANCE_H)               # [K,4] ~1 firmly planted
        swing_score = (tip_z.clamp(0.0, GAIT_SWING_H) / GAIT_SWING_H)                 # [K,4] 0..1 for 0..9cm, capped
        gait_match = desired_down * stance_score + (1.0 - desired_down) * swing_score  # [K,4] grounded trot (no over-lift)
        r_gait = W_GAIT * gait_match.mean(1)                              # stand -> 4-foot stance; move -> diagonal trot
        r_lin = 1.5 * torch.exp(-((lin_b[:, 0] - self.cmd[:, 0]) ** 2 + (lin_b[:, 1] - self.cmd[:, 1]) ** 2) / TRACK_SIG)
        r_ang = 1.0 * torch.exp(-((ang_b[:, 2] - self.cmd[:, 2]) ** 2) / 0.25)   # tighter yaw-rate tracking (straighter)
        r_fast = W_SPEED * torch.minimum(lin_b[:, 0].clamp(min=0.0), self.cmd[:, 0].clamp(min=0.0))  # stride out to the command
        overlift = torch.relu(tip_z - OVERLIFT_H).sum(1)       # feet flying too high -> close the fly basin
        rew = (r_lin + r_ang + r_gait + r_fast
               + 0.3                                           # small alive
               - W_HEADING * self.cur_yaw ** 2                 # heading error -> walk a STRAIGHT line
               - W_CROUCH * torch.relu(H_LOW - zz) ** 2        # anti-crouch/fold
               - W_HOP * torch.relu(zz - H_HIGH) ** 2          # anti-hop; free band between = trot room
               - 1.0 * lin_b[:, 2] ** 2                        # damp vertical bounce
               - 0.05 * (ang_b[:, 0] ** 2 + ang_b[:, 1] ** 2)  # damp roll/pitch rate
               - 2.5 * (proj_g[:, 0] ** 2 + proj_g[:, 1] ** 2)  # stay upright
               - W_OVERLIFT * overlift                         # don't fly (over-lift penalty -> stable grounded trot)
               - 0.0005 * (self.sim.joint_vel ** 2).sum(1)     # joint velocity (smoothness; eased a touch for the faster trot)
               - 0.06 * arate.pow(2).sum(1)                    # action rate (much smoother targets -> kills the jitter)
               - W_DRAG * drag)                                # planted-foot slip (no slide)
        fell = (up < 0.4) | (zz < 0.30)                        # a deeply folded base is a fall
        rew = rew - 2.0 * fell.float()

        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self.sim.read()
            obs = self._obs()
        else:
            obs = term_obs
        self.last_speed = lin_b[:, 0].mean().item()            # mean forward speed (the learning signal to watch)
        self.last_up = up.mean().item()
        self.last_fell = fell.float().mean().item()
        self.last_drag = drag.mean().item()                    # planted-foot drag (lower = cleaner stepping)
        self.last_clear = tip_z.max(dim=1).values.mean().item()  # mean best-foot clearance (higher = real steps)
        self.last_gait = r_gait.mean().item()                  # trot-schedule match (toward W_GAIT as the trot locks in)
        self.last_nc = num_contact.mean().item()               # feet on the ground (~2 = trot)
        # random shoves: perturb the base velocity of a few envs so the gait must RECOVER -> robust/transferable
        pmask = torch.rand(self.K, generator=self.g, device=self.sim.device) < PUSH_PROB
        pidx = torch.nonzero(pmask, as_tuple=False).squeeze(-1)
        if pidx.numel() > 0:
            lv = self.sim.root_linvel[pidx].clone()
            lv[:, :2] += (torch.rand(pidx.numel(), 2, generator=self.g, device=self.sim.device) - 0.5) * 2 * PUSH_VEL
            self.sim.set_root_state(pidx, self.sim.root_pose[pidx], lv, self.sim.root_angvel[pidx])
            self.sim.read()
        return obs, rew, done, term_obs, timeout


if __name__ == "__main__":
    import threepp as tp
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = SpotWalkEnv(num_envs=int(os.environ.get("K", "64")))
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    for _ in range(120):                                       # zero action -> should just STAND
        obs, rew, done, term, to = env.step(torch.zeros(env.K, ACT_DIM, device=env.sim.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action stand: fwd_speed={env.last_speed:+.2f} m/s  up={env.last_up:.2f}  "
          f"base_z={env.sim.root_position[:, 2].mean():.3f}  rew={rew.mean():+.2f}")
    print(f"foot links (auto-detected) = {env.foot_idx.tolist()}  "
          f"drag={env.last_drag:.3f} (should be ~0 standing)  clearance={env.last_clear:.3f}")
    print("SPOT-WALK ENV SELFTEST: PASS")
